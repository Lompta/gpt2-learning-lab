# My second attempt - I leaned more on modern LLM help to make sure the practices aligned with GPT-2's particular quirks/optimizations
# Comments are generally mine; I commented to make sure I knew which code artifacts corresponded to which steps I'd studied in autoregressive transformers architecture

# It works!
import torch
from torch.nn import functional as F
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pretrained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
epsilon = 1e-5

def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def load_layer_weights(layer):
    qkv_weight = layer.attn.c_attn.weight
    qkv_bias = layer.attn.c_attn.bias
    
    Wo = layer.attn.c_proj.weight
    Wo_bias = layer.attn.c_proj.bias
    
    ln_1_gamma = layer.ln_1.weight
    ln_1_beta = layer.ln_1.bias
    
    ff1_weight = layer.mlp.c_fc.weight
    ff1_bias = layer.mlp.c_fc.bias
    
    ff2_weight = layer.mlp.c_proj.weight
    ff2_bias = layer.mlp.c_proj.bias
    
    ln_2_gamma = layer.ln_2.weight
    ln_2_beta = layer.ln_2.bias
    
    return (qkv_weight, qkv_bias, Wo, Wo_bias, ln_1_gamma, ln_1_beta,
            ff1_weight, ff1_bias, ff2_weight, ff2_bias, ln_2_gamma, ln_2_beta)

def gpt2_inference(input_ids, model, num_layers=12):
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    embeddings = model.transformer.wte(input_ids)
    
    ## add positional encodings
    position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device).unsqueeze(0)
    embeddings += model.transformer.wpe(position_ids)

    ## move through layers
    hidden_states = embeddings
    for i in range(num_layers):
        layer = model.transformer.h[i]
        hidden_states = process_attention_sublayer(hidden_states, layer, num_heads=12, d_model=768)
        hidden_states = process_feedforward_sublayer(hidden_states, layer)

    # since GPT-2 pre-normalizes, we have to do one last normalization explicitly
    hidden_states = normalize_sublayer_input(hidden_states, model.transformer.ln_f.weight, model.transformer.ln_f.bias)

    # map input_size x hidden_dimension to input_size x vocabulary_dimension
    lm_logits = hidden_states @ model.lm_head.weight.T

    return lm_logits

def process_attention_sublayer(input, layer, num_heads, d_model):
    qkv_weight, qkv_bias, Wo, Wo_bias, ln_1_gamma, ln_1_beta, *_ = load_layer_weights(layer)
    
    # since GPT-2 pre-normalizes, we normalize here instead of at the end
    normalized_input = normalize_sublayer_input(input, ln_1_gamma, ln_1_beta)
    
    # compute the q, k, v matrices all at once in a big blob 
    qkv = normalized_input @ qkv_weight + qkv_bias
    # we permute to get the resulting tensor in a reasonable order to break apart into q, k, and v separately (eg. the dimension "3" should be first)
    qkv = qkv.view(input.size(0), input.size(1), 3, num_heads, d_model // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    # multiply q by the transpose of k, as is standard attention layer practice
    attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_model // num_heads)
    
    # Add attention mask for causal attention - no clairvoyance!
    mask = torch.tril(torch.ones(input.size(1), input.size(1), device=input.device)).view(1, 1, input.size(1), input.size(1))
    attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
    
    # softmax the weights row-wise then apply the result to v (across all heads)
    attn_weights = F.softmax(attn_weights, dim=-1)
    context = torch.matmul(attn_weights, v)

    # concatenate the heads horizontally so we can linearly transform by the output projection
    context = context.permute(0, 2, 1, 3).contiguous().view(input.size(0), input.size(1), d_model)
    
    # you guessed it - linearly transforming by the output projection
    attn_output = context @ Wo + Wo_bias
    output = input + attn_output
    
    return output

def process_feedforward_sublayer(input, layer):
    _, _, _, _, _, _, ff1_weight, ff1_bias, ff2_weight, ff2_bias, ln_2_gamma, ln_2_beta = load_layer_weights(layer)
    
    # we do pre-normalization in GPT-2
    normalized_input = normalize_sublayer_input(input, ln_2_gamma, ln_2_beta)
    
    # multiply by embiggening matrix to feedforward dimension
    hidden_states = normalized_input @ ff1_weight + ff1_bias
    # apply gelu (relu's more sophisticated big brother) for nonlinearity
    hidden_states = gelu_new(hidden_states)
    # multiply by ensmallening matrix to get back to input_length x embedding_dimension
    hidden_states = hidden_states @ ff2_weight + ff2_bias
    
    # residual connection - no normalization here since we do it at the beginning
    output = input + hidden_states
    return output

def normalize_sublayer_input(input, gamma, beta):
    mean = input.mean(dim=-1, keepdim=True)
    var = input.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (input - mean) / torch.sqrt(var + epsilon)
    # layer-based learned parameters time!
    output = gamma * normalized + beta
    return output

def get_next_token(logits, tokenizer):
    # each row of logits is softmaxed to predict the following token - so we want the last row of logits
    last_token_logits = logits[0, -1, :]
    # softmax to turn logits (somewhat arbitrary real numbers) into a probability distribution
    probabilities = F.softmax(last_token_logits, dim=-1)
    # in this implementation we're just getting the single most likely next token - ie. temperature is 0
    next_token_id = torch.argmax(probabilities)
    # figure out what letters correspond to the argmaxed id and return them
    next_token = tokenizer.decode(next_token_id)
    return next_token

def prepare_input(text, tokenizer, max_length=1024):
    # break the text up into tokens then figure out what numbers correspond to those tokens
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=max_length, truncation=True)
    # make that list of numbers a vector - we'll fetch a row from the embeddings matrix for each
    return input_ids.squeeze(0)

# Do all this under the hood in huggingface's implementation
def huggingface_inference(input_ids, model, tokenizer):
    with torch.no_grad():
        outputs = model(input_ids)
    
    logits = outputs.logits
    next_token_logits = logits[0, -1, :]
    next_token_id = torch.argmax(next_token_logits)
    next_token = tokenizer.decode(next_token_id)
    
    return next_token

# My implementation should be functionally identical to hf's
def compare_implementations(input_text):
    input_ids = prepare_input(input_text, tokenizer)

    # Your implementation
    your_logits = gpt2_inference(input_ids, model)
    your_next_token = get_next_token(your_logits, tokenizer)

    # Hugging Face implementation
    hf_next_token = huggingface_inference(input_ids.unsqueeze(0), model, tokenizer)


    print(f"Input text: '{input_text}'")
    print(f"Your implementation's next token: '{your_next_token}'")
    print(f"Hugging Face implementation's next token: '{hf_next_token}'")
    print(f"Match: {your_next_token == hf_next_token}")

compare_implementations("1 2 3 4 5 6 7")