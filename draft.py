# My first attempt - I did a lot of the math naively, which made it difficult to do things like input masking
# For example, I expected Wq, Wk, and Wv to be stored separately, rather than all attention heads processed in batch

# The wonders of empiricism vs. pure theory!
# Anyway, it doesn't really work, and recommends worse tokens than main.py and (identically, as far as I've tested) hf's implementation.
import torch
from torch.nn import functional as F
from math import log, sqrt, e
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pretrained model and tokenizer
model_name = "gpt2"  # This is the smallest GPT-2 model
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
epsilon = 1e-5

def gpt2_inference(input_ids, model, num_layers=12):
    # Get initial embeddings
    embeddings = model.transformer.wte(input_ids)
    
    # Add positional encodings
    position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device)
    embeddings += model.transformer.wpe(position_ids)

    # Process through layers
    hidden_states = embeddings
    for i in range(num_layers):
        layer = model.transformer.h[i]
        hidden_states = process_attention_sublayer(hidden_states, layer, num_heads=12, d_model=768)
        hidden_states = process_feedforward_sublayer(hidden_states, layer, d_model=768)

    # Final layer norm
    hidden_states = normalize_sublayer_input(hidden_states, model.transformer.ln_f.weight, model.transformer.ln_f.bias)

    # Final linear transformation to vocabulary size
    lm_logits = hidden_states @ model.lm_head.weight.T

    return lm_logits

def get_next_token(logits, tokenizer):
    last_token_logits = logits[-1, :]
    probabilities = F.softmax(last_token_logits, dim=-1)
    next_token_id = torch.argmax(probabilities)
    next_token = tokenizer.decode(next_token_id)
    return next_token

def prepare_input(text, model, tokenizer, max_length=1024):
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=max_length, truncation=True)
    return input_ids.squeeze(0)

def get_attention_weights(layer):
    qkv_weight = layer.attn.c_attn.weight
    qkv_bias = layer.attn.c_attn.bias

    hidden_size = 768
    num_heads = 12  # for GPT-2 small
    head_size = hidden_size // num_heads

    # Split weights
    qkv_weight = qkv_weight.view(num_heads, 3, head_size, hidden_size)
    Wq = qkv_weight[:, 0]  # Shape: [num_heads, head_size, hidden_size]
    Wk = qkv_weight[:, 1]
    Wv = qkv_weight[:, 2]

    # Split biases
    qkv_bias = qkv_bias.view(num_heads, 3, head_size)
    q_bias = qkv_bias[:, 0]  # Shape: [num_heads, head_size]
    k_bias = qkv_bias[:, 1]
    v_bias = qkv_bias[:, 2]

    # Output projection
    Wo = layer.attn.c_proj.weight
    Wo_bias = layer.attn.c_proj.bias

    # Layer norm parameters
    ln_1_gamma = layer.ln_1.weight
    ln_1_beta = layer.ln_1.bias

    return Wq, Wk, Wv, q_bias, k_bias, v_bias, Wo, Wo_bias, ln_1_gamma, ln_1_beta

def get_feedforward_weights(layer):
    # First linear layer (expansion)
    ff1_weight = layer.mlp.c_fc.weight
    ff1_bias = layer.mlp.c_fc.bias

    # Second linear layer (projection)
    ff2_weight = layer.mlp.c_proj.weight
    ff2_bias = layer.mlp.c_proj.bias

    # Layer normalization parameters
    ln_2_gamma = layer.ln_2.weight
    ln_2_beta = layer.ln_2.bias

    return ff1_weight, ff1_bias, ff2_weight, ff2_bias, ln_2_gamma, ln_2_beta

def process_attention_sublayer(input, layer, num_heads, d_model):
    Wq, Wk, Wv, q_bias, k_bias, v_bias, Wo, Wo_bias, ln_1_gamma, ln_1_beta = get_attention_weights(layer)

    # normalize first, since GPT-2 does pre-normalization
    normalized_input = normalize_sublayer_input(input, ln_1_gamma, ln_1_beta)
    attention_head_outputs = []

    for head in range(num_heads):
        Wq_head = Wq[head]  # Shape: [head_size, hidden_size]
        Wk_head = Wk[head]  # Shape: [head_size, hidden_size]
        Wv_head = Wv[head]  # Shape: [head_size, hidden_size]
        
        q_bias_head = q_bias[head]  # Shape: [head_size]
        k_bias_head = k_bias[head]  # Shape: [head_size]
        v_bias_head = v_bias[head]  # Shape: [head_size]
        
        Q = (normalized_input @ (Wq_head.T)) + q_bias_head
        K = (normalized_input @ (Wk_head.T)) + k_bias_head
        V = (normalized_input @ (Wv_head.T)) + v_bias_head

        raw_attention_scores = Q @ (K.T)
        scaled_attention_scores = raw_attention_scores * (1/sqrt(d_model/num_heads))

        softmaxed_scaled_attention_scores = F.softmax(scaled_attention_scores, dim=-1)
        attention_head_output = softmaxed_scaled_attention_scores @ V
        attention_head_outputs.append(attention_head_output)

    concatenated_attention_head_outputs = torch.hstack(attention_head_outputs)
    # TODO: verify whether this should, indeed, be transposed
    output_projection_transformed_output = (concatenated_attention_head_outputs @ Wo.T) + Wo_bias

    # residual connection
    sublayer_output = output_projection_transformed_output + input

    return sublayer_output

def process_feedforward_sublayer(input, layer, d_model):
    ff1_w, ff1_b, ff2_w, ff2_b, ln_2_g, ln_2_b = get_feedforward_weights(layer)
    normalized_input = normalize_sublayer_input(input, ln_2_g, ln_2_b)

    # weights are stored transposed for efficiency - transposition isn't integral to the flow here like in the attention sublayer
    first_transformation_output = (normalized_input @ ff1_w) + ff1_b
    activated_first_transformation_output = F.gelu(first_transformation_output)

    # weights are stored transposed for efficiency - transposition isn't integral to the flow here like in the attention sublayer
    second_transformation_output = (activated_first_transformation_output @ ff2_w) + ff2_b
    output = second_transformation_output + input

    return output

def normalize_sublayer_input(input, gamma, beta):
    # Calculate mean and variance along the last dimension
    mean = torch.mean(input, dim=-1, keepdim=True)
    var = torch.var(input, dim=-1, keepdim=True, unbiased=False)
    
    # Normalize
    normalized = (input - mean) / torch.sqrt(var + epsilon)
    
    # Scale and shift
    output = gamma * normalized + beta
    
    return output

## For getting the shape of the model
def report_weight_shapes():
    # Access model weights
    for name, param in model.named_parameters():
        print(f"Layer: {name}, Shape: {param.shape}")

    # Example of accessing specific layers
    embedding_weight = model.transformer.wte.weight
    attention_weight = model.transformer.h[0].attn.c_attn.weight

    print(f"Embedding weight shape: {embedding_weight.shape}")
    print(f"First attention layer weight shape: {attention_weight.shape}")

## For benchmarking
def huggingface_inference(input_text, model, tokenizer):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits
    next_token_id = torch.argmax(logits[0, -1, :])
    next_token = tokenizer.decode(next_token_id)
    return next_token

def compare_implementations(input_text):
    # Your implementation
    input_ids = prepare_input(input_text, model, tokenizer)
    your_logits = gpt2_inference(input_ids, model)
    your_next_token = get_next_token(your_logits, tokenizer)

    # Hugging Face implementation
    hf_next_token = huggingface_inference(input_text, model, tokenizer)

    print(f"Input text: '{input_text}'")
    print(f"Your implementation's next token: '{your_next_token}'")
    print(f"Hugging Face implementation's next token: '{hf_next_token}'")
    print(f"Match: {your_next_token == hf_next_token}")

# input_ids = prepare_input("Four score and ", model, tokenizer)
# logits = gpt2_inference(input_ids, model)
# next_token = get_next_token(logits, tokenizer)
# print("the next token is: ", next_token)

compare_implementations("Road road road road road road road")