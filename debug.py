# A debug file made in cooperation with a modern LLM to take my not-yet-functional code and figure out where it was diverging from hf's implementation
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
    print(f"Custom Embeddings shape: {embeddings.shape}")
    print(f"Custom Embeddings sum: {embeddings.sum().item()}")
    
    position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device).unsqueeze(0)
    embeddings += model.transformer.wpe(position_ids)
    print(f"Custom Embeddings + Positional shape: {embeddings.shape}")
    print(f"Custom Embeddings + Positional sum: {embeddings.sum().item()}")

    hidden_states = embeddings
    for i in range(num_layers):
        layer = model.transformer.h[i]
        hidden_states = process_attention_sublayer(hidden_states, layer, num_heads=12, d_model=768)
        hidden_states = process_feedforward_sublayer(hidden_states, layer, d_model=768)
        print(f"Custom Layer {i} output shape: {hidden_states.shape}")
        print(f"Custom Layer {i} output sum: {hidden_states.sum().item()}")

    hidden_states = normalize_sublayer_input(hidden_states, model.transformer.ln_f.weight, model.transformer.ln_f.bias)
    print(f"Custom Final layer norm output shape: {hidden_states.shape}")
    print(f"Custom Final layer norm output sum: {hidden_states.sum().item()}")

    lm_logits = hidden_states @ model.lm_head.weight.T
    print(f"Custom Logits shape: {lm_logits.shape}")
    print(f"Custom Logits sum: {lm_logits.sum().item()}")

    return lm_logits

def process_attention_sublayer(input, layer, num_heads, d_model):
    qkv_weight, qkv_bias, Wo, Wo_bias, ln_1_gamma, ln_1_beta, *_ = load_layer_weights(layer)
    
    normalized_input = normalize_sublayer_input(input, ln_1_gamma, ln_1_beta)
    print(f"Custom Attention normalized input sum: {normalized_input.sum().item()}")
    
    qkv = normalized_input @ qkv_weight + qkv_bias
    qkv = qkv.view(input.size(0), input.size(1), 3, num_heads, d_model // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_model // num_heads)
    
    # Add attention mask for causal attention
    mask = torch.tril(torch.ones(input.size(1), input.size(1), device=input.device)).view(1, 1, input.size(1), input.size(1))
    attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
    
    attn_weights = F.softmax(attn_weights, dim=-1)
    
    context = torch.matmul(attn_weights, v)
    context = context.permute(0, 2, 1, 3).contiguous().view(input.size(0), input.size(1), d_model)
    
    attn_output = context @ Wo + Wo_bias
    output = input + attn_output
    
    print(f"Custom Attention output sum: {output.sum().item()}")
    return output

def process_feedforward_sublayer(input, layer, d_model):
    _, _, _, _, _, _, ff1_weight, ff1_bias, ff2_weight, ff2_bias, ln_2_gamma, ln_2_beta = load_layer_weights(layer)
    
    normalized_input = normalize_sublayer_input(input, ln_2_gamma, ln_2_beta)
    
    hidden_states = normalized_input @ ff1_weight + ff1_bias
    hidden_states = gelu_new(hidden_states)
    hidden_states = hidden_states @ ff2_weight + ff2_bias
    
    output = input + hidden_states
    print(f"Custom FF output sum: {output.sum().item()}")
    return output

def normalize_sublayer_input(input, gamma, beta):
    mean = input.mean(dim=-1, keepdim=True)
    var = input.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (input - mean) / torch.sqrt(var + epsilon)
    output = gamma * normalized + beta
    return output

def get_next_token(logits, tokenizer):
    last_token_logits = logits[0, -1, :]
    probabilities = F.softmax(last_token_logits, dim=-1)
    next_token_id = torch.argmax(probabilities)
    next_token = tokenizer.decode(next_token_id)
    return next_token

def prepare_input(text, model, tokenizer, max_length=1024):
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=max_length, truncation=True)
    return input_ids.squeeze(0)

def huggingface_inference(input_ids, model):
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    
    for i, hidden_state in enumerate(outputs.hidden_states):
        print(f"HF Layer {i} output shape: {hidden_state.shape}")
        print(f"HF Layer {i} output sum: {hidden_state.sum().item()}")
    
    print(f"HF Logits shape: {outputs.logits.shape}")
    print(f"HF Logits sum: {outputs.logits.sum().item()}")
    
    return outputs.logits

def compare_implementations(input_text):
    input_ids = prepare_input(input_text, model, tokenizer)
    
    print("Custom implementation:")
    your_logits = gpt2_inference(input_ids, model)
    
    print("\nHugging Face implementation:")
    hf_logits = huggingface_inference(input_ids.unsqueeze(0), model)
    
    print("\nComparison:")
    print(f"Logits shape match: {your_logits.shape == hf_logits.shape}")
    print(f"Logits sum difference: {abs(your_logits.sum().item() - hf_logits.sum().item())}")
    print(f"Logits max difference: {(your_logits - hf_logits).abs().max().item()}")

compare_implementations("Road road road road")