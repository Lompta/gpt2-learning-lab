# File for various implementations I did for learning, but weren't efficient/shaped quite right for the big show
from math import log, sqrt, sum, e
import torch

epsilon = 1e-5

def positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    # exp and log cancel out - good for numerical stability
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

## File for inefficient or irrelevant implementations written to strengthen my understanding
def normalize_sublayer_input_scraps(input, gamma, beta):
    output = []
    for row_index in range(input):
        new_row = []
        row_mean = torch.mean(input[row_index])
        row_variance = torch.var(input[row_index])
        for column_index in range(input[row_index]):
            new_row.append(((input[row_index][column_index] - row_mean)/(sqrt(row_variance + epsilon))) * gamma[column_index] + beta[column_index])
        output.append(new_row)

    return output

def row_wise_softmax_scraps(input):
    output = []
    for row_index in range(input):
        new_row = []
        for column_index in range(input[row_index]):
            new_row.append(e ** input[row_index][column_index])
        normalize_factor = sum(new_row)
        # bad syntax - this might not do scalar multiplication
        normalized_new_row = new_row * (1.0/normalize_factor)
        output.append(normalized_new_row)
    return output

# GPT-2 actually uses GELU, but I want to capture how ReLU works as part of this learning project
def apply_ReLU(input):
    if input < 0:
        return 0
    else:
        return input
