import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()

        self.d_out = d_out
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
             torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch, num_tokens, d_in = x.shape
        key = self.W_k(x)
        query = self.W_q(x)
        value = self.W_v(x)

        attention_scores = query @ key.transpose(1, 2)
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attention_weights = torch.softmax(attention_scores / key.shape[-1] ** 0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vec = attention_weights @ value
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)]
        )
        self.out_proj = nn.Linear(d_out * num_heads, d_out * num_heads)

    def forward(self, x):
        context_vec = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.out_proj(context_vec)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch, num_tokens, d_in = x.shape

        key = self.W_k(x)  # shape: (batch, num_tokens, d_out)
        query = self.W_q(x)
        value = self.W_v(x)

        key = key.view(batch, num_tokens, self.num_heads, self.head_dim)
        value = value.view(batch, num_tokens, self.num_heads, self.head_dim)
        query = query.view(batch, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query = query.transpose(1, 2)

        attention_scores = query @ key.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attention_scores.masked_fill_(mask_bool, -torch.inf)

        attention_weight = torch.softmax(attention_scores / key.shape[-1] ** 0.5, dim=-1)
        attention_weight = self.dropout(attention_weight)

        # Shape: (batch, num_tokens, num_heads, head_dim)
        context_vec = (attention_weight @ value).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(batch, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
