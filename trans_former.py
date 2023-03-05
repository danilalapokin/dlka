import torch
from torch import Tensor
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, len: int = 5000):
        super(PositionalEncoding, self).__init__()
        denc = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, len).reshape(len, 1)
        pos_embedding = torch.zeros((len, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * denc)
        pos_embedding[:, 1::2] = torch.cos(pos * denc)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class Tembedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(Tembedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class Transformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, e_size, nhead, sv_size, tv_size, dim_feedforward = 512, dropout = 0.2):
        super(Transformer, self).__init__()
        self.transformer = nn.Transformer(d_model=e_size, nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward, dropout=dropout)
        self.layer = nn.Linear(e_size, tv_size)
        self.x_emb = Tembedding(sv_size, e_size)
        self.target_emb = Tembedding(tv_size, e_size)
        self.positional_encoding = PositionalEncoding(e_size, dropout=dropout)

    def forward(self, x, target, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, mask):
        x_emb = self.positional_encoding(self.x_emb(x))
        target_emb = self.positional_encoding(self.target_emb(target))
        outs = self.transformer(x_emb, target_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, mask)
        return self.layer(outs)

    def encode(self, x, x_mask):
        return self.transformer.encoder(self.positional_encoding(self.x_emb(x)), x_mask)

    def decode(self, target, memory: Tensor, target_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.target_emb(target)), memory,target_mask)
def create_mask(x, target, device):
    x_len = x.shape[0]
    target_len = target.shape[0]

    target_mask = (torch.triu(torch.ones((target_len, target_len), device=device)) == 1).transpose(0, 1)
    target_mask = target_mask.float().masked_fill(target_mask == 0, float('-inf')).masked_fill(target_mask == 1, float(0.0))
    x_mask = torch.zeros((x_len, x_len),device=device).type(torch.bool)

    x_padding_mask = (x == 1).transpose(0, 1)
    target_padding_mask = (target == 1).transpose(0, 1)
    return x_mask, target_mask, x_padding_mask, target_padding_mask