import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):  # h: number of heads, d_model: model dimension
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2).squeeze(-1)  # Ensure the mask shape is [batch_size, 1, 1, seq_len]
            #print(f"Shape of mask: {mask.shape}")

        nbatches = query.size(0)
        seq_len = query.size(1)  # Explicitly get the sequence length to use in reshaping

        # Apply linear layers, split heads, and reshape
        query, key, value = [
            l(x).view(nbatches, seq_len, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        #print(f"Before attention: query shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}")

        # Perform attention
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # Reshape back to [batch_size, seq_len, d_model] explicitly
        x = x.transpose(1, 2).contiguous().view(nbatches, seq_len, self.h * self.d_k)
        return self.linears[-1](x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -torch.tensor(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features)) #Multiplied
        self.b_2 = nn.Parameter(torch.zeros(features)) #Added
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        normed_x = self.norm(x)
        sublayer_output = sublayer(normed_x)
        #print(f"x shape: {x.shape}, sublayer output shape: {sublayer_output.shape}")
        return x + self.dropout(sublayer_output)

class SrcEmbed(nn.Module):
    def __init__(self, input_dim, d_model): 
        super(SrcEmbed, self).__init__()
        self.w = nn.Linear(input_dim, d_model)
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.w(x))


# Final layer for the Transformer
class TranFinalLayer(nn.Module):
    def __init__(self, d_model, num_classes):
        super(TranFinalLayer, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model // 2)
        self.norm = LayerNorm(d_model // 2)
        self.w_2 = nn.Linear(d_model // 2, num_classes)  # Use num_classes for flexibility

    def forward(self, x):
        x = F.relu(self.w_1(x))
        x = self.norm(x)
        x = self.w_2(x)
      # Softmax for probabilities
        #predicted_class = torch.argmax(probabilities, dim=-1)+1
        return F.softmax(x, dim=-1)
'''class TranFinalLayer(nn.Module):
    def __init__(self, d_model, num_classes):
        super(TranFinalLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_model // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=d_model // 2, out_channels=num_classes, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pooling to reduce sequence length to 1
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change from (batch, seq_len, d_model) to (batch, d_model, seq_len) for CNN
        x = F.relu(self.conv1(x))
        x = self.pool(self.conv2(x)).squeeze(-1)  # Apply pooling and remove extra dimension
        return F.softmax(x, dim=-1)  # Output probabilities'''



class Encoder(nn.Module):
    def __init__(self, layer, N, d_model, dropout, num_features,num_classes):
        super(Encoder, self).__init__()
        self.src_embed = SrcEmbed(num_features, d_model)
        self.position_encode = PositionalEncoding(d_model, dropout)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.final_layer = TranFinalLayer(d_model,num_classes)
        
    def forward(self, x, mask=None):
        x = self.position_encode(self.src_embed(x))
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_layer(x)
'''class Encoder(nn.Module):
    def __init__(self, layer, N, d_model, dropout, num_features, num_classes):
        super(Encoder, self).__init__()
        self.src_embed = SrcEmbed(num_features, d_model)
        self.position_encode = PositionalEncoding(d_model, dropout)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.final_layer = TranFinalLayer(d_model, num_classes)
        
    def forward(self, x, mask=None):
        x = self.position_encode(self.src_embed(x))
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_layer(x)'''



class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        #print(f"Before self-attention: x shape: {x.shape}")
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        #print(f"After self-attention: x shape: {x.shape}")
        return self.sublayer[1](x, self.feed_forward)

