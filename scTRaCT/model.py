import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiAttention(nn.Module):
    def __init__(self, num_heads, embedding_dim, dropout=0.2):
        super(MultiAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads
        self.WQ = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.WK = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.WV = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        queries = self.WQ(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.WK(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.WV(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, values)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.dropout(attention_output)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class ResidualConnection(nn.Module):
    def __init__(self, size, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return x + self.dropout(self.norm(sublayer_output))

class TransformerModel(nn.Module):
    def __init__(self, num_genes, num_classes, num_heads=8, dim_feedforward=2048, dropout=0.1, embedding_dim=1024):
        super(TransformerModel, self).__init__()
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.count_embedding = nn.Linear(num_genes, embedding_dim)
        self.dist_embedding = nn.Linear(num_genes, embedding_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim * 2))
        self.norm1 = nn.LayerNorm(embedding_dim * 2)
        self.norm2 = nn.LayerNorm(embedding_dim * 2)
        self.norm3 = nn.LayerNorm(embedding_dim * 2)
        self.multiattention1 = MultiAttention(num_heads, embedding_dim * 2, dropout)
        self.residual_connection1 = ResidualConnection(embedding_dim * 2, dropout)
        self.multiattention2 = MultiAttention(num_heads, embedding_dim * 2, dropout)
        self.residual_connection2 = ResidualConnection(embedding_dim * 2, dropout)
        self.multiattention3 = MultiAttention(num_heads, embedding_dim * 2, dropout)
        self.residual_connection3 = ResidualConnection(embedding_dim * 2, dropout)
        self.ffn1 = nn.Linear(embedding_dim * 2, dim_feedforward)
        self.ffn2 = nn.Linear(dim_feedforward, embedding_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.residual_feedforward = ResidualConnection(embedding_dim * 2, dropout)
        self.fc_out = nn.Linear(embedding_dim * 2, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def feedforward(self, x):
        out = F.gelu(self.ffn1(x))
        out = self.ffn2(self.dropout(out))
        return out

    def forward(self, x_counts, x_dist):
        count_embed = self.count_embedding(x_counts)
        dist_embed = self.dist_embedding(x_dist)
        x = torch.cat((count_embed, dist_embed), dim=-1).unsqueeze(1)
        cls_tokens = self.cls_token.repeat(x.size(0), 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.norm1(x)
        x = self.residual_connection1(x, self.multiattention1(x))
        x = self.norm2(x)
        x = self.residual_connection2(x, self.multiattention2(x))
        x = self.norm3(x)
        x = self.residual_connection3(x, self.multiattention3(x))
        x = self.residual_feedforward(x, self.feedforward(x))
        return self.fc_out(x[:, 0])

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_factor = self.alpha[targets]
            focal_weight *= alpha_factor
        focal_loss = focal_weight * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
