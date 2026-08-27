import numpy as np
import torch
import torch.nn as nn

def get_timestep_embedding(embed_dim, timesteps, device):
        half_dim = embed_dim // 2
        scale = torch.log(torch.tensor(1000, device=device)) / (half_dim - 1)
        exponents = torch.exp(-scale * torch.arange(half_dim, device=device))
        emb = timesteps[:, None] * exponents[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, embed_dim)
        return emb

class Projection(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.SiLU(),
            nn.Linear(d_out, d_out)
        )
    def forward(self, x):
        return self.projection(x)

class DenoiserTransformerFull(nn.Module):
    def __init__(self, dim_hidden, dim_timestep_embed, num_points=689, global_feature_dim=256, point_feature_dim=128, num_layers=4, nhead=8, dim_feedforward=512):
        super().__init__()
        
        self.num_points = num_points
        self.dim_timestep_embed = dim_timestep_embed
        self.projection_points = Projection(3, dim_hidden)
        self.projection_global = Projection(global_feature_dim, dim_hidden)
        self.projection_point = Projection(point_feature_dim, dim_hidden)
        self.projection_T = Projection(dim_timestep_embed, dim_hidden)
        self.layernorm = nn.LayerNorm(dim_hidden)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_hidden, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder_points = Projection(dim_hidden, 3)

    def get_sinusoidal_positional_encoding(self, seq_len, embed_dim, device):
        position = torch.arange(seq_len, device=device).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device).float() * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
        
    def forward(self, points, global_feat, point_feat, t):
        device = points.device
        B = points.shape[0]
        points_emb = self.projection_points(points)

        pos_enc = self.get_sinusoidal_positional_encoding(self.num_points, self.layernorm.normalized_shape[0], device)
        pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1)
        points_emb = points_emb + pos_enc

        point_emb = self.projection_point(point_feat)
        point_tokens = points_emb + point_emb
        global_token = self.projection_global(global_feat).unsqueeze(1)
        t_emb = get_timestep_embedding(self.dim_timestep_embed, t, device)
        time_token = self.projection_T(t_emb).unsqueeze(1)
        x = torch.cat([point_tokens, global_token, time_token], dim=1)
        x = self.layernorm(x)
        x = self.transformer_encoder(x)
        points_out = self.decoder_points(x[:, :self.num_points])
        return points_out
