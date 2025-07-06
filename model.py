import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import vendi_score, avg_cosine



class MultiheadSelfAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout=0.1, max_seq_len=376):
        super().__init__()
        assert emb_size % num_heads == 0
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        self.qkv_proj = nn.Linear(emb_size, 3 * emb_size)
        self.out_proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

        self.rope = Rotary1DPositionalEncoding(dim=self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x):
        B, L, C = x.shape 
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, self.num_heads, self.head_dim)
        k = k.view(B, L, self.num_heads, self.head_dim)
        v = v.view(B, L, self.num_heads, self.head_dim)

        q, k = self.rope(q, k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, C)
        return self.out_proj(attn_output)




class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size=512, num_heads=8, mlp_dim=1024, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_size)
        self.attn = MultiheadSelfAttention(emb_size, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_size),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SpeechSim(nn.Module):
    def __init__(self, emb_size=512, depth=12, num_heads=8, mlp_dim=1024, out_proj_dim=192):
        super().__init__()
        self.patch_embed = nn.Linear(156, emb_size, bias=False)
        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(emb_size, num_heads, mlp_dim) for _ in range(depth)
        ])
        self.head = nn.Linear(emb_size, out_proj_dim)
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)



def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k

class Rotary1DPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        assert dim % 2 == 0, "RoPE requires even dimension"
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        pos = torch.arange(0, max_seq_len).float()
        sinusoid = torch.einsum("i,j->ij", pos, inv_freq)

        sin, cos = sinusoid.sin(), sinusoid.cos()
        self.register_buffer("sin", sin[None, :, :], persistent=False)
        self.register_buffer("cos", cos[None, :, :], persistent=False)

    def forward(self, q, k):

        sin, cos = self.sin[:, :q.size(1)], self.cos[:, :q.size(1)]
        sin = sin.unsqueeze(2)
        cos = cos.unsqueeze(2)

        q1, k1 = q[..., :q.shape[-1]//2], k[..., :k.shape[-1]//2]
        q2, k2 = q[..., q.shape[-1]//2:], k[..., k.shape[-1]//2:]

        q_rot, k_rot = apply_rotary_pos_emb(q1, k1, cos, sin)
        q_rot2, k_rot2 = apply_rotary_pos_emb(q2, k2, cos, sin)

        q_out = torch.cat([q_rot, q_rot2], dim=-1)
        k_out = torch.cat([k_rot, k_rot2], dim=-1)
        return q_out, k_out


class Projector(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.layer_1 = nn.Linear(192, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_4 = nn.Linear(128, 128)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.gelu(self.layer_1(self.dropout(x)))
        x = self.gelu(self.layer_2(x))
        x = self.gelu(self.layer_3(x))
        out = self.layer_4(x)
        return out


class MAD_Speech(nn.Module):
    def __init__(self):
        super().__init__()
        self.speech_sim = SpeechSim()
        self.projs = nn.ModuleDict({"speakers": Projector(),
                                   "gender": Projector(),
                                   "emotion": Projector(),
                                   "accent": Projector(),
                                   "background_noise": Projector()})

    def forward(self, x):
        x = self.speech_sim(x)
        embs = {k: v(x) for k, v in self.projs.items()}
        diversity_scores = {k: vendi_score(v) if k != "gender" else avg_cosine(v) for k, v in embs.items()}
        return diversity_scores


