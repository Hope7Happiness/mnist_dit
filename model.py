import torch
import torch.nn as nn
import torch.nn.functional as F

class DiT(nn.Module):
    
    def __init__(
        self,
        num_layers,
        hidden_dim,
        num_head,
        patch_size = 2,
        image_size = 28,
        image_channel = 1,
        num_classes = 10
    ):
        super().__init__()

        self.image_size = image_size
        self.image_channel = image_channel
        self.patch_size = patch_size

        assert image_size % patch_size == 0, (image_size, patch_size)
        self.seq_len = (image_size // patch_size) ** 2
        self.seq_dim = image_channel * (patch_size ** 2)

        self.x_emb = nn.Linear(self.seq_dim, hidden_dim)
        self.t_emb = TimeEmbedding(hidden_dim)
        self.y_emb = LabelEmbedding(hidden_dim, num_classes)

        self.blocks = nn.ModuleList([Block(dim=hidden_dim, num_heads=num_head) for _ in range(num_layers)])

        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_mod = nn.Linear(hidden_dim, hidden_dim * 2)
        self.final_proj = nn.Linear(hidden_dim, self.seq_dim)

    def patchify(self, x):
        B, C, H, W = x.shape
        assert C == self.image_channel, (C, self.image_channel)
        assert H == W == self.image_size, (H, W, self.image_size)

        x = x.reshape(B, C, self.patch_size, H // self.patch_size, self.patch_size, W // self.patch_size).permute(0, 3, 5, 1, 2, 4)
        x = x.reshape(B, self.seq_len, self.seq_dim)
        return x

    def unpatchify(self, x):
        B, L, D = x.shape
        assert L == self.seq_len, (L, self.seq_len)
        assert D == self.seq_dim, (D, self.seq_dim)

        x = x.reshape(B, self.image_size // self.patch_size, self.image_size // self.patch_size, self.image_channel, self.patch_size, self.patch_size).permute(0, 3, 4, 1, 5, 2)
        x = x.reshape(B, self.image_channel, self.image_size, self.image_size)
        return x

    def forward(self, x, t, y):
        x = self.patchify(x)
        x = self.x_emb(x)

        B, L, D = x.shape
        temb = self.t_emb(t)
        yemb = self.y_emb(y)
        assert temb.shape == yemb.shape == (B, D)
        cond = temb + yemb

        for block in self.blocks:
            x = block(x, cond=cond)

        # final norm
        x = self.final_norm(x)
        s, b = self.final_mod(cond).unsqueeze(1).chunk(2, dim=-1)
        return self.unpatchify(self.final_proj(modulate(x, s, b)))


class Block(nn.Module):
    
    def __init__(self, dim, num_heads):
        super().__init__()

        self.attn = Attention(dim=dim, num_heads=num_heads)
        self.mlp = MLP(dim=dim)

        self.ln1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(dim, elementwise_affine=False)

        self.mod = nn.Linear(dim, dim * 6)

    def forward(self, x, cond):
        B, L, D = x.shape
        assert cond.shape == (B, D)

        s1, b1, g1, s2, b2, g2 = self.mod(cond).unsqueeze(1).chunk(6, dim=-1)
        x = x + g1 * self.attn(modulate(self.ln1(x), s1, b1))
        x = x + g2 * self.mlp(modulate(self.ln2(x), s2, b2))
        return x

class Attention(nn.Module):
    
    def __init__(self, dim, num_heads):
        super().__init__()

        assert dim % num_heads == 0, (dim, num_heads)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3*dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.reshape(B, L, self.num_heads, self.head_dim),
            (q, k, v)
        )

        score = torch.einsum('bihd, bjhd -> bijh', q, k)
        score = torch.softmax(score * self.scale, dim=2)
        out = torch.einsum('bijh, bjhd -> bihd', score, v)
        out = out.reshape(B, L, D)
        return self.out_proj(out)

class MLP(nn.Module):
    
    def __init__(self, dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        return self.net(x)

def modulate(x, scale, bias):
    return x * (1. + scale) + bias

class TimeEmbedding(nn.Module):

    def __init__(self, dim, max_T=10000.):
        super().__init__()

        assert dim % 2 == 0, dim
        self.register_buffer(
            'angles',
            max_T ** -torch.arange(0, dim, 2, dtype=torch.float32) / dim
        )
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        B, = t.shape
        emb = torch.einsum('i, b -> bi', self.angles, t.float())
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return self.net(emb)


class LabelEmbedding(nn.Module):

    def __init__(self, dim, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.emb = nn.Embedding(num_embeddings=num_classes, embedding_dim=dim)

    def forward(self, y):
        assert y.dtype != torch.float32
        return self.emb(y)

# build networks
from functools import partial
DiT_debug = partial(
    DiT,
    num_layers = 1,
    hidden_dim = 4,
    num_head = 2,
    patch_size = 2,
    image_size = 6,
    image_channel = 1,
    num_classes = 10
)

DiT_base = partial(
    DiT,
    num_layers = 12,
    hidden_dim = 192,
    num_head = 6,
)

if __name__ == '__main__':
    # print(sum(p.numel() for p in DiT_base().parameters()))

    model = DiT_debug()
    x = torch.randn((7, 1, 6, 6))
    y = torch.randint(0, 10, (7,))
    t = torch.randint(0, 1000, (7,))
    pred = model(x, t, y=y)
    assert pred.shape == x.shape
    print('Test passed!')