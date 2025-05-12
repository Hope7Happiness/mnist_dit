import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

class Attention(nn.Module):
    
    def __init__(self, dim, num_heads):
        super().__init__()
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)
        assert dim % num_heads == 0, (dim, num_heads)
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = self.head_dim ** -0.5
        
        # init
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()
        
    def forward(self, x):
        B, L, D = x.shape
        assert D == self.head_dim * self.num_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_heads, self.head_dim)
        v = v.reshape(B, L, self.num_heads, self.head_dim)
        attn = torch.einsum('bihd, bjhd -> bijh', q, k)
        attn = F.softmax(attn * self.scale, dim=2)
        out = torch.einsum('bijh, bjhd -> bihd', attn, v)
        out = out.reshape(B, L, D)
        return self.proj(out)

class MLP(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.SiLU(),
            nn.Linear(dim*4, dim)
        )
        
        # init
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()
    
    def forward(self, x):
        return self.net(x)

def modulate(x, scale, bias):
    return x * (1 + scale.unsqueeze(1)) + bias.unsqueeze(1)

class Block(nn.Module):
    
    def __init__(self, seq_len, channels, num_heads):
        super().__init__()
        self.channels = channels
        self.seq_len = seq_len
        
        self.attn = Attention(channels, num_heads)
        self.mlp = MLP(channels)
        self.cond_mlp = nn.Linear(channels, channels * 6)
        self.ln1 = nn.LayerNorm(channels, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(channels, elementwise_affine=False)
        
        # init
        self.cond_mlp.weight.data.zero_()
        self.cond_mlp.bias.data.zero_()

    def forward(self, x, cond):
        B, L, D = x.shape
        assert L == self.seq_len, x.shape
        assert D == self.channels, x.shape
        assert cond.shape == (B, D), cond.shape
        s1, b1, g1, s2, b2, g2 = self.cond_mlp(F.silu(cond)).chunk(6, dim=-1)
        
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.ln1(x), scale=s1, bias=b1))
        x = x + g2.unsqueeze(1) * self.mlp(modulate(self.ln2(x), scale=s2, bias=b2))
        return x

class TimeEmbed(nn.Module):
    
    def __init__(self, dim, max_l=10000.):
        # sinous pos embed
        super().__init__()
        assert dim % 2 == 0, dim
        self.register_buffer('angles', max_l ** (- torch.arange(0, dim, 2) / dim))
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, t):
        assert t.dtype != torch.float32, 't should be int'
        assert t.ndim == 1
        B = t.shape[0]
        emb = torch.einsum('i, b -> bi', self.angles, t.float())
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], axis=-1)
        return self.mlp(emb)

class LabelEmbed(nn.Module):
    
    def __init__(self, num_classes, dim):
        super().__init__()
        self.emb = nn.Embedding(num_classes, dim)
        
    def forward(self, y):
        assert y.ndim == 1, y.shape
        B = y.shape[0]
        return self.emb(y)

class DiT(nn.Module):
    
    def __init__(
        self,
        image_size=28,
        in_channel=1,
        patch_size=2,
        num_classes=10,
        num_layers=6,
        channels=256,
        num_heads=4,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channel = in_channel
        self.patch_size = patch_size
        assert image_size % patch_size == 0, (image_size, patch_size)
        self.seq_len = (image_size // patch_size) ** 2
        self.seq_dim = self.in_channel * self.patch_size * self.patch_size
        
        # embedddings
        self.x_embed = nn.Linear(self.seq_dim, channels)
        self.pos_embed = nn.Parameter(torch.randn(self.seq_len, channels) * .02)
        self.y_embed = LabelEmbed(num_classes, channels)
        self.t_embed = TimeEmbed(channels)
        
        # layers
        self.layers = nn.ModuleList([
            Block(self.seq_len, channels, num_heads)
            for _ in range(num_layers)
        ])
        
        # final layernorm
        self.final_ln = nn.LayerNorm(channels, elementwise_affine=False)
        self.final_mod = nn.Linear(channels, channels * 2)
        
        # init params
        nn.init.xavier_normal_(self.x_embed.weight)
    
    def patchify(self, x):
        B, C, H, W = x.shape
        assert C == self.in_channel, x.shape
        assert H == W == self.image_size, x.shape
        x = x.reshape(B, C, H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(B, self.seq_len, self.seq_dim)
    
    def unpatchify(self, x):
        B, L, D = x.shape
        assert L == self.seq_len, x.shape
        assert D == self.seq_dim, x.shape
        x = x.reshape(B, self.image_size//self.patch_size, self.image_size//self.patch_size, self.in_channel, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5)
        return x.reshape(B, self.in_channel, self.image_size, self.image_size)
    
    def forward(self, x, y=None, t=None):
        x = self.patchify(x)
        yemb = self.y_embed(y)
        temb = self.t_embed(t)
        cond = yemb + temb
        for ly in self.layers:
            x = ly(x, cond=cond)
        # final layernorm
        s, b = self.final_mod(cond).chunk(2, dim=-1)
        x = modulate(self.final_ln(x), scale=s, bias=b)
        return self.unpatchify(x)
    
DiT_base = DiT # 7M params

DiT_debug = partial(
    DiT,
    image_size=6,
    in_channel=1,
    patch_size=2,
    num_classes=10,
    num_layers=1,
    channels=4,
    num_heads=2
)
    
if __name__ == '__main__':
    model = DiT()
    print('num of model parameters:', sum(p.numel() for p in model.parameters()))

    # model = DiT_debug()
    # fake_x = torch.randn(7,1,6,6)
    # fake_y = torch.randint(0,10,(7,))
    # fake_t = torch.randint(0,1000,(7,))
    # out = model(fake_x, y=fake_y, t=fake_t)
    # assert out.shape == fake_x.shape
    # print('success!')
    