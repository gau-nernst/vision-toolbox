from torch import nn, Tensor


class MLP(nn.Sequential):
    def __init__(self, d_model: int, mlp_ratio: float) -> None:
        super().__init__()
        mlp_dim = int(d_model * mlp_ratio)
        self.linear1 = nn.Linear(d_model, mlp_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(mlp_dim, d_model)


class MixerBlock(nn.Module):
    def __init__(self, n_tokens: int, d_model: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.token_mixing = MLP(n_tokens, mlp_ratio)
        self.norm2 = nn.LayerNorm(d_model)
        self.channel_mixing = MLP(d_model, mlp_ratio)

    def forward(self, x: Tensor) -> Tensor:
        # x -> (B, n_tokens, d_model)
        x = x + self.token_mixing(self.norm1(x).transpose(-1, -2)).transpose(-1, -2)
        x = x + self.channel_mixing(self.norm2(x))
        return x


class MLPMixer(nn.Module):
    def __init__(self, img_size: int, patch_size: int, d_model: int, n_layers: int, mlp_ratio: float) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, patch_size, patch_size)
        n_tokens = (img_size // patch_size) ** 2
        self.layers = nn.Sequential(*[MixerBlock(n_tokens, d_model, mlp_ratio) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x).flatten(-2).transpose(-1, -2)
        x = self.layers(x)
        x = self.norm(x)
        return x
