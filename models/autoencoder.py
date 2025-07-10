from typing import Optional

import numpy as np

import torch

from torch import Tensor
from torch import nn
from torch import functional as F
from torch.autograd import grad

from einops import rearrange
from timm.layers import DropPath
from torch_cluster import fps


class PolynomialFeatures(torch.nn.Module):
    def __init__(self, degree):
        super(PolynomialFeatures, self).__init__()

        self.degree = degree

    def forward(self, x):

        polynomial_list = [x]

        for it in range(1, self.degree + 1):
            polynomial_list.append(
                torch.einsum("...i,...j->...ij", polynomial_list[-1], x).flatten(-2, -1)
            )

        return torch.cat(polynomial_list, -1)


class RandomGaussianFourierFeatures(nn.Module):
    def __init__(
        self,
        dim: int,
        embedding_dim: int,
        sigma: float = 1.0,
        basis: Optional[Tensor] = None,
        optimize_basis: bool = False,
    ):
        super().__init__()

        assert embedding_dim % 2 == 0
        assert basis is None or basis.shape == (dim, embedding_dim // 2)

        self.optimize_basis = optimize_basis

        basis = (
            basis if basis is not None else sigma * torch.randn(dim, embedding_dim // 2)
        )

        if optimize_basis:
            self.basis = nn.Parameter(basis)
        else:
            self.register_buffer("basis", basis)

    def forward(self, x: Tensor):
        x = 2 * np.pi * x @ self.basis

        return torch.cat((x.sin(), x.cos()), dim=-1)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads

        if context_dim is None:
            context_dim = query_dim

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim))

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)

        if context is None:
            context = x

        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        return self.to_out(out)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim

        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class DownsampleBlock(nn.Module):
    __constant__ = ["learnable_queries"]

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 1,
        dim_heads: int = 64,
        num_latents: float = 128,
        learnable_queries: bool = False,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_latents = num_latents

        self.cattn = CrossAttention(feature_dim, heads=num_heads, dim_head=dim_heads)
        self.attn = CrossAttention(feature_dim, heads=num_heads, dim_head=dim_heads)
        self.ff = FeedForward(feature_dim, glu=True)

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)

        self.learnable_queries = learnable_queries

        if learnable_queries:
            self.queries = nn.Parameter(0.01 * torch.randn(1, num_latents, feature_dim))

    def forward(self, pos: Tensor, x: Tensor, num_latents: Optional[float] = None):
        B, N, D = pos.shape

        num_latents = self.num_latents if num_latents is None else num_latents

        if not self.learnable_queries:
            ratio = num_latents / N

            x_down = x.view(-1, self.feature_dim)
            pos = pos.view(-1, D)

            batch = torch.arange(B, device=pos.device).repeat_interleave(N)
            downsample_idx = fps(pos, batch=batch, ratio=ratio)

            x_down = x_down[downsample_idx].view(B, -1, self.feature_dim)
            pos = pos[downsample_idx].view(B, -1, D)

            x_down = self.norm1(x_down)
        else:
            x_down = self.queries.expand(B, -1, -1)

        x = self.cattn(x_down, self.norm2(x)) + x_down
        x = self.ff(self.norm3(x)) + x

        return pos, x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 1,
        dim_heads: int = 64,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads

        self.attn1 = CrossAttention(feature_dim, heads=num_heads, dim_head=dim_heads)
        self.ff = FeedForward(feature_dim, glu=False)

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        self.drop1 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.drop2 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: Tensor):
        x = self.drop1(self.attn1(self.norm1(x))) + x
        x = self.drop2(self.ff(self.norm2(x))) + x

        return x


class SDFTransformer(nn.Module):
    __constants__ = [
        "feature_dim",
        "dim_heads",
        "num_heads",
        "latent_dim",
        "decoder_depth",
        "mlp_depth",
        "mlp_residual",
        "variational",
        "global_embedding",
    ]

    def __init__(
        self,
        feature_dim: int,
        dim_heads: int,
        num_heads: int,
        latent_dim: int = 64,
        num_latents: int = 256,
        encoder_depth: int = 0,
        decoder_depth: int = 4,
        mlp_depth: int = 1,
        drop_path_rate: float = 0.1,
        num_labels: int = 13,
        mlp_residual: bool = False,
        variational: bool = False,
        global_embedding: bool = True,
        learnable_queries: bool = False,
        embedding_type: str = "polynomical",
        probabilistic: bool = False,
        encode_labels: bool = False,
        semantic: bool = True,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.dim_heads = dim_heads
        self.num_heads = num_heads
        self.latent_dim = latent_dim

        self.num_latents = num_latents
        self.decoder_depth = decoder_depth
        self.encoder_depth = encoder_depth

        self.mlp_depth = mlp_depth
        self.mlp_residual = mlp_residual

        self.drop_path_rate = drop_path_rate

        self.num_labels = num_labels

        self.variational = variational
        self.probabilistic = probabilistic

        self.encode_labels = encode_labels

        self.global_embedding = global_embedding
        self.embedding_type = embedding_type

        self.semantic = semantic

        self.act_fn = nn.GELU()

        if embedding_type == "linear":
            self.embed = nn.Sequential(
                nn.LazyLinear(feature_dim),
            )
        elif embedding_type == "polynomial":
            self.embed = nn.Sequential(
                PolynomialFeatures(3),
                nn.LazyLinear(feature_dim),
            )
        elif embedding_type == "rff":
            self.embed = nn.Sequential(
                RandomGaussianFourierFeatures(3, feature_dim // 2, sigma=5),
                nn.LazyLinear(feature_dim),
            )
        else:
            raise NotImplementedError

        self.embed_with_labels = nn.LazyLinear(feature_dim)

        self.downsample_blocks = nn.ModuleList()

        num_latents = [num_latents] if type(num_latents) is int else num_latents

        for num_latents in num_latents:
            self.downsample_blocks.append(
                DownsampleBlock(
                    feature_dim,
                    num_heads=1,
                    dim_heads=feature_dim,
                    num_latents=num_latents,
                    learnable_queries=learnable_queries,
                )
            )

        self.encoder_blocks = nn.ModuleList()

        for _ in range(encoder_depth):
            self.encoder_blocks.append(
                AttentionBlock(
                    feature_dim,
                    num_heads=num_heads,
                    dim_heads=dim_heads,
                    drop_path_rate=drop_path_rate,
                )
            )

        # predict mu logvar if variational
        self.to_latent = nn.Linear(
            feature_dim, 2 * latent_dim if self.variational else latent_dim
        )
        self.from_latent = nn.Linear(latent_dim, feature_dim)

        self.decoder_blocks = nn.ModuleList()

        for _ in range(decoder_depth):
            self.decoder_blocks.append(
                AttentionBlock(
                    feature_dim,
                    num_heads=num_heads,
                    dim_heads=dim_heads,
                    drop_path_rate=drop_path_rate,
                )
            )

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)
        self.attn1 = CrossAttention(feature_dim, feature_dim, 1, feature_dim)

        self.mlp = nn.ModuleList()

        for _ in range(mlp_depth):
            self.mlp.append(nn.Linear(feature_dim, feature_dim))

        self.to_sdf = nn.Linear(feature_dim, 1 + self.probabilistic + self.num_labels)

    def sample_latents(self, mean: Tensor, logvar: Tensor):
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

    def kl(self, mean: Tensor, logvar: Tensor):
        return 0.5 * (mean**2 + logvar.exp() - 1.0 - logvar)

    def encode(
        self,
        pos: Tensor,
        return_latent_pos: bool = False,
        num_latents: Optional[float] = None,
    ):
        x = self.embed_with_labels(pos)
        pos = pos[..., :3]

        for downsample_block in self.downsample_blocks:
            pos, x = downsample_block(pos, x, num_latents=num_latents)

        for block in self.encoder_blocks:
            x = block(x)

        x = self.to_latent(x)

        if self.variational:
            mean, logvar = x.chunk(2, dim=-1)

            x = self.sample_latents(mean, logvar)
            kl = self.kl(mean, logvar)
        else:
            kl = None

        if return_latent_pos:
            return x, pos, kl

        return x, kl

    def encode_to_mu_logvar(
        self,
        pos: Tensor,
        return_latent_pos: bool = False,
        num_latents: Optional[float] = None,
    ):
        if not self.variational:
            return self.encode(pos, return_latent_pos, num_latents)

        x = self.embed_with_labels(pos)
        pos = pos[..., :3]

        for downsample_block in self.downsample_blocks:
            pos, x = downsample_block(pos, x, num_latents=num_latents)

        for block in self.encoder_blocks:
            x = block(x)

        x = self.to_latent(x)

        return x, None

    def decode_from_mu_logvar(self, x: Tensor, queries: Tensor):
        if self.variational:
            mean, logvar = x.chunk(2, dim=-1)
            x = self.sample_latents(mean, logvar)

        return self.decode(x, queries)

    def decode(self, x: Tensor, queries: Tensor):
        queries = self.embed(queries)

        x = self.from_latent(x)

        for block in self.decoder_blocks:
            x = block(x)

        queries = self.attn1(self.norm1(queries), self.norm2(x)) + queries
        queries = self.norm3(queries)

        # residual connection
        if self.mlp_residual:
            residual = queries

        for i, fc in enumerate(self.mlp):
            if self.mlp_residual and i != 0 and i == self.mlp_depth // 2:
                queries = queries + residual

            queries = self.act_fn(fc(queries))

        sdf = self.to_sdf(queries)

        return sdf

    def forward(self, pos, queries):
        latents, kl = self.encode(pos)
        sdf = self.decode(latents, queries)

        return sdf, latents, kl


class GaussianNLLLossBounded:
    def __init__(self, sigma_min: float, sigma_max: float):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.bound_min = np.log(sigma_min**2)
        self.bound_max = np.log(sigma_max**2)

        self.C = -self.nll(0, torch.zeros(1), torch.as_tensor(self.bound_min)).item()
        self.Z = (
            1
            / (
                self.nll(0, torch.zeros(1), torch.as_tensor(self.bound_max)) + self.C
            ).item()
        )

    @staticmethod
    def nll(target: Tensor, mean_pred: Tensor, logvar_pred: Tensor):
        return 0.5 * (((target - mean_pred).abs() / logvar_pred.exp()) + logvar_pred)

    def __call__(self, target: Tensor, mean_pred: Tensor, logvar_pred: Tensor):
        logvar_pred = logvar_pred.clip(self.bound_min, self.bound_max)
        print(logvar_pred)

        return self.Z * (self.nll(target, mean_pred, logvar_pred) + self.C)


class SDFLoss:
    def __init__(
        self,
        num_surface: int,
        regularize_latents: bool = True,
        w_surface: float = 1.0,
        w_eikonal: float = 0.1,
        w_reg: float = 1e-4,
        normal_loss_fn: str = "l1",
        weight_labels: bool = False,
    ):
        assert normal_loss_fn in ["l1", "l2", "cosine"]

        self.regularize_latents = regularize_latents
        self.num_surface = num_surface

        self.w_surface = w_surface
        self.w_eikonal = w_eikonal
        self.w_reg = w_reg

        self.weight_labels = weight_labels

        self.normal_loss_fn = getattr(self, f"loss_{normal_loss_fn}")

        self.nll_bounded = GaussianNLLLossBounded(0.001, 0.1)

    @staticmethod
    def loss_l1(x: Tensor, y: Tensor):
        return (x - y).abs().mean()

    @staticmethod
    def loss_l2(x: Tensor, y: Tensor):
        return (x - y).norm(dim=-1).mean()

    @staticmethod
    def loss_cosine(x: Tensor, y: Tensor, eps: float = 1e-7):
        y = y / (y.norm(dim=-1, keepdim=True) + eps)
        return (1 - (x * y).sum(-1)).mean()

    @staticmethod
    def mse_loss(target: Tensor, pred: Tensor, weight: float = 1.0):
        return (weight * (pred - target) ** 2).mean()

    @staticmethod
    def ce_loss(target: Tensor, pred: Tensor, weight: Optional[Tensor] = None):
        return torch.nn.functional.cross_entropy(pred, target, weight).mean()

    @staticmethod
    def nll(target: Tensor, mean_pred: Tensor, logvar_pred: Tensor):
        return 0.5 * (((target - mean_pred).abs() / logvar_pred.exp()) + logvar_pred)

    @staticmethod
    def kl(target: Tensor, mean: Tensor, logvar: Tensor):
        return 0.5 * ((mean - target).abs() + logvar.exp() - logvar)

    @staticmethod
    def _grad(inputs: Tensor, outputs: Tensor) -> Tensor:
        d = torch.ones_like(outputs, requires_grad=False)
        return grad(outputs, inputs, d, True, True, True)[0]

    def __call__(
        self,
        model: nn.Module,
        pointcloud: Tensor,
        surface: Tensor,
        volume: Tensor,
        normals: Tensor,
        labels_surface: Tensor,
    ):
        latents, kl = model.encode(pointcloud)

        volume = torch.cat((surface, volume), dim=1)
        volume.requires_grad_()

        pred_sdf = model.decode(latents, volume)

        if model.probabilistic:
            pred_sdf, pred_logvar, pred_label = pred_sdf.split(
                (1, 1, model.num_labels), dim=-1
            )

            loss_nll = self.nll(
                0,
                pred_sdf[:, : surface.shape[1]],  # uncertainty in true scale
                pred_logvar[:, : surface.shape[1]],
            ).mean()
        else:
            pred_sdf, pred_label = pred_sdf[..., :1], pred_sdf[..., 1:]
            loss_nll = None

        loss_surface = (pred_sdf[:, : surface.shape[1]]).abs().mean()

        pred_label_surface = pred_label[:, : surface.shape[1]]

        if model.semantic:
            if self.weight_labels and pred_label_surface.shape[-1] == 13:
                weight = torch.tensor(
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 20.0, 5.0, 5.0, 1.0],
                    device=pred_label.device,
                )
            else:
                weight = 1

            loss_label_surface = self.mse_loss(
                labels_surface, pred_label_surface, weight=weight
            )
        else:
            loss_label_surface = None

        gradients = self._grad(volume, pred_sdf)

        # Eikonal loss
        loss_eikonal = ((gradients.norm(dim=-1) - 1) ** 2).mean()

        # normal loss

        pred_normals = gradients[:, : surface.shape[1]]
        loss_normals = self.normal_loss_fn(normals, pred_normals)

        # latent regularization loss
        if model.variational:
            loss_reg = kl.mean()
        else:
            loss_reg = (latents.norm(dim=-1) ** 2).mean()

        loss = (
            self.w_surface * loss_surface
            + (loss_label_surface if loss_label_surface is not None else 0)
            + self.w_eikonal * loss_normals
            + (self.w_reg * loss_nll if loss_nll is not None else 0)
            + self.w_eikonal * loss_eikonal
            + self.w_reg * loss_reg
        )

        loss_dict = {
            "loss": loss.item(),
            "loss_surface": loss_surface.item(),
            "loss_eikonal": loss_eikonal.item(),
            "loss_normals": loss_normals.item(),
            "loss_reg": loss_reg.item(),
        }

        if loss_label_surface is not None:
            loss_dict["loss_label_surface"] = loss_label_surface.item()

        if loss_nll is not None:
            loss_dict["loss_nll"] = loss_nll.item()

        return loss, loss_dict
