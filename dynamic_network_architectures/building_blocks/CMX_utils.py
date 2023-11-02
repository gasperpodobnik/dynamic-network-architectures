import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
import math


# --------------------------------------------------------------
### Cross-Modal Feature Rectify Module
# Channel-wise rectification
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        """MLP after max and avg pooling operations on FMs
        We get four vectors after the pooling operation (two modalities, each FM goes through max and avg pooling)

        Args:
            dim (_type_): _description_
            reduction (int, optional): _description_. Defaults to 1.
        """
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # output size is 1, 1, 1
        self.max_pool = nn.AdaptiveMaxPool3d(1)  # output size is 1, 1, 1
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid(),
        )

        # TODO maybe add some normalization to the FM of each modality to make it have similar values

    def forward(self, x0, x1):
        """

        Args:
            x1 (_type_): the reference modality FM
            x2 (_type_): the auxiliar modality FM

        Returns:
            _type_: _description_
        """
        B, _, D, H, W = x0.shape
        x = torch.cat((x0, x1), dim=1)  # concat in channel dim
        avg = self.avg_pool(x).view(B, self.dim * 2)  # B 2C
        max = self.max_pool(x).view(B, self.dim * 2)  # B 2C
        y = torch.cat((avg, max), dim=1)  # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)

        # make channel first dim so that we can later use image0 * channel_weights[1] and image1 * channel_weights[0]
        channel_weights = y.reshape(B, 2, self.dim, 1, 1, 1).permute(
            1, 0, 2, 3, 4, 5
        )  # 2 B C 1 1 1
        return channel_weights


# Spatial-wise rectification
class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv3d(self.dim * 2, self.dim // reduction, kernel_size=1),  # 1x1x1 conv
            nn.ReLU(inplace=True),
            nn.Conv3d(self.dim // reduction, 2, kernel_size=1),  # 1x1x1 conv
            nn.Sigmoid(),
        )

    def forward(self, x0, x1):
        B, _, D, H, W = x0.shape
        x = torch.cat((x0, x1), dim=1)  # B 2C D H W

        # make channel first dim so that we can later use image0 * spatial_weights[1] and image1 * spatial_weights[0]
        spatial_weights = (
            self.mlp(x).reshape(B, 2, 1, D, H, W).permute(1, 0, 2, 3, 4, 5)
        )  # 2 B 1 D H W
        return spatial_weights


class FeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=0.5, lambda_s=0.5):
        super(FeatureRectifyModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x0, x1):
        channel_weights = self.channel_weights(x0, x1)
        spatial_weights = self.spatial_weights(x0, x1)
        out_x0 = (
            x0
            + self.lambda_c * channel_weights[1] * x1
            + self.lambda_s * spatial_weights[1] * x1
        )
        out_x1 = (
            x1
            + self.lambda_c * channel_weights[0] * x0
            + self.lambda_s * spatial_weights[0] * x0
        )
        return out_x0, out_x1


# --------------------------------------------------------------
### Feature Fusion Module
## Stage 1
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        if num_heads is None:
            num_heads = 8
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.kv0 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x0, x1):
        B, N, C = x0.shape
        q0 = (
            x0.reshape(B, -1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        q1 = (
            x1.reshape(B, -1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        # queries have shape B, num_heads, N, C//num_heads
        k0, v0 = (
            self.kv0(x0)
            .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        k1, v1 = (
            self.kv1(x1)
            .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )

        # this is G_RGB in paper
        # this is (hopefully was) the source of the nan values
        # (k0.transpose(-2, -1) @ v0) * self.scale) produces very high values, which in float16 are inf
        # after softmax, the values are nan, which propagate to the rest of the network and essentially break it
        # current solution first multiplies by scale and then does matrix multiplication
        ctx0 = k0.transpose(-2, -1) @ (v0 * self.scale)
        ctx0 = ctx0.softmax(dim=-3)  # NOTE: original code has dim=-2, but this is wrong

        # this is G_X in paper
        ctx1 = k1.transpose(-2, -1) @ (v1 * self.scale)
        ctx1 = ctx1.softmax(dim=-3)  # NOTE: original code has dim=-2, but this is wrong

        x0 = (q0 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x1 = (q1 @ ctx0).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()

        return x0, x1


class EfficientCrossAttentionHead(nn.Module):
    def __init__(self, n_embed, head_size, qkv_bias=False):
        super().__init__()

        self.head_size = head_size

        self.key0 = nn.Linear(n_embed, head_size, bias=qkv_bias)
        self.key1 = nn.Linear(n_embed, head_size, bias=qkv_bias)

        self.query0 = nn.Linear(n_embed, head_size, bias=qkv_bias)
        self.query1 = nn.Linear(n_embed, head_size, bias=qkv_bias)

        self.value0 = nn.Linear(n_embed, head_size, bias=qkv_bias)
        self.value1 = nn.Linear(n_embed, head_size, bias=qkv_bias)

    def forward(self, x0, x1):
        """
        inputs should be B, C, N, where N = DxHxW for 3D or N = HxW for 2D
        """

        k0 = self.key0(x0)
        k1 = self.key1(x1)

        q0 = self.query0(x0)
        q1 = self.query1(x1)

        v0 = self.value0(x0)
        v1 = self.value1(x1)

        ## test varaiances
        # B, N, C = x0.shape
        # N = 10
        # k0, k1 = torch.randn(B, N, self.head_size), torch.randn(B, N, self.head_size)
        # q0, q1 = torch.randn(B, N, self.head_size), torch.randn(B, N, self.head_size)
        # v0, v1 = torch.randn(B, N, self.head_size), torch.randn(B, N, self.head_size)

        # q0, k0, v0 = q0[:,:100], k0[:,:100], v0[:,:100]
        # F.scaled_dot_product_attention(q0[:,:100], k0[:,:100], v0[:,:100]).var()
        # equvalent ((q0 @ k0.transpose(-2, -1) * self.head_size **-0.5).softmax(dim=-2) @ v0).var()
        # v0[:,:100].var()

        k0 = k0.softmax(dim=-2)
        k1 = k1.softmax(dim=-2)

        q0 = q0.softmax(dim=-1)
        q1 = q1.softmax(dim=-1)

        w0 = k0.transpose(-2, -1) @ v0
        w1 = k1.transpose(-2, -1) @ v1

        # (q0 @ w0).var()

        attn_x0 = q0 @ w1
        attn_x1 = q1 @ w0

        ## test varaiances
        # print(k0.var(), k1.var(), q0.var(), q1.var(), v0.var(), v1.var(), w0.var(), w1.var())
        # print(x0.var(), x1.var())
        # print(attn_x0.var(), attn_x1.var())

        return attn_x0, attn_x1


class EfficientCrossAttentionHeadV2(nn.Module):
    def __init__(self, n_embed, head_size, qkv_bias=False):
        super().__init__()

        self.head_size = head_size

        self.key0 = nn.Linear(n_embed, head_size, bias=qkv_bias)
        self.key1 = nn.Linear(n_embed, head_size, bias=qkv_bias)

        self.query0 = nn.Linear(n_embed, head_size, bias=qkv_bias)
        self.query1 = nn.Linear(n_embed, head_size, bias=qkv_bias)

        self.value0 = nn.Linear(n_embed, head_size, bias=qkv_bias)
        self.value1 = nn.Linear(n_embed, head_size, bias=qkv_bias)

    def forward(self, x0, x1):
        """
        inputs should be B, C, N, where N = DxHxW for 3D or N = HxW for 2D
        """

        k0 = self.key0(x0).softmax(dim=-1).transpose(-2, -1)
        k1 = self.key1(x1).softmax(dim=-1).transpose(-2, -1)

        q0 = self.query0(x0).softmax(dim=-2)
        q1 = self.query1(x1).softmax(dim=-2)

        v0 = self.value0(x0)
        v1 = self.value1(x1)

        w0 = k0 @ v0
        w1 = k1 @ v1

        attn_x0 = q0 @ w1
        attn_x1 = q1 @ w0

        return attn_x0, attn_x1


class EfficientCrossAttentionHead3modals(nn.Module):
    def __init__(self, n_embed, head_size, qkv_bias=False):
        super().__init__()

        self.head_size = head_size

        self.key0 = nn.Linear(n_embed, head_size, bias=qkv_bias)
        self.key1 = nn.Linear(n_embed, head_size, bias=qkv_bias)
        self.key2 = nn.Linear(n_embed, head_size, bias=qkv_bias)

        self.query0 = nn.Linear(n_embed, head_size, bias=qkv_bias)
        self.query1 = nn.Linear(n_embed, head_size, bias=qkv_bias)
        self.query2 = nn.Linear(n_embed, head_size, bias=qkv_bias)

        self.value0 = nn.Linear(n_embed, head_size, bias=qkv_bias)
        self.value1 = nn.Linear(n_embed, head_size, bias=qkv_bias)
        self.value2 = nn.Linear(n_embed, head_size, bias=qkv_bias)

    def forward(self, x0, x1, x2):
        k0 = self.key0(x0)
        k1 = self.key1(x1)
        k2 = self.key1(x2)

        q0 = self.query0(x0)
        q1 = self.query1(x1)
        q2 = self.query1(x2)

        v0 = self.value0(x0)
        v1 = self.value1(x1)
        v2 = self.value1(x2)

        k0 = k0.softmax(dim=-2)
        k1 = k1.softmax(dim=-2)
        k2 = k2.softmax(dim=-2)

        q0 = q0.softmax(dim=-1)
        q1 = q1.softmax(dim=-1)
        q2 = q2.softmax(dim=-1)

        w0 = k0.transpose(-2, -1) @ v0
        w1 = k1.transpose(-2, -1) @ v1
        w2 = k2.transpose(-2, -1) @ v2

        attn_x01 = q0 @ w1
        attn_x02 = q0 @ w2
        attn_x10 = q1 @ w0
        attn_x12 = q1 @ w2
        attn_x20 = q2 @ w0
        attn_x21 = q2 @ w1

        attn_x0 = attn_x01 + attn_x02
        attn_x1 = attn_x10 + attn_x12
        attn_x2 = attn_x20 + attn_x21

        return attn_x0, attn_x1, attn_x2


class MultiHeadAttention(nn.Module):
    def __init__(
        self, num_heads, head_size, n_embed, attention_block=EfficientCrossAttentionHead
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                attention_block(n_embed=n_embed, head_size=head_size)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x0, x1):
        x0_list = []
        x1_list = []
        for head in self.heads:
            x0_out, x1_out = head(x0, x1)
            x0_list.append(x0_out)
            x1_list.append(x1_out)

        x0 = torch.cat(x0_list, dim=-1) + x0
        x1 = torch.cat(x1_list, dim=-1) + x1
        return x0, x1


class MultiHeadAttentionNoResidual(nn.Module):
    def __init__(
        self,
        num_heads,
        head_size,
        n_embed,
        attention_block=EfficientCrossAttentionHeadV2,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                attention_block(n_embed=n_embed, head_size=head_size)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x0, x1):
        x0_list = []
        x1_list = []
        for head in self.heads:
            x0_out, x1_out = head(x0, x1)
            x0_list.append(x0_out)
            x1_list.append(x1_out)

        x0 = torch.cat(x0_list, dim=-1)
        x1 = torch.cat(x1_list, dim=-1)
        return x0, x1


class MultiHeadAttention3modals(nn.Module):
    def __init__(
        self, num_heads, head_size, n_embed, attention_block=EfficientCrossAttentionHead
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                attention_block(n_embed=n_embed, head_size=head_size)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x0, x1, x2):
        x0_list = []
        x1_list = []
        x2_list = []
        for head in self.heads:
            x0_out, x1_out, x2_out = head(x0, x1, x2)
            x0_list.append(x0_out)
            x1_list.append(x1_out)
            x2_list.append(x2_out)

        x0 = torch.cat(x0_list, dim=-1) + x0
        x1 = torch.cat(x1_list, dim=-1) + x1
        x2 = torch.cat(x2_list, dim=-1) + x2
        return x0, x1, x2


import torch.nn.functional as F


class CrossAttentionHead(nn.Module):
    def __init__(self, n_embed, head_size, qkv_bias=False):
        super(CrossAttentionHead, self).__init__()

        self.head_size = head_size
        self.scale = head_size**-0.5

        self.key0 = nn.Linear(n_embed, head_size, bias=qkv_bias)
        self.key1 = nn.Linear(n_embed, head_size, bias=qkv_bias)

        self.query0 = nn.Linear(n_embed, head_size, bias=qkv_bias)
        self.query1 = nn.Linear(n_embed, head_size, bias=qkv_bias)

        self.value0 = nn.Linear(n_embed, head_size, bias=qkv_bias)
        self.value1 = nn.Linear(n_embed, head_size, bias=qkv_bias)

    def forward(self, x0, x1):
        B, N, C = x0.shape
        k0 = self.key0(x0)
        k1 = self.key1(x1)

        q0 = self.query0(x0)
        q1 = self.query1(x1)

        v0 = self.value0(x0)
        v1 = self.value1(x1)

        # test varaiances
        # N = 100
        # k0, k1 = torch.randn(B, N, self.head_size), torch.randn(B, N, self.head_size)
        # q0, q1 = torch.randn(B, N, self.head_size), torch.randn(B, N, self.head_size)
        # v0, v1 = torch.randn(B, N, self.head_size), torch.randn(B, N, self.head_size)

        wei0 = q0 @ (k1.transpose(-2, -1)) * self.head_size**-0.5
        wei0 = wei0.softmax(dim=-1)
        x0 = wei0 @ v1

        wei1 = q1 @ (k0.transpose(-2, -1)) * self.head_size**-0.5
        wei1 = wei1.softmax(dim=-1)
        x1 = wei1 @ v0

        # print(k0.var(), k1.var(), q0.var(), q1.var(), v0.var(), v1.var(), x0.var(), x1.var(), wei0.var(), wei1.var())

        return x0, x1


class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        # TODO: this is a temporary solution for very high values in x1 and x2 that create nans in outputs
        # self.act1 = nn.Sigmoid()
        # self.act2 = nn.Sigmoid()
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x0, x1):
        # x0 and x1 are featuremaps from two modalities that have shape B, N, C matrix, where N = DxHxW
        # applies linear to get B, N, Ci and chunks it into B, N, Ci/2
        residual_0, interact_0 = self.act1(self.channel_proj1(x0)).chunk(2, dim=-1)
        residual_1, interact_1 = self.act2(self.channel_proj2(x1)).chunk(2, dim=-1)
        v0, v1 = self.cross_attn(
            interact_0, interact_1
        )  # each output has shape B, N, Ci/2
        residual_0 = torch.cat((residual_0, v0), dim=-1)
        residual_1 = torch.cat((residual_1, v1), dim=-1)
        out_x1 = self.norm1(x0 + self.end_proj1(residual_0))
        out_x2 = self.norm2(x1 + self.end_proj2(residual_1))
        # both outputs have shape B, N, C
        return out_x1, out_x2


# Stage 2
class ChannelEmbed(nn.Module):
    def __init__(
        self, in_channels, out_channels, reduction=1, norm_layer=nn.InstanceNorm3d
    ):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            # depth wise convolution
            nn.Conv3d(
                out_channels // reduction,
                out_channels // reduction,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=out_channels // reduction,
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                out_channels // reduction, out_channels, kernel_size=1, bias=True
            ),
            norm_layer(out_channels),
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, D, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out


class FeatureFusionModuleStage1(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x0, x1):
        B, C, D, H, W = x0.shape
        # flatten(2) means that its start dim is 2, meaning that we preserve B, C and flatten the D, H, W dimensions
        # this creates a B, C, N matrix, where N = DxHxW
        # we then transpose it to B, N, C
        x0 = x0.flatten(2).transpose(1, 2)
        x1 = x1.flatten(2).transpose(1, 2)
        x0, x1 = self.cross(x0, x1)

        x0 = x0.transpose(1, 2).reshape(B, C, D, H, W).contiguous()
        x1 = x1.transpose(1, 2).reshape(B, C, D, H, W).contiguous()

        return x0, x1


class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.InstanceNorm3d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(
            in_channels=dim * 2,
            out_channels=dim,
            reduction=reduction,
            norm_layer=norm_layer,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x0, x1):
        B, C, D, H, W = x0.shape
        # flatten(2) means that its start dim is 2, meaning that we preserve B, C and flatten the D, H, W dimensions
        # this creates a B, C, N matrix, where N = DxHxW
        # we then transpose it to B, N, C
        x0 = x0.flatten(2).transpose(1, 2)
        x1 = x1.flatten(2).transpose(1, 2)
        x0, x1 = self.cross(x0, x1)

        # x0, x1 have shape B, N, C
        merge = torch.cat((x0, x1), dim=-1)
        merge = self.channel_emb(merge, D, H, W)

        return merge


class Tmp(nn.Module):
    def __init__(
        self, n_embed, reduction=1, num_heads=None, norm_layer=nn.InstanceNorm3d
    ):
        super().__init__()
        self.cross = EfficientCrossAttentionHead(n_embed=n_embed, head_size=n_embed)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x0, x1):
        x0: torch.Tensor
        x1: torch.Tensor
        B, C, D, H, W = x0.shape
        # flatten(2) means that its start dim is 2, meaning that we preserve B, C and flatten the D, H, W dimensions
        # this creates a B, C, N matrix, where N = DxHxW
        # we then transpose it to B, N, C
        x0 = x0.flatten(2).transpose(1, 2)
        x1 = x1.flatten(2).transpose(1, 2)
        x0, x1 = self.cross(x0, x1)

        x0 = x0.transpose(1, 2).reshape(B, C, D, H, W).contiguous()
        x1 = x1.transpose(1, 2).reshape(B, C, D, H, W).contiguous()

        return x0, x1


class Tmp2(nn.Module):
    def __init__(
        self,
        n_embed,
        num_heads=None,
        norm_layer=nn.InstanceNorm3d,
        attention_block=EfficientCrossAttentionHead,
    ):
        super().__init__()
        self.cross = MultiHeadAttention(
            n_embed=n_embed,
            head_size=n_embed // num_heads,
            num_heads=num_heads,
            attention_block=attention_block,
        )
        self.norm_layer = norm_layer(n_embed)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x0, x1):
        x0: torch.Tensor
        x1: torch.Tensor
        B, C, D, H, W = x0.shape
        # flatten(2) means that its start dim is 2, meaning that we preserve B, C and flatten the D, H, W dimensions
        # this creates a B, C, N matrix, where N = DxHxW
        # we then transpose it to B, N, C
        x0 = x0.flatten(2).transpose(1, 2)
        x1 = x1.flatten(2).transpose(1, 2)
        x0, x1 = self.cross(x0, x1)

        x0 = x0.transpose(1, 2).reshape(B, C, D, H, W).contiguous()
        x1 = x1.transpose(1, 2).reshape(B, C, D, H, W).contiguous()

        x0 = self.norm_layer(x0)
        x1 = self.norm_layer(x1)

        return x0, x1


from einops.layers.torch import Rearrange


class PyTorchMultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        num_heads,
        patch_depth,
        patch_height,
        patch_width,
        shp,
        kernel_size=None,
        stride=None,
    ) -> None:
        """This is a multihead cross attention module that takes two feature maps and applies cross attention to them.
        The standard pytorch multihead attention module is employed (not the efficient variant, but it is optimized for speed by pytorch engineers, so I trust that the internals work correctly).
        Around this module, I built a wrapper that takes two feature maps, compresses them to a lower dimensionality (spatial dims are compressed only in case the stride is used),
        applies multihead attention, and then decompresses them back to the original dimensionality.
        It reduces/increases the number of channels to 64 to limit the computational cost of the attention for large featuremaps.

        Args:
            in_channels (_type_): _description_
            num_heads (_type_): _description_
            patch_depth (_type_): _description_
            patch_height (_type_): _description_
            patch_width (_type_): _description_
            kernel_size (_type_, optional): _description_. Defaults to None.
            stride (_type_, optional): _description_. Defaults to None.
        """
        super().__init__()

        # for now this is hardcoded to 64, because it is a reasonable number (not to small to obstruct the attention, not too big to be too expensive)
        self.out_channels = 64

        self.num_heads = num_heads
        embed_dim = self.out_channels * patch_depth * patch_height * patch_width
        D, H, W = shp

        if kernel_size is None:
            kernel_size = [1, 1, 1]
        if stride is None:
            stride = [1, 1, 1]

        self.compress0 = nn.Conv3d(
            in_channels,
            self.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=[(i - 1) // 2 for i in kernel_size],
        )
        self.compress1 = nn.Conv3d(
            in_channels,
            self.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=[(i - 1) // 2 for i in kernel_size],
        )
        self.decompress0 = nn.ConvTranspose3d(
            self.out_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=[(i - 1) // 2 for i in kernel_size],
        )
        self.decompress1 = nn.ConvTranspose3d(
            self.out_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=[(i - 1) // 2 for i in kernel_size],
        )

        print(embed_dim)
        self.mha0 = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.mha1 = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

        self.FM_to_patch_embedding0 = nn.Sequential(
            Rearrange(
                "b c (d p0) (h p1) (w p2) -> b (d h w) (p0 p1 p2 c)",
                p0=patch_depth,
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(embed_dim),
        )
        self.FM_to_patch_embedding1 = nn.Sequential(
            Rearrange(
                "b c (d p0) (h p1) (w p2) -> b (d h w) (p0 p1 p2 c)",
                p0=patch_depth,
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(embed_dim),
        )

        self.patch_embedding_to_FM0 = nn.Sequential(
            Rearrange(
                "b (d h w) (p0 p1 p2 c) -> b c (d p0) (h p1) (w p2)",
                p0=patch_depth,
                p1=patch_height,
                p2=patch_width,
                d=D // patch_depth,
                h=H // patch_height,
                w=W // patch_width,
            ),
        )
        self.patch_embedding_to_FM1 = nn.Sequential(
            Rearrange(
                "b (d h w) (p0 p1 p2 c) -> b c (d p0) (h p1) (w p2)",
                p0=patch_depth,
                p1=patch_height,
                p2=patch_width,
                d=D // patch_depth,
                h=H // patch_height,
                w=W // patch_width,
            ),
        )

        self.to_qkv0 = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.to_qkv1 = nn.Linear(embed_dim, embed_dim * 3, bias=False)

    def forward(self, x0, x1):
        q0, k0, v0 = self.to_qkv0(
            self.FM_to_patch_embedding0(self.compress0(x0))
        ).chunk(3, dim=-1)
        q1, k1, v1 = self.to_qkv1(
            self.FM_to_patch_embedding1(self.compress1(x1))
        ).chunk(3, dim=-1)

        out0, _ = self.mha0(q0, k1, v1)
        out1, _ = self.mha1(q1, k0, v0)

        out0 = self.decompress0(self.patch_embedding_to_FM0(out0))
        out1 = self.decompress1(self.patch_embedding_to_FM1(out1))

        return out0, out1


class Tmp2_3modals(nn.Module):
    def __init__(self, n_embed, num_heads=None, norm_layer=nn.InstanceNorm3d):
        super().__init__()
        self.cross = MultiHeadAttention3modals(
            n_embed=n_embed,
            head_size=n_embed // num_heads,
            num_heads=num_heads,
            attention_block=EfficientCrossAttentionHead3modals,
        )
        self.norm_layer0 = norm_layer(n_embed)
        self.norm_layer1 = norm_layer(n_embed)
        self.norm_layer2 = norm_layer(n_embed)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x0, x1, x2):
        x0: torch.Tensor
        x1: torch.Tensor
        x2: torch.Tensor
        B, C, D, H, W = x0.shape
        # flatten(2) means that its start dim is 2, meaning that we preserve B, C and flatten the D, H, W dimensions
        # this creates a B, C, N matrix, where N = DxHxW
        # we then transpose it to B, N, C
        x0 = x0.flatten(2).transpose(1, 2)
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)

        x0, x1, x2 = self.cross(x0, x1, x2)

        x0 = x0.transpose(1, 2).reshape(B, C, D, H, W).contiguous()
        x1 = x1.transpose(1, 2).reshape(B, C, D, H, W).contiguous()
        x2 = x2.transpose(1, 2).reshape(B, C, D, H, W).contiguous()

        x0 = self.norm_layer0(x0)
        x1 = self.norm_layer1(x1)
        x2 = self.norm_layer2(x2)

        return x0, x1, x2
