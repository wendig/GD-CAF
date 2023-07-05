import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_base import Precipitation_base

dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel=(3, 3)):
        super().__init__()

        mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel, padding='same', bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.double_conv.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.double_conv(x)


class CreateHeads(nn.Module):
    """
    Create attention heads
        x:      Batch, Vertices, Time, Height, Width
        :return Batch, Vertices, Time, Heads, Height, Width
    """

    def __init__(self, in_channels, out_channels, pooling, kernel_size=(3, 3)):
        super(CreateHeads, self).__init__()
        # DoubleConv with conditional pooling layer
        layers = []
        if pooling:
            layers.append(nn.MaxPool2d(2))
        layers.append(DoubleConv(in_channels, out_channels, kernel=kernel_size))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        B, V, T, H, W = x.size()
        x = x.view(B * V, T, H, W)  # concat Vertices and Time
        x = self.conv(x)
        _, _, h, w = x.size()
        x = x.view(B, V, T, -1, h, w)
        return x


class CombineHeads(nn.Module):
    """
    Combine attention heads and time steps
        x:      Batch, Vertices, Time, Heads, Height, Width
        :return Batch, Vertices, Height, Width
    """

    def __init__(self, in_channels, out_channels, pooling, kernel_size=(3, 3)):
        super(CombineHeads, self).__init__()
        # DoubleConv with conditional upsampling layer
        layers = []
        if pooling:
            layers.append(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=(2, 2), stride=(2, 2)))
        else:
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding='same', bias=False)) ##########
        layers.extend([
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding='same', bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        B, V, T, Heads, H, W = x.size()
        # [Batch * Vertices, Time * Heads, Height, Width]
        x = x.view(B * V, T * Heads, H, W)  # (To apply 2d conv only on images)
        x = self.conv(x)  # Combine Heads
        # [Batch, Time, Vertices, H, W]
        _, _, h, w = x.size()
        x = x.view(B, V, h, w)
        return torch.squeeze(x)


class DownDS(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernels_per_layer=2, heads=1, use_polling=False):
        super().__init__()
        # DoubleConvDS with conditional pooling layer
        layers = []
        if use_polling:
            layers.append(nn.MaxPool2d(2))
        layers.append(DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer, heads=heads))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        # [Batch, Vertices, Time, Heads, Height, Width]
        B, V, T, Heads, H, W = x.size()
        # [Batch * Vertices, Time * Heads, Height, Width]
        x = x.view(B * V, T * Heads, H, W)
        x = self.conv(x)
        _, _, h, w = x.size()
        x = x.view(B, V, -1, Heads, h, w)
        return x


class UpDS(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernels_per_layer=2, heads=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=(2, 2), stride=(2, 2))
        self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer, heads=heads)

    def forward(self, x):
        # [Batch, Vertices, Time, Heads, Height, Width]
        B, V, T, Heads, H, W = x.size()
        # [Batch * Vertices, Time * Heads, Height, Width]
        x = x.view(B * V, T * Heads, H, W)  # (To apply 2d conv only on images)
        x = self.up(x)
        x = self.conv(x)
        # permute back to T and V
        _, _, h, w = x.size()
        x = x.view(B, V, T, Heads, h, w)
        return x


class DoubleConvDS(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernels_per_layer=2, heads=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1, heads=heads),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1, heads=heads),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=2, heads=1):
        super(DepthwiseSeparableConv, self).__init__()
        # In Tensorflow DepthwiseConv2D has depth_multiplier instead of kernels_per_layer
        self.depthwise = nn.Conv2d(in_channels, in_channels * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=(1, 1), groups=heads)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SpatialAttention(nn.Module):
    """
    Spatial Attention mechanism
    X:              [Batch, Vertices, Time, K_heads, Height, Width]
    K:              number of attention heads
    return:         [Batch, Vertices, Time, K_heads, Height, Width]
    """

    def __init__(self, in_channels, out_channels, kpl, heads, poll_qk=True, poll_v=False, _index=0):
        super(SpatialAttention, self).__init__()
        self.c_q = DownDS(in_channels, out_channels, kpl, heads, use_polling=poll_qk)
        self.c_k = DownDS(in_channels, out_channels, kpl, heads, use_polling=poll_qk)
        self.c_v = DownDS(in_channels, out_channels, kpl, heads, use_polling=poll_v)

        if poll_v:
            self.c = UpDS(out_channels, out_channels, kpl)
        else:
            self.c = DownDS(out_channels, out_channels, kpl)

        self.m = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.1)
        self._index = _index
        self.count = 0

    def forward(self, X):
        B, V, T, K, H, W = X.size()

        # [Batch, Vertices, Time, K_heads, Height, Width]
        query = self.c_q(X)
        key = self.c_k(X)
        value = self.c_v(X)
        _, _, _, _, h, w = value.size()

        # [K * Batch, Time, Vertices, Height * Width]
        query = query.permute(0, 3, 2, 1, 4, 5).contiguous().view(B * K, T, V, -1)
        key = key.permute(0, 3, 2, 1, 4, 5).contiguous().view(B * K, T, V, -1)
        value = value.permute(0, 3, 2, 1, 4, 5).contiguous().view(B * K, T, V, -1)

        # [K * Batch, Time, Vertices, Vertices]
        attention = torch.matmul(query, key.transpose(2, 3))
        attention = self.m(attention)
        attention = F.softmax(attention, dim=-1)

        # [K * Batch, Time, Vertices, Height * Width]
        X = torch.matmul(attention, value)

        # [Batch, Vertices, Time, K, Height, Width]
        X = X.view(B, K, T, V, h, w).permute(0, 3, 2, 1, 4, 5).contiguous()
        X = self.dropout(X)
        X = self.c(X)
        return X


################################################################################

class TemporalAttention(nn.Module):
    '''
    temporal attention mechanism
    X:      [Batch, Vertices, Time, Heads, Height, Width]
    K:      number of attention heads
    return: [Batch, Vertices, Time, Heads, Height, Width]
    '''

    def __init__(self, in_channels, out_channels, kpl, heads, mask=True, poll_qk=True, poll_v=False, _index=0):
        super(TemporalAttention, self).__init__()
        self.mask = mask

        self.c_q = DownDS(in_channels, out_channels, kpl, heads, use_polling=poll_qk)
        self.c_k = DownDS(in_channels, out_channels, kpl, heads, use_polling=poll_qk)
        self.c_v = DownDS(in_channels, out_channels, kpl, heads, use_polling=poll_v)

        if poll_v:
            self.c = UpDS(out_channels, out_channels, kpl)
        else:
            self.c = DownDS(out_channels, out_channels, kpl)

        self.m = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.1)
        self._index = _index
        self.count = 0

    def forward(self, X):
        B, V, T, K, H, W = X.size()

        # X:  [Batch, Vertices, Time, Heads, Height, Width]
        query = self.c_q(X)
        key = self.c_k(X)
        value = self.c_v(X)
        _, _, _, _, h, w = value.size()

        # query: [K * Batch, Vertices, Time, Height*Width]
        query = query.permute(0, 2, 3, 1, 4, 5).contiguous().view(B * K, V, T, -1)
        # Key:   [K * Batch, Vertices, Height*Width, Time]
        key = key.permute(0, 2, 5, 1, 3, 4).contiguous().view(B * K, V, -1, T)
        # Attention: [K * Batch, Vertices, Time, Time]
        attention = torch.matmul(query, key)
        # value: [K * Batch, Vertices, Time, Height*Width]
        value = value.permute(0, 2, 3, 1, 4, 5).contiguous().view(B * K, V, T, -1)

        # mask attention score
        if self.mask:
            batch_size = X.shape[0]
            mask = torch.ones(T, T).to(dev)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(K * batch_size, V, 1, 1)
            mask = mask.to(torch.bool)
            attention = torch.where(mask, attention, -2 ** 15 + 1)
        # softmax
        attention = self.m(attention)
        attention = F.softmax(attention, dim=-1)
        # [Batch * K, Vertices, Time, Height*Width]
        X = torch.matmul(attention, value)

        # [Batch, Vertices, Vertices, K, H, W]
        X = X.view(B, K, V, T, h, w).permute(0, 3, 2, 1, 4, 5).contiguous()
        X = X.permute(0, 2, 1, 3, 4, 5).contiguous()
        X = self.dropout(X)
        X = self.c(X)
        return X


class GatedFusion(nn.Module):
    """
    Gated Fusion
        P:          [Batch, Vertices, Time, K, Height, Width]   (Spatial Attention)
        R:          [Batch, Vertices, Time, K, Height, Width]   (Temporal Attention)
        :return:    [Batch, Vertices, Time, K, Height, Width]
    """
    def __init__(self, in_channels, out_channels, kpl):
        super(GatedFusion, self).__init__()
        self.conv = DoubleConvDS(in_channels * 2, out_channels, kernels_per_layer=kpl)

    def forward(self, P, R):
        # [Batch, Vertices, Time, 2*K, Height, Width]
        X = torch.cat((P, R), 3)
        B, V, T, double_heads, H, W = X.size()
        # [Batch * Vertices, Time * 2*K, Height, Width]
        X = X.view(B * V, double_heads * T, H, W)
        X = self.conv(X)
        # [Batch, Vertices, Time, K, Height, Width]
        X = X.view(B, V, T, double_heads // 2, H, W)
        return X


class STAttBlock(nn.Module):
    def __init__(self, hp, in_channels, out_channels, _index):
        super(STAttBlock, self).__init__()
        self.spatialAttention = SpatialAttention(in_channels, out_channels, kpl=hp.kernels_per_layer, heads=hp.K, poll_qk=hp.poll_qk, poll_v=hp.poll_v, _index=_index)
        self.temporalAttention = TemporalAttention(in_channels, out_channels, kpl=hp.kernels_per_layer, heads=hp.K, mask=True, poll_qk=hp.poll_qk, poll_v=hp.poll_v, _index=_index)
        self.gatedFusion = GatedFusion(in_channels, out_channels, kpl=hp.kernels_per_layer)

    def forward(self, X):
        P = self.spatialAttention(X)
        R = self.temporalAttention(X)
        H = self.gatedFusion(P, R)

        return torch.add(X, H)


class GDCAF(Precipitation_base):
    """
    GDCAF
        X：         [Batch, Time, Vertices, Height, Width]
        L：         number of STAtt blocks
        K：         number of attention heads
        T:          number of input time steps
        poll_input: Use pooling on the observations directly
        poll_qk:    Use pooling in spatial and temporal attention when calculating Query and Key
        poll_v:     Use pooling in spatial and temporal attention when calculating Value
        return：  [Batch, Vertices, Height, Width]
    """

    def __init__(self, hparams):
        super(GDCAF, self).__init__(hparams=hparams)
        T = hparams.num_input_images
        K = hparams.K
        self.create_heads = CreateHeads(in_channels=T, out_channels=T*K, pooling=hparams.poll_input, kernel_size=(3, 3))
        self.blocks = nn.ModuleList([STAttBlock(hparams, T*K, T*K, _) for _ in range(hparams.L)])
        self.combine_heads = CombineHeads(in_channels=T*K, out_channels=1, pooling=hparams.poll_input, kernel_size=(3, 3))

    def forward(self, X):
        # [Batch, Vertices, Time, Height, Width]
        X = X.permute(0, 2, 1, 3, 4).contiguous()
        # [Batch, Vertices, Time, Heads, Height, Width]
        X = self.create_heads(X)

        for one_block in self.blocks:
            X = one_block(X)

        # [Batch, Vertices, Height, Width]
        X = self.combine_heads(X)
        return X
