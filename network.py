import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from einops import rearrange
import math
from swin_transformer import SwinTransformer3D

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        d_model,
        n_cls,
        dropout=0.1,
        drop_path_rate=0.0,
        channels=4,
        swin_depths=[2,2],
        swin_heads=[3,3]
    ):
        super().__init__()
        self.patch_embed = SwinTransformer3D( pretrained=None,
                                      pretrained2d=True,
                                      patch_size=(patch_size,patch_size,patch_size),
                                      in_chans=channels,
                                      embed_dim=d_model//(2**(len(swin_depths)-1)),
                                      depths=swin_depths,
                                      num_heads=swin_heads,
                                      window_size=(8, 8, 8),
                                      mlp_ratio=4.,
                                      qkv_bias=True,
                                      qk_scale=None,
                                      drop_rate=0.,
                                      attn_drop_rate=0.,
                                      drop_path_rate=0.2,
                                      norm_layer=nn.LayerNorm,
                                      patch_norm=False,
                                      frozen_stages=-1,
                                      use_checkpoint=False)
        self.patch_size = patch_size
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.apply(init_weights)

    def forward(self, im):
        B, _, D, H, W = im.shape
        PS = self.patch_size

        x = self.patch_embed(im)
        x = self.dropout(x)
        x = self.norm(x)
        return x

def create_segmenter(model_cfg):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop("decoder")
    decoder_cfg["n_cls"] = model_cfg["n_cls"]
    decoder_cfg['n_layers'] = len(model_cfg['swin_depths'])
    encoder = create_vit(model_cfg)
    decoder = create_decoder(encoder, decoder_cfg)
    model = Segmenter(encoder, decoder, n_cls=model_cfg["n_cls"])

    return model

def create_vit(model_cfg):
    model_cfg = model_cfg.copy()
    model = VisionTransformer(**model_cfg)
    return model

def create_decoder(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    decoder_cfg["d_encoder"] = encoder.d_model
    decoder_cfg["patch_size"] = encoder.patch_size

    if "linear" in name:
        decoder = DecoderLinear(**decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder

class DecoderLinear(nn.Module):
    def __init__(self, n_cls, patch_size, d_encoder, n_layers):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls
        self.n_layers = n_layers
        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    def forward(self, x, im_size):
        D, H, W= im_size
        GS = D // self.patch_size // (2**(self.n_layers-1))
        WS = H // self.patch_size // (2**(self.n_layers-1))
        x = self.head(x)
        x = rearrange(x, "b (d h w) c -> b c d h w", d=GS, h=WS)

        return x

def padding(im, patch_size, fill_value=0):
    # make the image sizes divisible by patch_size
    D, H, W = im.size(2), im.size(3), im.size(4)
    pad_h, pad_w, pad_d = 0, 0, 0
    if H % patch_size > 0:
        pad_h = patch_size - (H % patch_size)
    if W % patch_size > 0:
        pad_w = patch_size - (W % patch_size)
    if D % patch_size > 0:
        pad_d = patch_size - (D % patch_size)
    im_padded = im
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        im_padded = F.pad(im, (0, pad_w, 0, pad_h, 0, pad_d), value=fill_value)
    return im_padded

def unpadding(y, target_size):
    D, H, W = target_size
    D_pad, H_pad, W_pad = y.size(2),y.size(3),y.size(4)
    # crop predictions on extra pixels coming from padding
    extra_h = H_pad - H
    extra_w = W_pad - W
    extra_d = D_pad - D
    if extra_d > 0:
        y = y[:, :, :-extra_d]
    if extra_h > 0:
        y = y[:, :, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :, :-extra_w]

    return y

class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder
        self.numuplayers = round(math.log2(self.patch_size))
        self.output = nn.Linear(n_cls + 4, n_cls)

    def forward(self, im):
        D_ori, H_ori, W_ori = im.size(2), im.size(3), im.size(4)
        im = padding(im, self.patch_size * (2 ** (self.decoder.n_layers - 1)))
        D, H, W = im.size(2), im.size(3), im.size(4)

        x = self.encoder(im)

        masks = self.decoder(x, (D, H, W))
        masks = F.interpolate(masks, size=(D, H, W), mode="trilinear")
        masks = torch.concat([masks,im],dim=1)
        masks = masks.permute(0,2,3,4,1)
        masks = self.output(masks)
        masks = masks.permute(0,4,1,2,3)
        masks = unpadding(masks, (D_ori, H_ori, W_ori))

        return masks