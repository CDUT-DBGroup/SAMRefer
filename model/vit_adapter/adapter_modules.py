from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from timm.models.layers import DropPath

from ..tranformer_decoder import *


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(h1, w1, h2, w2, device, actual_spatial_shapes=None):
    """
    Args:
        h1, w1: 输入图像的高度和宽度
        h2, w2: ViT特征图的高度和宽度
        device: 设备
        actual_spatial_shapes: 可选，实际的特征图空间尺寸列表 [(h2, w2), (h3, w3), (h4, w4)]
                              如果提供，将使用这些实际尺寸而不是整数除法计算
    """
    if actual_spatial_shapes is not None:
        # 使用实际的特征图尺寸
        spatial_shapes = torch.as_tensor(actual_spatial_shapes, dtype=torch.long, device=device)
    else:
        # 使用整数除法（向后兼容）
        spatial_shapes = torch.as_tensor([(h1 // 8, w1 // 8),
                                          (h1 // 16, w1 // 16),
                                          (h1 // 32, w1 // 32)],
                                         dtype=torch.long, device=device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h2, w2)], device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]
    
    spatial_shapes = torch.as_tensor([(h2, w2)], dtype=torch.long, device=device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    if actual_spatial_shapes is not None:
        reference_points = get_reference_points(actual_spatial_shapes, device)
    else:
        reference_points = get_reference_points([(h1 // 8, w1 // 8),
                                                 (h1 // 16, w1 // 16),
                                                 (h1 // 32, w1 // 32)], device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    
    return deform_inputs1, deform_inputs2


def deform_inputsv2(h1, w1, h2, w2, device):
    spatial_shapes = torch.as_tensor([(h1 // 8, w1 // 8),
                                      (h1 // 16, w1 // 16),
                                      (h1 // 32, w1 // 32)],
                                     dtype=torch.long, device=device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    # reference_points = get_reference_points([(h2, w2)], device)
    reference_points = get_reference_points([(h1 // 8, w1 // 8),
                                             (h1 // 16, w1 // 16),
                                             (h1 // 32, w1 // 32)], device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]
    
    spatial_shapes = torch.as_tensor([(h2, w2)], dtype=torch.long, device=device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h1 // 8, w1 // 8),
                                             (h1 // 16, w1 // 16),
                                             (h1 // 32, w1 // 32)], device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    
    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False, norm_layer=nn.SyncBatchNorm, use_c1_proj=True):
        super().__init__()
        self.with_cp = with_cp
        self.use_c1_proj = use_c1_proj

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        if use_c1_proj:
            self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        
        def _inner_forward(x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            if self.use_c1_proj:
                c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)
    
            bs, dim, _, _ = c2.shape
            # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s
    
            return c1, c2, c3, c4
        
        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs


class Extractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False, v_pre_norm=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        if v_pre_norm:
            self.value_norm = norm_layer(dim)
        self.attn = MultiScaleDeformableAttention(embed_dims=dim, num_levels=n_levels, num_heads=num_heads, dropout=drop,
                                                 num_points=n_points, value_proj_ratio=deform_ratio, batch_first=True)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        self.v_pre_norm = v_pre_norm
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W, query_pos=None):
        
        def _inner_forward(query, feat):
            dtype = query.dtype
            if self.v_pre_norm:
                value = self.value_norm(feat)
            else:
                value = feat
            self.attn.float()
            attn = self.attn(query=query.to(torch.float), query_pos=query_pos, 
                             value=value.to(torch.float), key_padding_mask=None, 
                             reference_points=reference_points.to(torch.float), spatial_shapes=spatial_shapes,
                             level_start_index=level_start_index)
            query = attn.to(dtype)
            query = self.query_norm(query)
            if self.with_cffn:
                query = query + self.drop_path(self.ffn(query, H, W))
                query = self.ffn_norm(query)
            return query
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
            
        return query


class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.attn = MultiScaleDeformableAttention(embed_dims=dim, num_levels=n_levels, num_heads=num_heads, dropout=0.,
                                 num_points=n_points, value_proj_ratio=deform_ratio, batch_first=True)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        
        def _inner_forward(query, feat):
            dtype = query.dtype
            self.attn.float()
            attn = self.attn(query=self.query_norm(query).to(torch.float), identity=torch.zeros_like(query, dtype=torch.float32, requires_grad=False), 
                             value=feat.to(torch.float), key_padding_mask=None,
                             reference_points=reference_points.to(torch.float), spatial_shapes=spatial_shapes,
                             level_start_index=level_start_index,)
            return query + self.gamma * attn.to(dtype)
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
            
        return query


class InteractionBlock(nn.Module):
    def __init__(self, dim, lang_dim, vl_dim=1024, num_heads=6, vl_heads=16, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, with_cp=False, num_extra_layers=-1):
        super().__init__()
        self.num_extra_layers = num_extra_layers
        self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                         norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=False,
                                         cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp, v_pre_norm=True)
        self.vl_extractor = VLBiAttnLayer(dim, lang_dim, vl_dim, vl_heads, n_levels=3, mlp_ratio=cffn_ratio, dropout=drop, norm_layer=norm_layer, with_gamma=True, with_post_norm=True)
        if num_extra_layers > 0:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                            norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=False,
                            cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp, v_pre_norm=True)
                for i in range(num_extra_layers)
            ])   
            self.extra_vl_extractors = nn.Sequential(*[
                VLBiAttnLayer(dim, lang_dim, vl_dim, vl_heads, n_levels=3, mlp_ratio=cffn_ratio,  dropout=drop, norm_layer=norm_layer, with_gamma=True, with_post_norm=True)
                for i in range(num_extra_layers)
            ])
        else:
            self.extra_extractors = None
            self.extra_vl_extractors = None

    def forward(self, vit_feats, adapter_feats, lvl_pos_emb, lang_feats, lang_mask, prompts, blocks, deform_inputs1, deform_inputs2, H, W):
        b, h, w, c = vit_feats.shape
        vit_feats = self.injector(query=vit_feats.flatten(1,2), reference_points=deform_inputs1[0],
                          feat=adapter_feats, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        vit_feats = vit_feats.reshape(b, h, w, c)

        for idx, blk in enumerate(blocks):
            vit_feats = blk(vit_feats)

        adapter_feats = self.extractor(query=adapter_feats, reference_points=deform_inputs2[0],
                                             feat=vit_feats.flatten(1,2), spatial_shapes=deform_inputs2[1],
                                             level_start_index=deform_inputs2[2], H=H, W=W)

        adapter_feats, lang_feats, prompts = self.vl_extractor(adapter_feats, lang_feats, prompts, lang_mask=lang_mask, 
                                                                                 reference_points=deform_inputs2[0], spatial_shapes=deform_inputs1[1], level_start_index=deform_inputs1[2], vis_pos=lvl_pos_emb)

        if self.num_extra_layers > 0:
            for i in range(self.num_extra_layers):
                adapter_feats = self.extra_extractors[i](query=adapter_feats, reference_points=deform_inputs2[0],
                                                    feat=vit_feats.flatten(1,2), spatial_shapes=deform_inputs2[1],
                                                    level_start_index=deform_inputs2[2], H=H, W=W)
                adapter_feats, lang_feats, prompts = self.extra_vl_extractors[i](adapter_feats, lang_feats, prompts, lang_mask=lang_mask, 
                                                                                 reference_points=deform_inputs2[0], spatial_shapes=deform_inputs1[1], level_start_index=deform_inputs1[2], vis_pos=lvl_pos_emb)

        return vit_feats, adapter_feats, lang_feats, prompts


class MultiScaleFusion(nn.Module):
    """
    增强的多尺度特征融合模块
    使用跨尺度注意力和可学习权重进行智能融合
    注意：这个模块主要用于增强特征，最终仍然返回拼接后的特征以保持兼容性
    """
    def __init__(self, dim, num_scales=3, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_scales = num_scales
        self.dim = dim
        
        # 跨尺度注意力机制 - 每个尺度都能关注其他尺度
        self.cross_scale_attns = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_scales)
        ])
        
        # 可学习的尺度权重
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        
        # 特征增强（对每个尺度独立增强）
        self.enhance_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim, eps=1e-6),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim)
            ) for _ in range(num_scales)
        ])
        
        # 残差连接的权重（初始化为较小值，避免过度平滑）
        self.residual_weights = nn.Parameter(torch.ones(num_scales) * 0.3)
        
        # 边界保护机制：使用门控控制融合强度
        self.boundary_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.Sigmoid()
            ) for _ in range(num_scales)
        ])
        
    def forward(self, feats_list):
        """
        Args:
            feats_list: List of [B, N_i, D] tensors, where N_i is the number of tokens at scale i
        Returns:
            enhanced_feats_list: List of enhanced features, same structure as input
        Note: 这个模块增强每个尺度的特征，但保持原始结构不变
        """
        enhanced_feats = []
        
        for i, feat in enumerate(feats_list):
            # 1. 跨尺度注意力融合（使用门控保护边界信息）
            other_feats = [feats_list[j] for j in range(len(feats_list)) if j != i]
            if len(other_feats) > 0:
                # 将其他尺度特征拼接
                other_feats_cat = torch.cat(other_feats, dim=1)  # [B, sum(N_j), D]
                # 跨尺度注意力
                attn_out, _ = self.cross_scale_attns[i](
                    query=feat,
                    key=other_feats_cat,
                    value=other_feats_cat
                )
                # 使用门控控制融合强度，保护原始特征（特别是边界信息）
                gate = self.boundary_gates[i](feat.mean(dim=1, keepdim=True))  # [B, 1, D]
                attn_out = attn_out * gate
                # 残差连接（使用较小的权重，避免过度平滑）
                enhanced_feat = feat + self.residual_weights[i] * attn_out
            else:
                enhanced_feat = feat
            
            # 2. 特征增强（使用较小的权重，避免过度修改）
            enhanced_feat = enhanced_feat + 0.5 * self.enhance_layers[i](enhanced_feat)
            
            enhanced_feats.append(enhanced_feat)
        
        # 3. 可学习权重加权（可选，这里我们保持原始特征，只在需要时加权）
        # scale_weights_norm = torch.softmax(self.scale_weights, dim=0)
        # weighted_feats = [feat * weight for feat, weight in zip(enhanced_feats, scale_weights_norm)]
        
        # 返回增强后的特征列表（保持原始结构）
        # 注意：为了保持兼容性，我们仍然需要返回拼接后的特征
        # 但这里我们增强每个尺度后再拼接
        return enhanced_feats


class EnhancedC1C2Fusion(nn.Module):
    """
    增强的c1和c2融合模块
    使用空间注意力和门控融合替代简单的上采样相加
    注意：使用空间注意力而非序列注意力，避免内存爆炸
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        
        # 使用空间注意力（Channel Attention + Spatial Attention）替代序列注意力
        # 这样可以避免对256x256的特征做65536x65536的注意力矩阵计算
        
        # Channel Attention: 关注哪些通道更重要
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial Attention: 使用轻量级卷积注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=7, padding=3, groups=dim),
            nn.GroupNorm(1, dim),
            nn.Sigmoid()
        )
        
        # c2特征增强（用于更好的融合）
        self.c2_enhance = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GroupNorm(1, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        )
        
        # 门控融合机制
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.GroupNorm(1, dim),
            nn.Sigmoid()
        )
        
        # 特征增强（轻量级，避免与out_conv重复）
        # 注意：out_conv 已经有完整的特征处理，这里只做轻量级增强
        self.enhance = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GroupNorm(1, dim),
            nn.GELU()
            # 移除最后的 Conv2d，避免与 out_conv 重复
        )
        
        # 可学习的融合权重
        self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))
        
    def forward(self, c1, c2):
        """
        Args:
            c1: [B, C, H1, W1] - 1/4尺度特征
            c2: [B, C, H2, W2] - 1/8尺度特征，需要上采样到c1的尺寸
        Returns:
            fused_c1: [B, C, H1, W1] - 融合后的特征
        """
        B, C, H1, W1 = c1.shape
        _, _, H2, W2 = c2.shape
        
        # 1. 将c2上采样到c1的尺寸
        c2_up = F.interpolate(c2.float(), size=(H1, W1), mode='bilinear', align_corners=False).to(c1.dtype)
        
        # 2. 增强c2特征
        c2_enhanced = self.c2_enhance(c2_up)
        
        # 3. 使用空间注意力机制融合（避免序列注意力的内存问题）
        concat_feat = torch.cat([c1, c2_enhanced], dim=1)  # [B, 2*C, H1, W1]
        
        # Channel Attention: 关注重要通道
        channel_attn = self.channel_attention(concat_feat)  # [B, C, 1, 1]
        c1_channel = c1 * channel_attn
        c2_channel = c2_enhanced * channel_attn
        
        # Spatial Attention: 关注重要空间位置
        spatial_attn = self.spatial_attention(concat_feat)  # [B, C, H1, W1]
        c1_spatial = c1_channel * spatial_attn
        c2_spatial = c2_channel * spatial_attn
        
        # 4. 加权融合
        fusion_weights = F.softmax(self.fusion_weight, dim=0)
        fused = fusion_weights[0] * c1_spatial + fusion_weights[1] * c2_spatial
        
        # 5. 门控融合
        gate = self.gate(concat_feat)  # [B, C, H1, W1]
        fused = fused * gate + c1 * (1 - gate)  # 门控残差连接
        
        # 6. 轻量级特征增强（主要增强细节，最终处理由out_conv完成）
        enhanced = self.enhance(fused)
        fused_c1 = fused + 0.5 * enhanced  # 使用较小的权重，避免与out_conv重复处理
        
        return fused_c1