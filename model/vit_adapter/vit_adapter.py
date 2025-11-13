import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_
from functools import partial
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmdet.models.layers import SinePositionalEncoding

from .adapter_modules import *


class ViTAdapter(nn.Module):
    def __init__(self, vis_model, vis_dim, lang_dim, vl_dim, num_prompts=[10, 8], conv_inplane=64, n_points=4, deform_ratio=1.0, 
                 deform_num_heads=6, interaction_indexes=None, with_cffn=True, init_values=0.,
                 cffn_ratio=0.25, add_vit_feature=False, drop_path_rate=0., dropout=0.,
                 with_cp=False, with_deconv=True, num_extra_layers=-1, num_prompt_layers=2, using_clip=True):
        
        super().__init__()
        self.using_clip = using_clip
        self.vis_model = vis_model
        self.drop_path_rate = drop_path_rate
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.with_deconv = with_deconv
        embed_dim = vis_dim
        out_dim = self.vis_model.out_chans
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.postional_encoding = SinePositionalEncoding(num_feats=embed_dim//2, normalize=True)
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False, norm_layer=partial(nn.GroupNorm, 1, eps=1e-6), use_c1_proj=False)
        self.lang_prompts = nn.Embedding(num_prompts[0], lang_dim)
        self.interactions = nn.Sequential(*[
            InteractionBlock(embed_dim, lang_dim, vl_dim=vl_dim, num_heads=deform_num_heads, vl_heads=deform_num_heads, 
                             n_points=n_points, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                             drop=dropout, drop_path=self.drop_path_rate, 
                             with_cffn=with_cffn, cffn_ratio=cffn_ratio, init_values=init_values,
                             deform_ratio=deform_ratio, with_cp=with_cp,
                             num_extra_layers=(num_extra_layers if (i == len(interaction_indexes) - 1) else -1))
            for i in range(len(interaction_indexes))
        ])
        self.adapter_proj = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, out_dim), 
                                          nn.LayerNorm(out_dim, eps=1e-6)) for i in range(3)])
        self.lang_proj = nn.Sequential(nn.Linear(lang_dim, out_dim), 
                                          nn.GELU(), 
                                          nn.Linear(out_dim, out_dim),
                                          nn.LayerNorm(out_dim, eps=1e-6))

        self.sparse_prompts = nn.Embedding(num_prompts[1], out_dim)
        self.prompt_pos = nn.Embedding(1+num_prompts[1], out_dim)
        self.level_embed_prompter = nn.Parameter(torch.zeros(3, out_dim))
        self.postional_encoding_prompter = SinePositionalEncoding(num_feats=out_dim//2, normalize=True)
        # 可学习的权重参数，用于组合基础token和加权聚合的文本特征
        # 初始化为接近原始权重0.7和0.3的值（通过logit空间：log(0.7/0.3) ≈ 0.85）
        self.lang_fusion_weights = nn.Parameter(torch.tensor([0.85, -0.85]))
        self.prompt_blocks = nn.Sequential(*[
            PromptAttnLayer(out_dim, out_dim, out_dim, heads=8, mlp_ratio=4, n_levels=3, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
            for i in range(num_prompt_layers)
        ])

        # 改进：多尺度特征融合模块（用于mask_feature，更直接有效）
        # mask_feature用于生成coarse_masks，作为SAM decoder的dense prompt，直接影响最终结果
        # 融合c1, c2, c3三个尺度的特征，提升mask_feature的质量
        self.mask_feature_fusion = nn.Sequential(
            nn.Conv2d(out_dim * 3, out_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_dim, eps=1e-6),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_dim, eps=1e-6)
        )
        self.mask_feature_fusion.apply(self._init_weights)
        
        # 可学习的残差连接系数（初始化为接近0，确保训练初期不影响原始特征）
        # 使用sigmoid确保系数在(0, 1)范围内，初始值约为0.1
        # logit(0.1) ≈ -2.2，所以初始化为-2.2
        self.mask_feature_fusion_alpha = nn.Parameter(torch.tensor(-2.2))

        if with_deconv:
            self.c1_conv = nn.Conv2d(conv_inplane, out_dim, kernel_size=1, bias=False)
            self.c1_norm = nn.GroupNorm(1, out_dim, eps=1e-6)
            self.out_conv = nn.Sequential(
                                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                                nn.GroupNorm(1, out_dim, eps=1e-6),
                                nn.GELU(),
                                nn.Conv2d(out_dim, out_dim, kernel_size=1))
            self.c1_conv.apply(self._init_weights)
            self.c1_norm.apply(self._init_weights)
            self.out_conv.apply(self._init_weights)

        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.prompt_blocks.apply(self._init_weights)
        self.adapter_proj.apply(self._init_weights)
        self.lang_proj.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)
        normal_(self.level_embed_prompter)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MultiScaleDeformableAttention):
            m.init_weights() # init_weights defined in MultiScaleDeformableAttention

    def _get_lvl_pos_embed(self, B, H, W):
        dtype = self.level_embed.dtype
        lvl_pos_emb_list = []
        for i in range(3):
            r = 2 ** (i + 3)
            pos_embed = self.postional_encoding(self.level_embed.new_zeros((B, H//r, W//r), dtype=torch.bool)).to(dtype)
            curr_lvl_pos_emb = self.level_embed[i] + pos_embed.flatten(2).permute(0, 2, 1)
            lvl_pos_emb_list.append(curr_lvl_pos_emb)
        lvl_pos_emb = torch.cat(lvl_pos_emb_list, dim=1)
        return lvl_pos_emb
    
    def _get_lvl_pos_embed_prompter(self, B, H, W):
        dtype = self.level_embed_prompter.dtype
        lvl_pos_emb_list = []
        for i in range(3):
            r = 2 ** (i + 3)
            pos_embed = self.postional_encoding_prompter(self.level_embed_prompter.new_zeros((B, H//r, W//r), dtype=torch.bool)).to(dtype)
            curr_lvl_pos_emb = self.level_embed_prompter[i] + pos_embed.flatten(2).permute(0, 2, 1)
            lvl_pos_emb_list.append(curr_lvl_pos_emb)
        lvl_pos_emb = torch.cat(lvl_pos_emb_list, dim=1)
        return lvl_pos_emb

    def forward(self, x, lang_feats, lang_mask):
        H, W = x.shape[-2:]
        h, w = H // 16, W // 16
        # ViT Patch Embedding forward
        vit_feats = self.vis_model.patch_embed(x) # [B, vit_h, vit_w, C]
        bs, vit_h, vit_w, dim = vit_feats.shape
        if self.vis_model.pos_embed is not None:
            dtype = vit_feats.dtype
            absolute_pos_embed = F.interpolate(self.vis_model.pos_embed.float().permute(0,3,1,2), size=(vit_h, vit_w), mode='bilinear')
            vit_feats = vit_feats + absolute_pos_embed.to(dtype).permute(0,2,3,1).contiguous()
        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        adapter_feats = torch.cat([c2, c3, c4], dim=1) # [B, HW/8^2+HW/16^2+HW/32^2, D]
        
        # 从实际的特征图形状计算空间尺寸（而不是使用整数除法）
        # c2, c3, c4 已经被 reshape 为 [B, N, D]，其中 N = H*W
        # 由于特征图通常是正方形，使用 sqrt 计算空间尺寸
        c2_h = c2_w = int(math.sqrt(c2.shape[1]))
        c3_h = c3_w = int(math.sqrt(c3.shape[1]))
        c4_h = c4_w = int(math.sqrt(c4.shape[1]))
        actual_spatial_shapes = [(c2_h, c2_w), (c3_h, c3_w), (c4_h, c4_w)]
        
        lvl_pos_emb = self._get_lvl_pos_embed(bs, H, W)
        # Interaction - 使用实际的特征图尺寸
        deform_inputs1, deform_inputs2 = deform_inputs(H, W, vit_h, vit_w, x.device, 
                                                        actual_spatial_shapes=actual_spatial_shapes)
        prompts = self.lang_prompts.weight.unsqueeze(0).expand(bs, -1, -1) # [B, P, C]
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            vit_feats, adapter_feats, lang_feats, prompts = layer(vit_feats, adapter_feats, lvl_pos_emb,
                         lang_feats, lang_mask, prompts,
                         self.vis_model.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, h, w)
        lang_feats = self.lang_proj(lang_feats)

        # Split & Reshape
        c2 = adapter_feats[:, 0:c2.size(1), :] # [B, HW/8^2, D]
        c3 = adapter_feats[:, c2.size(1):c2.size(1) + c3.size(1), :] # [B, HW/16^2, D]
        c4 = adapter_feats[:, c2.size(1) + c3.size(1):, :] # [B, HW/32^2, D]
        c2 = self.adapter_proj[0](c2)
        c3 = self.adapter_proj[1](c3)
        c4 = self.adapter_proj[2](c4)
        adapter_feats = torch.cat([c2, c3, c4], dim=1)

        c2 = c2.transpose(1, 2).view(bs, -1, h * 2, w * 2).contiguous() # 1/8
        c3 = c3.transpose(1, 2).view(bs, -1, h, w).contiguous() # 1/16
        c4 = c4.transpose(1, 2).view(bs, -1, h // 2, w // 2).contiguous() # 1/32
        adapter_feats_list = [c2, c3, c4]
        if self.with_deconv:
            c1 = self.c1_norm(self.c1_conv(c1))
            c1 = c1 + F.interpolate(c2.float(), size=c1.shape[-2:], mode='bilinear', align_corners=False).to(c1.dtype) # 1/4
            c1 = self.out_conv(c1)
            adapter_feats_list = [c1,] + adapter_feats_list

        # Prompt aggregation
        lvl_pos_emb_prompter = self._get_lvl_pos_embed_prompter(bs, H, W)
        # if self.using_clip:
        #     eos_index = lang_mask.sum(1).long() - 1
        #     lang_g = lang_feats[torch.arange(bs), eos_index].unsqueeze(1) # [B, 1, C], [EOS] embeddings
        # else:
        #     lang_g = lang_feats[:, 0].unsqueeze(1) # [B, 1, C], [CLS] embeddings

        # # dense_prompts = lang_g.clone().detach()
        # dense_prompts = lang_g.clone()
        # 不使用单个token，而是使用注意力聚合
# 在 model/vit_adapter/vit_adapter.py 第166-172行
        if self.using_clip:
            eos_index = lang_mask.sum(1).long() - 1
            lang_g_base = lang_feats[torch.arange(bs), eos_index].unsqueeze(1)
        else:
            lang_g_base = lang_feats[:, 0].unsqueeze(1)

        # 使用注意力聚合所有token（更充分利用文本信息）
        text_attn_weights = F.softmax(lang_feats.mean(-1), dim=1)  # [B, N_l]
        lang_g_weighted = (lang_feats * text_attn_weights.unsqueeze(-1)).sum(1).unsqueeze(1)  # [B, 1, C]

        # 结合基础token和加权聚合（使用可学习权重）
        fusion_weights = F.softmax(self.lang_fusion_weights, dim=0)  # [2] -> 归一化为和为1的权重
        lang_g = fusion_weights[0] * lang_g_base + fusion_weights[1] * lang_g_weighted

        dense_prompts = lang_g  # 不需要detach，也不需要clone（因为后面会concat）
        sparse_prompts = self.sparse_prompts.weight.unsqueeze(0).expand(bs, -1, -1) # [B, P, C]
        prompt_pos = self.prompt_pos.weight.unsqueeze(0).expand(bs, -1, -1) # [B, 1+P, C]
        all_prompts = torch.cat([dense_prompts, sparse_prompts], dim=1)

        for i, layer in enumerate(self.prompt_blocks):
            all_prompts = layer(all_prompts, adapter_feats, lang_feats, lang_mask, spatial_shapes=deform_inputs1[1], prompt_pos=prompt_pos, vis_pos=lvl_pos_emb_prompter)

        # ViT neck
        vit_feats = vit_feats.permute(0, 3, 1, 2)
        vit_feats = self.vis_model.neck(vit_feats)  # [B, out_chans, h, w]

        # 改进：多尺度特征融合用于mask_feature（更直接有效）
        # mask_feature用于生成coarse_masks，作为SAM decoder的dense prompt，直接影响最终结果
        # 获取c1的尺寸（用于mask_feature）
        c1_h, c1_w = adapter_feats_list[0].shape[-2:] if self.with_deconv else (h * 2, w * 2)
        
        # 将c2, c3上采样到c1的分辨率
        c2_to_c1 = F.interpolate(c2, size=(c1_h, c1_w), mode='bilinear', align_corners=True)
        c3_to_c1 = F.interpolate(c3, size=(c1_h, c1_w), mode='bilinear', align_corners=True)
        
        # 如果with_deconv，c1已经在adapter_feats_list[0]中
        if self.with_deconv:
            c1_feat = adapter_feats_list[0]
        else:
            # 如果没有deconv，需要从c2生成c1
            c1_feat = F.interpolate(c2, size=(c1_h, c1_w), mode='bilinear', align_corners=True)
        
        # 融合多尺度特征用于mask_feature
        multi_scale_mask_feats = torch.cat([c1_feat, c2_to_c1, c3_to_c1], dim=1)  # [B, out_dim*3, c1_h, c1_w]
        enhanced_mask_feature = self.mask_feature_fusion(multi_scale_mask_feats)  # [B, out_dim, c1_h, c1_w]
        
        # 残差连接融合到mask_feature（使用可学习系数控制融合强度）
        # 使用sigmoid确保系数在(0, 1)范围内，初始值约为0.1，训练过程中可学习最优值
        alpha = torch.sigmoid(self.mask_feature_fusion_alpha)  # 初始值≈0.1，范围(0, 1)
        adapter_feats_list[0] = adapter_feats_list[0] + alpha * enhanced_mask_feature

        return adapter_feats_list, vit_feats, lang_feats, all_prompts