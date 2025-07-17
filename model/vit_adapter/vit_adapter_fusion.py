import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from torch.nn.init import normal_
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmdet.models.layers import SinePositionalEncoding
# ===== 模块依赖部分（需替换为你已有工程中的定义）=====
from model.vit_adapter.adapter_modules import *

# ===== 双向融合模块定义 =====
class BiResidualFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv3to2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, in_channels),
            nn.GELU()
        )
        self.conv4to3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, in_channels),
            nn.GELU()
        )
        self.out_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, in_channels),
            nn.GELU()
        )
        self.out_conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, in_channels),
            nn.GELU()
        )
        self.out_conv4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, in_channels),
            nn.GELU()
        )

    def forward(self, c2, c3, c4):
        c3_from_c4 = F.interpolate(self.conv4to3(c4), size=c3.shape[-2:], mode='bilinear', align_corners=False)
        c3_fused = c3 + c3_from_c4
        c2_from_c3 = F.interpolate(self.conv3to2(c3_fused), size=c2.shape[-2:], mode='bilinear', align_corners=False)
        c2_fused = c2 + c2_from_c3
        c2_fused = self.out_conv2(c2_fused)
        c3_fused = self.out_conv3(c3_fused)
        c4_fused = self.out_conv4(c4)
        return c2_fused, c3_fused, c4_fused

# ===== ViTAdapter 模块定义（含双向融合） =====
class ViTAdapterWithBiFusion(nn.Module):
    def __init__(self, vis_model, vis_dim, lang_dim, vl_dim=768, num_prompts=[10, 8], conv_inplane=64, n_points=4,
                 deform_ratio=1.0, deform_num_heads=6, interaction_indexes=[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]],
                 with_cffn=True, init_values=0., cffn_ratio=0.25, add_vit_feature=False, drop_path_rate=0.,
                 dropout=0., with_cp=False, with_deconv=True, num_extra_layers=-1, num_prompt_layers=2,
                 using_clip=True):
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
        self.postional_encoding = SinePositionalEncoding(num_feats=embed_dim // 2, normalize=True)

        self.spm = SpatialPriorModule(conv_inplane, embed_dim, with_cp=False,
                                      norm_layer=partial(nn.GroupNorm, 1, eps=1e-6), use_c1_proj=False)

        self.lang_prompts = nn.Embedding(num_prompts[0], lang_dim)

        self.interactions = nn.Sequential(*[
            InteractionBlock(embed_dim, lang_dim, vl_dim=vl_dim, num_heads=deform_num_heads,
                             vl_heads=deform_num_heads, n_points=n_points,
                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
                             drop=dropout, drop_path=self.drop_path_rate,
                             with_cffn=with_cffn, cffn_ratio=cffn_ratio,
                             init_values=init_values, deform_ratio=deform_ratio,
                             with_cp=with_cp, num_extra_layers=(num_extra_layers if (i == len(interaction_indexes) - 1) else -1))
            for i in range(len(interaction_indexes))
        ])

        self.adapter_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, out_dim), nn.LayerNorm(out_dim, eps=1e-6))
            for _ in range(3)
        ])

        self.lang_proj = nn.Sequential(
            nn.Linear(lang_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim, eps=1e-6)
        )

        self.sparse_prompts = nn.Embedding(num_prompts[1], out_dim)
        self.prompt_pos = nn.Embedding(1 + num_prompts[1], out_dim)
        self.level_embed_prompter = nn.Parameter(torch.zeros(3, out_dim))
        self.postional_encoding_prompter = SinePositionalEncoding(num_feats=out_dim // 2, normalize=True)

        self.prompt_blocks = nn.Sequential(*[
            PromptAttnLayer(out_dim, out_dim, out_dim, heads=8, mlp_ratio=4, n_levels=3,
                            norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(num_prompt_layers)
        ])

        self.bidirectional_fusion = BiResidualFusion(out_dim)

        if with_deconv:
            self.c1_conv = nn.Conv2d(conv_inplane, out_dim, kernel_size=1, bias=False)
            self.c1_norm = nn.GroupNorm(1, out_dim, eps=1e-6)
            self.out_conv = nn.Sequential(
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(1, out_dim, eps=1e-6),
                nn.GELU(),
                nn.Conv2d(out_dim, out_dim, kernel_size=1)
            )
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
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MultiScaleDeformableAttention):
            m.init_weights()

    def _get_lvl_pos_embed(self, B, H, W):
        lvl_pos_emb_list = []
        for i in range(3):
            r = 2 ** (i + 3)
            pos_embed = self.postional_encoding(self.level_embed.new_zeros((B, H // r, W // r), dtype=torch.bool))
            curr_lvl_pos_emb = self.level_embed[i] + pos_embed.flatten(2).permute(0, 2, 1)
            lvl_pos_emb_list.append(curr_lvl_pos_emb)
        return torch.cat(lvl_pos_emb_list, dim=1)

    def _get_lvl_pos_embed_prompter(self, B, H, W):
        lvl_pos_emb_list = []
        for i in range(3):
            r = 2 ** (i + 3)
            pos_embed = self.postional_encoding_prompter(self.level_embed_prompter.new_zeros((B, H // r, W // r), dtype=torch.bool))
            curr_lvl_pos_emb = self.level_embed_prompter[i] + pos_embed.flatten(2).permute(0, 2, 1)
            lvl_pos_emb_list.append(curr_lvl_pos_emb)
        return torch.cat(lvl_pos_emb_list, dim=1)

    def forward(self, x, lang_feats, lang_mask):
        B, _, H, W = x.shape
        h, w = H // 16, W // 16
        vit_feats = self.vis_model.patch_embed(x)
        bs, vit_h, vit_w, _ = vit_feats.shape
        if self.vis_model.pos_embed is not None:
            absolute_pos_embed = F.interpolate(self.vis_model.pos_embed.float().permute(0, 3, 1, 2),
                                               size=(vit_h, vit_w), mode='bilinear')
            vit_feats = vit_feats + absolute_pos_embed.to(vit_feats.dtype).permute(0, 2, 3, 1)

        c1, c2, c3, c4 = self.spm(x)
        adapter_feats = torch.cat([c2, c3, c4], dim=1)
        lvl_pos_emb = self._get_lvl_pos_embed(B, H, W)

        deform_inputs1, deform_inputs2 = deform_inputs(H, W, vit_h, vit_w, x.device)
        prompts = self.lang_prompts.weight.unsqueeze(0).expand(B, -1, -1)
        for i, layer in enumerate(self.interactions):
            idx = self.interaction_indexes[i]
            vit_feats, adapter_feats, lang_feats, prompts = layer(
                vit_feats, adapter_feats, lvl_pos_emb, lang_feats, lang_mask, prompts,
                self.vis_model.blocks[idx[0]:idx[-1] + 1], deform_inputs1, deform_inputs2, h, w)

        lang_feats = self.lang_proj(lang_feats)
        c2 = self.adapter_proj[0](adapter_feats[:, 0:c2.size(1), :])
        c3 = self.adapter_proj[1](adapter_feats[:, c2.size(1):c2.size(1) + c3.size(1), :])
        c4 = self.adapter_proj[2](adapter_feats[:, c2.size(1) + c3.size(1):, :])

        # 还原为特征图形式并融合
        c2_map = c2.transpose(1, 2).reshape(B, -1, h * 2, w * 2) # [2,256,40,40]
        c3_map = c3.transpose(1, 2).reshape(B, -1, h, w) # [2,256,20,20]
        c4_map = c4.transpose(1, 2).reshape(B, -1, h // 2, w // 2) # [2,256,10,10]
        c2_map, c3_map, c4_map = self.bidirectional_fusion(c2_map, c3_map, c4_map)
        adapter_feats_list = [c2_map, c3_map, c4_map]

        # === 新增：融合后还原为token序列，拼接为adapter_feats，供后续prompt block使用 ===
        c2_tokens = c2_map.flatten(2).transpose(1, 2)  # [B, HW/8^2, C] 这个对应原本的c2、c3、c4
        c3_tokens = c3_map.flatten(2).transpose(1, 2)  # [B, HW/16^2, C]
        c4_tokens = c4_map.flatten(2).transpose(1, 2)  # [B, HW/32^2, C]
        adapter_feats = torch.cat([c2_tokens, c3_tokens, c4_tokens], dim=1)  # [B, HW/8^2+HW/16^2+HW/32^2, C] [2,2100,256]

        if self.with_deconv:
            c1 = self.c1_norm(self.c1_conv(c1))
            c1 = c1 + F.interpolate(c2_map.float(), size=c1.shape[-2:], mode='bilinear', align_corners=False).to(c1.dtype)
            c1 = self.out_conv(c1)
            adapter_feats_list = [c1,] + adapter_feats_list

        lvl_pos_emb_prompter = self._get_lvl_pos_embed_prompter(B, H, W)
        if self.using_clip:
            eos_index = lang_mask.sum(1).long() - 1
            lang_g = lang_feats[torch.arange(B), eos_index].unsqueeze(1)
        else:
            lang_g = lang_feats[:, 0].unsqueeze(1)

        dense_prompts = lang_g.clone().detach()
        sparse_prompts = self.sparse_prompts.weight.unsqueeze(0).expand(B, -1, -1)
        prompt_pos = self.prompt_pos.weight.unsqueeze(0).expand(B, -1, -1)
        all_prompts = torch.cat([dense_prompts, sparse_prompts], dim=1)

        for layer in self.prompt_blocks:
            all_prompts = layer(all_prompts, adapter_feats, lang_feats, lang_mask,
                                spatial_shapes=deform_inputs1[1], prompt_pos=prompt_pos, vis_pos=lvl_pos_emb_prompter)

        vit_feats = vit_feats.permute(0, 3, 1, 2)
        vit_feats = self.vis_model.neck(vit_feats)
        vit_feats = vit_feats + c3_map

        return adapter_feats_list, vit_feats, lang_feats, all_prompts


if __name__ == "__main__":
    import torch
    from transformers import BertTokenizer, BertModel
    from model.segment_anything import sam_model_registry
    from get_args import get_args
    args = get_args()
    # 1. 构造真实SAM backbone
    sam_type = "vit_b"  # 你可以改成vit_l/vit_h
    sam_model = sam_model_registry[sam_type](checkpoint=args.checkpoint)
    sam_model.eval()

    # 2. 构造真实BERT文本编码器
    tokenizer = BertTokenizer.from_pretrained(args.ck_bert)
    text_encoder = BertModel.from_pretrained(args.ck_bert)
    text_encoder.eval()

    # 3. 构造一组假文本
    texts = ["a cat on the mat", "a dog in the park"]
    tokens = tokenizer(texts, padding="max_length", max_length=30, truncation=True, return_tensors="pt")
    word_ids = tokens["input_ids"]  # [B, L]
    word_masks = tokens["attention_mask"]  # [B, L]

    # 4. 构造一组假图片
    B, C, H, W = 2, 3, 320, 320
    img = torch.randn(B, C, H, W)

    # 5. 获取文本特征
    with torch.no_grad():
        lang_feats = text_encoder(input_ids=word_ids, attention_mask=word_masks)[0]  # [B, L, 768]

    # 6. 构造ViTAdapter
    from model.vit_adapter.vit_adapter_fusion import ViTAdapterWithBiFusion
    adapter_kwargs = dict(
        vl_dim=768,
        num_prompts=[16, 4],
        conv_inplane=64,
        n_points=4,
        deform_ratio=0.5,
        deform_num_heads=12,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        with_cffn=True,
        init_values=1e-6,
        cffn_ratio=2.0,
        add_vit_feature=False,
        drop_path_rate=0.0,
        dropout=0.0,
        with_cp=False,
        with_deconv=True,
        num_extra_layers=2,
        num_prompt_layers=2,
        using_clip=False
    )
    vis_model = sam_model.image_encoder
    model = ViTAdapterWithBiFusion(vis_model, vis_dim=768, lang_dim=768, **adapter_kwargs)
    model.eval()

    with torch.no_grad():
        out = model(img, lang_feats, word_masks)
        print("ViTAdapter 返回格式：")
        for i, o in enumerate(out):
            print(f"output[{i}]: shape={o.shape}, dtype={o.dtype}")

