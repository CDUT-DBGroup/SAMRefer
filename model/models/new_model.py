import torch
from torch import nn
from torch.nn import functional as F

class ReferSAM(nn.Module):
    def __init__(self, sam_model, text_encoder, args, num_classes=1, criterion=None, **kwargs):
        super(ReferSAM, self).__init__()
        self.sam_prompt_encoder = sam_model.prompt_encoder
        self.sam_mask_decoder = sam_model.mask_decoder
        self.text_encoder = text_encoder
        self.image_encoder = sam_model.image_encoder

        self.using_clip = bool(args.clip_path)
        self.use_vl_adapter = getattr(args, "use_vl_adapter", False)

        self.num_classes = num_classes
        self.criterion = criterion
        self.base_lr = args.lr

        # === 1. 文本投影层：BERT -> SAM.prompt_encoder.dim ===
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, self.sam_prompt_encoder.embed_dim)

        # === 2. 适配器相关（你暂未使用） ===
        if self.use_vl_adapter:
            from ..vit_adapter import ViTAdapter
            self.vl_adapter = ViTAdapter(
                self.image_encoder,
                self.image_encoder.embed_dim,
                lang_dim=self.text_encoder.config.hidden_size,
                with_deconv=True,
                using_clip=self.using_clip,
                **kwargs
            )
            self.mask_embedding = nn.Sequential(
                nn.Linear(self.sam_mask_decoder.transformer_dim, self.sam_mask_decoder.transformer_dim),
                nn.GELU(),
                nn.Linear(self.sam_mask_decoder.transformer_dim, self.sam_mask_decoder.transformer_dim)
            )
            self.mask_scaling = nn.Conv2d(1, 1, kernel_size=1)
            self.sparse_embedding = nn.Sequential(
                nn.Linear(self.sam_mask_decoder.transformer_dim, self.sam_mask_decoder.transformer_dim),
                nn.GELU(),
                nn.Linear(self.sam_mask_decoder.transformer_dim, self.sam_mask_decoder.transformer_dim),
                nn.LayerNorm(self.sam_mask_decoder.transformer_dim, eps=1e-6),
            )
            nn.init.constant_(self.mask_scaling.weight, 1.)
            nn.init.constant_(self.mask_scaling.bias, 0.)
            self.mask_embedding.apply(self._init_weights)
            self.sparse_embedding.apply(self._init_weights)
        else:
            self.vl_adapter = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encode_prompt(self, embedding_size, masks=None, text_embeds=None):
        bs = embedding_size[0]
        spatial_shape = (embedding_size[-2], embedding_size[-1])

        sparse_embeddings = torch.empty(
            (bs, 0, self.sam_prompt_encoder.embed_dim),
            device=self.sam_prompt_encoder._get_device()
        )

        if text_embeds is not None:
            sparse_embeddings = torch.cat([sparse_embeddings, text_embeds], dim=1)

        if masks is not None:
            dense_embeddings = self.sam_prompt_encoder._embed_masks(masks)
        else:
            dense_embeddings = self.sam_prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, spatial_shape[0], spatial_shape[1]
            )
        dense_pe = self.sam_prompt_encoder.pe_layer(spatial_shape).unsqueeze(0)
        return sparse_embeddings, dense_embeddings, dense_pe

    def forward(self, img, text, l_mask, targets=None, return_probs=False):
        batch_size = img.shape[0]
        input_shape = img.shape[-2:]

        # === 文本编码 + 投影到 SAM 需要的维度 ===
        if self.using_clip:
            with torch.no_grad():
                l_feats = self.text_encoder(text, l_mask)[0]
        else:
            l_feats = self.text_encoder(text, l_mask)[0]

        l_feats_proj = self.text_proj(l_feats)  # [B, L, 256]

        # === 图像编码 ===
        with torch.no_grad():
            vit_feats = self.image_encoder(img)  # [B, 256, H/16, W/16]

        # === 构造 sparse prompt：文本均值向量 ===
        sparse_embeds = l_feats_proj.mean(1, keepdim=True)  # [B, 1, 256]
        sparse_embeddings, dense_embeddings, dense_pe = self.encode_prompt(
            vit_feats.shape,
            text_embeds=sparse_embeds
        )

        # === 掩膜解码 ===
        low_res_masks, iou_predictions = self.sam_mask_decoder(
            image_embeddings=vit_feats,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings.to(dtype=dense_embeddings.dtype),
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        low_res_masks = low_res_masks.float()
        masks = F.interpolate(low_res_masks, size=input_shape, mode='bilinear', align_corners=True)
        pred_masks = masks.squeeze(1)  # [B, H, W]

        if self.training and self.criterion is not None:
            losses = self.criterion(pred_masks, targets, None)
            return losses

        if not return_probs:
            pred_masks = pred_masks.sigmoid()
            pred_masks = (pred_masks >= 0.5).long()

        return pred_masks
