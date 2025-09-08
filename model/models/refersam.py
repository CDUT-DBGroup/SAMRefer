import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from ..vit_adapter import *
import numpy as np
from PIL import Image
from torch.nn.init import trunc_normal_
from torch.cuda.amp import autocast

class ReferSAM(nn.Module):
    def __init__(self, sam_model, text_encoder, args, num_classes=1, criterion=None, **kwargs):
        super(ReferSAM, self).__init__()
        self.sam_prompt_encoder = sam_model.prompt_encoder
        self.sam_mask_decoder = sam_model.mask_decoder
        self.text_encoder = text_encoder
        self.vis_dim = sam_model.image_encoder.embed_dim
        self.lang_dim = self.text_encoder.config.hidden_size
        self.decoder_dim = self.sam_mask_decoder.transformer_dim
        #最初的 
        self.vl_adapter = ViTAdapter(sam_model.image_encoder, self.vis_dim, lang_dim=self.lang_dim, with_deconv=True, using_clip=bool(args.clip_path),**kwargs)
        self.mask_embedding = nn.Sequential(nn.Linear(self.decoder_dim, self.decoder_dim), 
                                          nn.GELU(), 
                                          nn.Linear(self.decoder_dim, self.decoder_dim))
        self.mask_scaling = nn.Conv2d(1, 1, kernel_size=1)
        self.sparse_embedding = nn.Sequential(
                nn.Linear(self.decoder_dim, self.decoder_dim),
                nn.GELU(), 
                nn.Linear(self.decoder_dim, self.decoder_dim),
                nn.LayerNorm(self.decoder_dim, eps=1e-6),
                )

        self.using_clip = bool(args.clip_path)
        self.num_classes = num_classes
        self.criterion = criterion
        self.base_lr = args.lr
        nn.init.constant_(self.mask_scaling.weight, 1.)
        nn.init.constant_(self.mask_scaling.bias, 0.)
        self.mask_embedding.apply(self._init_weights)
        self.sparse_embedding.apply(self._init_weights)
        # print(self.params_to_optimize())

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def params_to_optimize(self):
        # parameters to optimize
        # 解冻SAM image encoder的最后几层，保持前面层冻结
        trainable_param_names_sam_vit = ["blocks.11", "blocks.10", "blocks.9", "neck"]  # 解冻最后3层+neck
        if self.using_clip:
            trainable_param_names = ["vl_adapter", 
                                    "mask_embedding", "mask_scaling", "sparse_embedding",
                                    "mask_downscaling", "sam_mask_decoder"]
        else:
            trainable_param_names = ["vl_adapter", 
                                    "mask_embedding", "mask_scaling", "sparse_embedding",
                                    "sam_mask_decoder"]
        names_frozen = list()
        names_learnable = list()
        params_learnable = list()
        for name, m in self.named_parameters():
            if "vis_model" in name:
                # 解冻SAM image encoder的最后几层
                if any([x in name for x in trainable_param_names_sam_vit]):
                    m.requires_grad = True
                    names_learnable.append(name)
                    params_learnable.append(m)
                else:
                    m.requires_grad = False
                    names_frozen.append(name)
            elif any([x in name for x in trainable_param_names]):
                m.requires_grad = True
                names_learnable.append(name)
                params_learnable.append(m)
            else:
                m.requires_grad = False
                names_frozen.append(name)

        # print('LEARNABLE params: ', names_learnable)
        return params_learnable
    
    def encode_prompt(self, embedding_size, masks=None, text_embeds=None):
        bs = embedding_size[0]
        spatial_shape = (embedding_size[-2], embedding_size[-1])
        sparse_embeddings = torch.empty(
            (bs, 0, self.sam_prompt_encoder.embed_dim), device=self.sam_prompt_encoder._get_device()
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
    
    # def forward(self, img, text, l_mask, targets=None, orig_size=None, return_probs=False):
    #     '''
    #         img: [B, 3, H, W] tensor
    #         targets: [B, 1, H, W] tensor
    #         orig_size: [B, 2] numpy array or tensor (H, W)
    #     '''
    #     batch_size = img.shape[0]
    #     input_shape = img.shape[-2:]

    #     with autocast(dtype=torch.bfloat16, enabled=self.training):  # AMP context for bf16
    #         # ----------- Text Encoding (保持 float32，尤其是 BERT/CLIP 稳定性) ----------
    #         if self.using_clip:
    #             with torch.no_grad():
    #                 l_feats = self.text_encoder(text, l_mask)[0].float()
    #         else:
    #             with autocast(dtype=torch.float32):  # 强制 text_encoder 为 float32
    #                 l_feats = self.text_encoder(text, l_mask)[0]

    #         # ----------- VL pixel decoder (ViT + Adapter) -----------
    #         with autocast(dtype=torch.float32):
    #             adapter_feats_list, vit_feats, l_feats, all_prompts = self.vl_adapter(img, l_feats, l_mask)

    #         mask_feature = adapter_feats_list[0]
    #         dense_prompts = all_prompts[:, :1]
    #         sparse_prompts = all_prompts[:, 1:]

    #         dense_prompts = self.mask_embedding(dense_prompts)
    #         coarse_masks = torch.einsum('bqc,bchw->bqhw', dense_prompts, mask_feature)
    #         mask_prompt = self.mask_scaling(coarse_masks)

    #         sparse_prompts = self.sparse_embedding(sparse_prompts)
    #         sparse_embeddings, dense_embeddings, dense_pe = self.encode_prompt(
    #             vit_feats.shape,
    #             masks=mask_prompt,
    #             text_embeds=sparse_prompts
    #         )

    #         low_res_masks, iou_predictions = self.sam_mask_decoder(
    #             image_embeddings=vit_feats,
    #             image_pe=dense_pe,
    #             sparse_prompt_embeddings=sparse_embeddings.to(dense_embeddings.dtype),
    #             dense_prompt_embeddings=dense_embeddings,
    #             multimask_output=False,
    #         )

    #         low_res_masks = low_res_masks.float()  # 避免精度问题
    #         coarse_masks = coarse_masks.float()

    #     # ----------- Interpolation & Post-processing outside AMP -----------
    #     masks = F.interpolate(low_res_masks, size=input_shape, mode='bilinear', align_corners=True)
    #     pred_masks = masks.squeeze(1)
    #     coarse_masks = coarse_masks.squeeze(1)

    #     if orig_size is not None:
    #         if isinstance(orig_size, np.ndarray):
    #             orig_size = torch.from_numpy(orig_size).to(pred_masks.device)
    #         if orig_size.dtype != torch.long:
    #             orig_size = orig_size.long()
    #         pred_masks_up = []
    #         for i in range(pred_masks.shape[0]):
    #             h, w = orig_size[i]
    #             up = F.interpolate(
    #                 pred_masks[i:i+1].unsqueeze(1),
    #                 size=(h, w),
    #                 mode='bilinear',
    #                 align_corners=False
    #             ).squeeze(1)
    #             pred_masks_up.append(up)
    #         pred_masks = torch.cat(pred_masks_up, dim=0)

    #     if self.training:
    #         if self.criterion is not None:
    #             losses = self.criterion(pred_masks, targets, coarse_masks)
    #             return losses

    #     # ----------- Inference mode output -----------
    #     if not return_probs:
    #         pred_masks = pred_masks.sigmoid()
    #         pred_masks = (pred_masks >= 0.5).long()
    #     return pred_masks

    def forward(self, img, text, l_mask, targets=None, orig_size=None, return_probs=False):
        '''
            img: [B, 3, H, W] tensor [10,3,320,320]
            targets: [B, 1, H, W] tensor []
            orig_size: [B, 2] numpy array or tensor (H, W)
        '''
        batch_size = img.shape[0]
        input_shape = img.shape[-2:]

        # Text encoding
        if self.using_clip:
            with torch.no_grad():
                l_feats = self.text_encoder(text, l_mask)[0]
        else:
            l_feats = self.text_encoder(text, l_mask)[0]

        # print(img.shape)[B,3,320,320]
        # print(l_feats.shape)[B,30,768]
        # print(l_mask.shape)[B,30]
        # VL pixel decoder
        adapter_feats_list, vit_feats, l_feats, all_prompts = self.vl_adapter(img, l_feats, l_mask)

        mask_feature = adapter_feats_list[0]
        dense_prompts = all_prompts[:, :1]
        sparse_prompts = all_prompts[:, 1:]

        dense_prompts = self.mask_embedding(dense_prompts)
        coarse_masks = torch.einsum('bqc,bchw->bqhw', dense_prompts, mask_feature)
        mask_prompt = self.mask_scaling(coarse_masks)

        sparse_prompts = self.sparse_embedding(sparse_prompts)
        sparse_embeddings, dense_embeddings, dense_pe = self.encode_prompt(vit_feats.shape, masks=mask_prompt, text_embeds=sparse_prompts)

        low_res_masks, iou_predictions = self.sam_mask_decoder(
            image_embeddings=vit_feats,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings.to(dense_embeddings.dtype),
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        low_res_masks = low_res_masks.float()
        coarse_masks = coarse_masks.float()

        masks = F.interpolate(low_res_masks, size=input_shape, mode='bilinear', align_corners=True)
        pred_masks = masks.squeeze(1)
        coarse_masks = coarse_masks.squeeze(1)

        # Upsample to original size if needed
        if orig_size is not None:
            # orig_size: numpy array (B, 2) or tensor
            if isinstance(orig_size, np.ndarray):
                orig_size = torch.from_numpy(orig_size).to(pred_masks.device)
            if orig_size.dtype != torch.long:
                orig_size = orig_size.long()
            pred_masks_up = []
            for i in range(pred_masks.shape[0]):
                h, w = orig_size[i]
                up = F.interpolate(pred_masks[i:i+1].unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)
                pred_masks_up.append(up)
            pred_masks = torch.cat(pred_masks_up, dim=0)

        if self.training:
            if self.criterion is not None:
                losses = self.criterion(pred_masks, targets, coarse_masks)
                return losses

        # 在验证时返回概率值，让验证函数来处理二值化
        if not return_probs:
            pred_masks = pred_masks.sigmoid()
            pred_masks = (pred_masks >= 0.5).long()
        return pred_masks
