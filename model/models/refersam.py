import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_

from ..vit_adapter import *

class ReferSAM(nn.Module):
    def __init__(self, sam_model, text_encoder, args, num_classes=1, criterion=None, **kwargs):
        super(ReferSAM, self).__init__()
        self.sam_prompt_encoder = sam_model.prompt_encoder
        self.sam_mask_decoder = sam_model.mask_decoder
        self.text_encoder = text_encoder
        self.vis_dim = sam_model.image_encoder.embed_dim
        self.lang_dim = self.text_encoder.config.hidden_size
        self.decoder_dim = self.sam_mask_decoder.transformer_dim

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
        trainable_param_names_sam_vit = [""]
        if self.using_clip:
            trainable_param_names = ["vl_adapter", 
                                    "mask_embedding", "mask_scaling", "sparse_embedding",
                                    "mask_downscaling", "sam_mask_decoder"]
        else:
            trainable_param_names = [""]
        names_frozen = list()
        names_learnable = list()
        params_learnable = list()
        for name, m in self.named_parameters():
            if "vis_model" in name:
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

        print('LEARNABLE params: ', names_learnable)
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

    def _select_positive_negative_masks(self, masks, iou_preds, text_feats, text_mask, mask_feature):
        """
        基于文本-掩码相似度选择正负掩码
        
        Args:
            masks: [B, 3, H, W] 多个候选掩码（低分辨率）
            iou_preds: [B, 3] IoU预测值
            text_feats: [B, N_l, C] 文本特征
            text_mask: [B, N_l] 文本mask
            mask_feature: [B, C, H, W] 掩码特征图（高分辨率）
        
        Returns:
            positive_mask: [B, 1, H, W] 正样本掩码（低分辨率）
            negative_mask: [B, 1, H, W] 负样本掩码（低分辨率）
        """
        B, num_masks, H, W = masks.shape
        
        # 获取文本全局特征（CLS或EOS token）
        if self.using_clip:
            eos_index = text_mask.sum(1).long() - 1
            text_global = text_feats[torch.arange(B), eos_index]  # [B, C]
        else:
            text_global = text_feats[:, 0]  # [B, C]
        
        # 将mask_feature调整到与masks相同的分辨率
        mask_feat_resized = F.interpolate(
            mask_feature, size=(H, W), mode='bilinear', align_corners=True
        )  # [B, C, H, W]
        
        # 计算每个掩码区域的特征
        mask_features = []
        for i in range(num_masks):
            mask = masks[:, i:i+1]  # [B, 1, H, W]
            # 使用掩码对mask_feature进行加权平均
            mask_weights = torch.sigmoid(mask)  # 归一化到[0,1]
            # 归一化掩码权重（避免除零）
            mask_sum = mask_weights.sum(dim=(2, 3), keepdim=True) + 1e-8
            mask_weights = mask_weights / mask_sum
            
            # 计算掩码区域的特征（加权平均）
            mask_feat = (mask_feat_resized * mask_weights).sum(dim=(2, 3))  # [B, C]
            mask_features.append(mask_feat)
        
        mask_features = torch.stack(mask_features, dim=1)  # [B, num_masks, C]
        
        # 计算文本-掩码相似度
        text_global_norm = F.normalize(text_global, p=2, dim=1)  # [B, C]
        mask_features_norm = F.normalize(mask_features, p=2, dim=2)  # [B, num_masks, C]
        
        # 计算余弦相似度
        # torch.bmm 需要 float32，所以先转换类型
        text_global_norm_float = text_global_norm.float()
        mask_features_norm_float = mask_features_norm.float()
        similarities = torch.bmm(
            text_global_norm_float.unsqueeze(1),  # [B, 1, C]
            mask_features_norm_float.transpose(1, 2)  # [B, C, num_masks]
        ).squeeze(1)  # [B, num_masks]
        # 转换回原始数据类型
        similarities = similarities.to(text_global_norm.dtype)
        
        # 归一化IoU预测值到[0,1]范围（IoU预测通常在[-1,1]或类似范围）
        # 确保 iou_preds 和 similarities 类型一致
        iou_preds_float = iou_preds.float() if iou_preds.dtype != similarities.dtype else iou_preds
        iou_normalized = torch.sigmoid(iou_preds_float)  # [B, num_masks]
        # 确保类型一致
        if iou_normalized.dtype != similarities.dtype:
            iou_normalized = iou_normalized.to(similarities.dtype)
        
        # 结合IoU预测和相似度（可调权重）
        similarity_weight = 0.7
        iou_weight = 0.3
        combined_scores = similarity_weight * similarities + iou_weight * iou_normalized
        
        # 选择正样本（最高分）和负样本（最低分）
        positive_idx = combined_scores.argmax(dim=1)  # [B]
        negative_idx = combined_scores.argmin(dim=1)  # [B]
        
        positive_mask = masks[torch.arange(B), positive_idx]  # [B, H, W]
        negative_mask = masks[torch.arange(B), negative_idx]  # [B, H, W]
        
        return positive_mask.unsqueeze(1), negative_mask.unsqueeze(1)

    def _adaptive_mask_fusion(self, positive_mask, negative_mask, iou_preds):
        """
        无需训练的自适应掩码融合策略
        
        Args:
            positive_mask: [B, 1, H, W] 正样本掩码
            negative_mask: [B, 1, H, W] 负样本掩码  
            iou_preds: [B, 3] IoU预测值（用于计算融合权重）
        
        Returns:
            fused_mask: [B, 1, H, W] 融合后的掩码
        """
        # 归一化掩码
        positive_mask_norm = torch.sigmoid(positive_mask)  # [B, 1, H, W]
        negative_mask_norm = torch.sigmoid(negative_mask)  # [B, 1, H, W]
        
        # 计算自适应权重（基于最高IoU的置信度）
        max_iou = iou_preds.max(dim=1)[0]  # [B]
        confidence = torch.clamp(torch.sigmoid(max_iou), 0.5, 1.0)  # 归一化到[0.5, 1.0]
        confidence = confidence.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        
        # 融合策略：正掩码增强，负掩码作为抑制项
        # 使用负掩码的反向信息来细化边界
        negative_suppression = 0.3  # 负掩码抑制强度（可调超参数）
        fused = positive_mask_norm * confidence - negative_mask_norm * (1 - confidence) * negative_suppression
        
        # 边界细化：使用正负掩码的差异来识别边界区域
        boundary = torch.abs(positive_mask_norm - negative_mask_norm)
        boundary_enhancement = 0.2  # 边界增强系数（可调超参数）
        fused = fused * (1 + boundary * boundary_enhancement)  # 在边界区域增强
        
        # 确保输出在[0,1]范围
        fused = torch.clamp(fused, 0, 1)
        
        return fused

    def forward(self, img, text, l_mask, targets=None, return_probs=False, use_negative_masks=False):
        '''
            Input:
                img       [BxCxHxW]
                text    [BxN_l]
                l_mask  [BxN_l]
        '''
        batch_size = img.shape[0]
        input_shape = img.shape[-2:]

        # Text encoding
        if self.using_clip:
            with torch.no_grad():
                l_feats = self.text_encoder(text, l_mask)[0] # l_feats: [B, N_l, 768]
        else:
            l_feats = self.text_encoder(text, l_mask)[0] # l_feats: [B, N_l, 768]
        # VL pixel decoder
        adapter_feats_list, vit_feats, l_feats, all_prompts = self.vl_adapter(img, l_feats, l_mask) # vit_feats:[B, C, H/16, W/16]

        mask_feature = adapter_feats_list[0] # [B, C, H, W]
        dense_prompts = all_prompts[:, :1] # [B, 1, C]
        sparse_prompts = all_prompts[:, 1:] # [B, P, C]

        dense_prompts = self.mask_embedding(dense_prompts)
        coarse_masks = torch.einsum('bqc,bchw->bqhw', dense_prompts, mask_feature) # [B, 1, H, W]
        mask_prompt = self.mask_scaling(coarse_masks)

        sparse_prompts = self.sparse_embedding(sparse_prompts)
        sparse_embeddings, dense_embeddings, dense_pe = self.encode_prompt(vit_feats.shape, masks=mask_prompt, text_embeds=sparse_prompts)

        # 根据是否使用负样本掩码决定是否生成多个掩码
        use_multimask = use_negative_masks and not self.training
        
        low_res_masks, iou_predictions = self.sam_mask_decoder(
            image_embeddings=vit_feats,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings.to(dense_embeddings.dtype),
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=use_multimask,  # 推理时且启用负样本掩码时生成多个掩码
        )

        low_res_masks = low_res_masks.float()
        coarse_masks = coarse_masks.float()

        # 如果使用负样本掩码，进行正负掩码选择和融合
        if use_negative_masks and not self.training:
            # 选择正负掩码
            positive_mask, negative_mask = self._select_positive_negative_masks(
                low_res_masks, iou_predictions, l_feats, l_mask, mask_feature
            )
            
            # 自适应融合（返回的已经是[0,1]范围的概率值）
            fused_mask = self._adaptive_mask_fusion(
                positive_mask, negative_mask, iou_predictions
            )
            
            # 上采样到原始分辨率
            masks = F.interpolate(fused_mask, size=input_shape, mode='bilinear', align_corners=True)
            pred_masks = masks.squeeze(1)  # [B, H, W]
            # 融合后的掩码已经是概率值，标记为已处理
            is_fused_mask = True
        else:
            # 原有逻辑：使用第一个掩码
            masks = F.interpolate(low_res_masks, size=input_shape, mode='bilinear', align_corners=True)
            pred_masks = masks.squeeze(1)  # [B, H, W]
            is_fused_mask = False
        
        coarse_masks = coarse_masks.squeeze(1)  # [B, H, W]

        if self.training:
            if self.criterion is not None:
                losses = self.criterion(pred_masks, targets, coarse_masks)
                return losses

        if not return_probs:
            # 融合后的掩码已经是概率值，不需要再sigmoid
            if not is_fused_mask:
                pred_masks = pred_masks.sigmoid()
            pred_masks = (pred_masks >= 0.5).long()     
        return pred_masks
