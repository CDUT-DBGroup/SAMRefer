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

    def _compute_mask_weights(self, masks, iou_preds, text_feats, text_mask, mask_feature):
        """
        计算多个掩码的融合权重（基于IoU和文本-掩码相似度）
        
        Args:
            masks: [B, 3, H, W] 多个候选掩码（低分辨率）
            iou_preds: [B, 3] IoU预测值
            text_feats: [B, N_l, C] 文本特征
            text_mask: [B, N_l] 文本mask
            mask_feature: [B, C, H, W] 掩码特征图（高分辨率）
        
        Returns:
            mask_weights: [B, 3] 每个掩码的融合权重（已归一化）
            best_mask_idx: [B] 最佳掩码的索引
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
            mask_prob = torch.sigmoid(mask)  # 归一化到[0,1]
            # 归一化掩码权重（避免除零）
            mask_sum = mask_prob.sum(dim=(2, 3), keepdim=True) + 1e-8
            mask_prob_normalized = mask_prob / mask_sum
            
            # 计算掩码区域的特征（加权平均）
            mask_feat = (mask_feat_resized * mask_prob_normalized).sum(dim=(2, 3))  # [B, C]
            mask_features.append(mask_feat)
        
        mask_features = torch.stack(mask_features, dim=1)  # [B, num_masks, C]
        
        # 计算文本-掩码相似度
        text_global_norm = F.normalize(text_global, p=2, dim=1)  # [B, C]
        mask_features_norm = F.normalize(mask_features, p=2, dim=2)  # [B, num_masks, C]
        
        # 计算余弦相似度
        text_global_norm_float = text_global_norm.float()
        mask_features_norm_float = mask_features_norm.float()
        similarities = torch.bmm(
            text_global_norm_float.unsqueeze(1),  # [B, 1, C]
            mask_features_norm_float.transpose(1, 2)  # [B, C, num_masks]
        ).squeeze(1)  # [B, num_masks]
        similarities = similarities.to(text_global_norm.dtype)
        
        # 归一化IoU预测值到[0,1]范围
        iou_preds_float = iou_preds.float() if iou_preds.dtype != similarities.dtype else iou_preds
        iou_normalized = torch.sigmoid(iou_preds_float)  # [B, num_masks]
        if iou_normalized.dtype != similarities.dtype:
            iou_normalized = iou_normalized.to(similarities.dtype)
        
        # 结合IoU和相似度计算综合分数
        # IoU权重更高，因为它是模型直接预测的质量指标
        similarity_weight = 0.4
        iou_weight = 0.6
        combined_scores = similarity_weight * similarities + iou_weight * iou_normalized
        
        # 使用softmax计算归一化权重，使得高质量掩码获得更高权重
        # 添加温度参数，使权重分布更平滑
        temperature = 2.0
        mask_weights = F.softmax(combined_scores / temperature, dim=1)  # [B, num_masks]
        
        # 找到最佳掩码索引
        best_mask_idx = combined_scores.argmax(dim=1)  # [B]
        
        return mask_weights, best_mask_idx

    def _adaptive_multi_mask_fusion(self, masks, mask_weights, iou_preds, best_mask_idx):
        """
        多掩码智能融合策略：根据权重对所有掩码进行加权融合
        
        Args:
            masks: [B, 3, H, W] 多个候选掩码（低分辨率）
            mask_weights: [B, 3] 每个掩码的融合权重（已归一化）
            iou_preds: [B, 3] IoU预测值
            best_mask_idx: [B] 最佳掩码的索引
        
        Returns:
            fused_mask: [B, 1, H, W] 融合后的掩码
        """
        B, num_masks, H, W = masks.shape
        
        # 归一化所有掩码
        masks_norm = torch.sigmoid(masks)  # [B, 3, H, W]
        
        # 计算最佳掩码的IoU作为置信度
        best_iou = iou_preds[torch.arange(B), best_mask_idx]  # [B]
        confidence = torch.clamp(torch.sigmoid(best_iou * 2.0), 0.5, 1.0)  # [B]
        confidence = confidence.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        
        # 策略1：如果最佳掩码的IoU很高（>0.8），主要使用最佳掩码，其他掩码作为微调
        # 策略2：如果最佳掩码的IoU较低，使用加权融合利用所有掩码的互补信息
        high_confidence_threshold = 0.8
        best_iou_normalized = torch.sigmoid(best_iou * 2.0)
        use_weighted_fusion = (best_iou_normalized < high_confidence_threshold).float().view(-1, 1, 1, 1)
        
        # 基础加权融合：根据权重对所有掩码进行加权平均
        mask_weights_expanded = mask_weights.unsqueeze(-1).unsqueeze(-1)  # [B, 3, 1, 1]
        weighted_fusion = (masks_norm * mask_weights_expanded).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 最佳掩码（用于高置信度情况）
        best_mask = masks_norm[torch.arange(B), best_mask_idx].unsqueeze(1)  # [B, 1, H, W]
        
        # 自适应融合：高置信度时主要用最佳掩码，低置信度时用加权融合
        # 但即使高置信度，也加入少量其他掩码信息来细化边界
        fusion_ratio = 0.85  # 高置信度时，85%使用最佳掩码，15%使用加权融合
        fused = (best_mask * fusion_ratio + weighted_fusion * (1 - fusion_ratio)) * (1 - use_weighted_fusion) + \
                weighted_fusion * use_weighted_fusion
        
        # 边界细化：使用多个掩码的方差来识别不确定区域（边界）
        # 计算掩码间的差异，在差异大的区域（边界）进行增强
        mask_variance = masks_norm.var(dim=1, keepdim=True)  # [B, 1, H, W] 掩码间的方差
        boundary_mask = (mask_variance > 0.05).float()  # 方差大于阈值的是边界区域
        
        # 在边界区域，使用多个掩码的最大值来增强（确保边界完整）
        mask_max = masks_norm.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]
        boundary_enhancement = 0.1  # 边界增强系数
        fused = fused * (1 - boundary_mask * boundary_enhancement) + \
                mask_max * boundary_mask * boundary_enhancement
        
        # 根据置信度进行最终调整：高置信度时稍微增强，低置信度时保持原样
        confidence_adjustment = 1.0 + (confidence - 0.5) * 0.05  # 在[1.0, 1.025]范围内
        fused = fused * confidence_adjustment
        
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

        # 如果使用多掩码融合，进行智能融合
        if use_negative_masks and not self.training:
            # 计算每个掩码的融合权重
            mask_weights, best_mask_idx = self._compute_mask_weights(
                low_res_masks, iou_predictions, l_feats, l_mask, mask_feature
            )
            
            # 多掩码智能融合（返回的已经是[0,1]范围的概率值）
            fused_mask = self._adaptive_multi_mask_fusion(
                low_res_masks, mask_weights, iou_predictions, best_mask_idx
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
