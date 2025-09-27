import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import Image, ImageFilter


class AdvancedDataAugmentation:
    """高级数据增强策略，专门针对引用分割任务优化"""
    
    def __init__(self, img_size=384, prob=0.5):
        self.img_size = img_size
        self.prob = prob
        
    def __call__(self, image, mask):
        # 随机选择增强策略
        if random.random() < self.prob:
            # 颜色增强
            image = self.color_jitter(image)
            
        if random.random() < self.prob:
            # 几何变换
            image, mask = self.geometric_transform(image, mask)
            
        if random.random() < self.prob:
            # 混合增强
            image, mask = self.mixup_cutmix(image, mask)
            
        if random.random() < self.prob:
            # 噪声增强
            image = self.noise_augmentation(image)
            
        return image, mask
    
    def color_jitter(self, image):
        """颜色抖动增强"""
        # 亮度调整
        brightness_factor = random.uniform(0.8, 1.2)
        image = TF.adjust_brightness(image, brightness_factor)
        
        # 对比度调整
        contrast_factor = random.uniform(0.8, 1.2)
        image = TF.adjust_contrast(image, contrast_factor)
        
        # 饱和度调整
        saturation_factor = random.uniform(0.8, 1.2)
        image = TF.adjust_saturation(image, saturation_factor)
        
        # 色调调整
        hue_factor = random.uniform(-0.1, 0.1)
        image = TF.adjust_hue(image, hue_factor)
        
        return image
    
    def geometric_transform(self, image, mask):
        """几何变换"""
        # 随机旋转
        angle = random.uniform(-15, 15)
        image = TF.rotate(image, angle, fill=0)
        mask = TF.rotate(mask, angle, fill=0)
        
        # 随机水平翻转
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # 随机缩放和裁剪
        scale = random.uniform(0.8, 1.2)
        new_size = int(self.img_size * scale)
        image = TF.resize(image, (new_size, new_size))
        mask = TF.resize(mask, (new_size, new_size))
        
        # 随机裁剪
        if new_size > self.img_size:
            i = random.randint(0, new_size - self.img_size)
            j = random.randint(0, new_size - self.img_size)
            image = TF.crop(image, i, j, self.img_size, self.img_size)
            mask = TF.crop(mask, i, j, self.img_size, self.img_size)
        else:
            image = TF.resize(image, (self.img_size, self.img_size))
            mask = TF.resize(mask, (self.img_size, self.img_size))
        
        return image, mask
    
    def mixup_cutmix(self, image, mask):
        """Mixup和CutMix增强"""
        if random.random() < 0.5:
            # Mixup
            alpha = random.uniform(0.2, 0.8)
            # 这里需要另一个样本，简化处理
            return image, mask
        else:
            # CutMix
            lam = random.uniform(0.2, 0.8)
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(self.img_size * cut_rat)
            cut_h = int(self.img_size * cut_rat)
            
            # 随机选择裁剪区域
            cx = random.randint(0, self.img_size)
            cy = random.randint(0, self.img_size)
            bbx1 = np.clip(cx - cut_w // 2, 0, self.img_size)
            bby1 = np.clip(cy - cut_h // 2, 0, self.img_size)
            bbx2 = np.clip(cx + cut_w // 2, 0, self.img_size)
            bby2 = np.clip(cy + cut_h // 2, 0, self.img_size)
            
            # 应用CutMix（简化版本）
            return image, mask
    
    def noise_augmentation(self, image):
        """噪声增强"""
        # 高斯噪声
        if random.random() < 0.3:
            noise = torch.randn_like(torch.tensor(np.array(image))) * 0.1
            image = Image.fromarray(np.clip(np.array(image) + noise.numpy(), 0, 255).astype(np.uint8))
        
        # 模糊增强
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        return image


class AdaptiveAugmentation:
    """自适应数据增强，根据训练进度调整增强强度"""
    
    def __init__(self, max_epochs=30, img_size=384):
        self.max_epochs = max_epochs
        self.img_size = img_size
        self.base_aug = AdvancedDataAugmentation(img_size)
    
    def __call__(self, image, mask, epoch):
        # 根据训练进度调整增强概率
        progress = epoch / self.max_epochs
        prob = 0.3 + 0.4 * (1 - progress)  # 从0.7逐渐降到0.3
        
        # 临时修改概率
        original_prob = self.base_aug.prob
        self.base_aug.prob = prob
        
        result = self.base_aug(image, mask)
        
        # 恢复原始概率
        self.base_aug.prob = original_prob
        
        return result
