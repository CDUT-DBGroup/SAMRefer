"""
模型推理可视化脚本
将原图和推理后的框叠加显示在一起
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import argparse

# 配置matplotlib支持中文显示
try:
    # 尝试设置中文字体
    import matplotlib.font_manager as fm
    # 查找可用的中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 
                     'Noto Sans CJK SC', 'STHeiti', 'Arial Unicode MS']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    font_found = False
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            font_found = True
            break
    if not font_found:
        # 如果没有找到中文字体，使用默认字体并给出警告
        print("Warning: No Chinese font found. Chinese characters may display as squares.")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']
except Exception as e:
    print(f"Warning: Failed to configure Chinese font: {e}")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

from dataset.ReferDataset import ReferDataset
from dataset.GRefDataset import GRefDataset
from dataset.Dataset_referit import ReferitDataset
from model.enhanced_builder import refersam, load_model_with_checkpoint
from validation.evaluation import get_bbox_from_mask
import yaml
from types import SimpleNamespace


def denormalize_image(image_tensor):
    """
    将归一化的图像张量还原为可显示的图像
    """
    image = image_tensor.cpu().clone()
    # 确保是 [C, H, W] 格式
    if image.dim() == 4:
        image = image[0]
    # ImageNet归一化的逆操作
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    return image


def visualize_inference_with_bbox(image_tensor, pred_mask, sentence, save_path, 
                                   gt_mask=None, gt_bbox=None, show_gt=False, show_bbox=False):
    """
    可视化推理结果，将原图和推理后的mask叠加显示
    
    Args:
        image_tensor: 图像张量 [C, H, W]
        pred_mask: 预测的mask [H, W]
        sentence: 文本描述
        save_path: 保存路径
        gt_mask: 可选的ground truth mask
        gt_bbox: 可选的ground truth bounding box [x1, y1, x2, y2]
        show_gt: 是否显示ground truth
        show_bbox: 是否显示bounding box（默认False，语义分割任务通常不需要）
    """
    # 处理图像
    if isinstance(image_tensor, torch.Tensor):
        image = denormalize_image(image_tensor)
    else:
        image = np.array(image_tensor)
        if image.max() > 1.0:
            image = image / 255.0
        if len(image.shape) == 3 and image.shape[0] == 3:
            # 如果是 [C, H, W] 格式，转换为 [H, W, C]
            image = image.transpose(1, 2, 0)
    
    # 处理预测mask
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if pred_mask.ndim == 3:
        pred_mask = pred_mask.squeeze(0)
    if pred_mask.ndim == 2:
        pred_mask_bool = pred_mask > 0.5  # 二值化
    else:
        pred_mask_bool = pred_mask.astype(bool)
    
    # 从mask中提取bounding box（仅在需要显示box时计算）
    pred_bbox = None
    if show_bbox:
        pred_mask_tensor = torch.from_numpy(pred_mask_bool.astype(float))
        pred_bbox = get_bbox_from_mask(pred_mask_tensor)
        pred_bbox = pred_bbox.numpy().astype(int)
    
    # 创建图形
    if show_gt and gt_mask is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes = [axes[0], axes[1]]
    
    # 第一个子图：原图
    axes[0].imshow(image)
    axes[0].set_title('original image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 第二个子图：原图 + 预测mask叠加
    axes[1].imshow(image)
    
    # 绘制预测的mask（半透明叠加）
    if pred_mask_bool.sum() > 0:
        # 创建彩色mask
        color_mask = np.zeros_like(image)
        color_mask[:, :, 0] = 1.0  # 红色通道
        color_mask[:, :, 1] = 0.0
        color_mask[:, :, 2] = 0.0
        axes[1].imshow(color_mask, alpha=0.3 * pred_mask_bool.astype(float))
        
        # 可选：绘制预测的bounding box
        if show_bbox and pred_bbox is not None:
            if pred_bbox[2] > pred_bbox[0] and pred_bbox[3] > pred_bbox[1]:
                rect = Rectangle(
                    (pred_bbox[0], pred_bbox[1]),
                    pred_bbox[2] - pred_bbox[0],
                    pred_bbox[3] - pred_bbox[1],
                    linewidth=3,
                    edgecolor='red',
                    facecolor='none',
                    label='预测框'
                )
                axes[1].add_patch(rect)
    
    axes[1].set_title(f'inference result overlay\n"{sentence}"', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    if pred_mask_bool.sum() > 0 and show_bbox and pred_bbox is not None:
        axes[1].legend(loc='upper right', fontsize=10)
    
    # 第三个子图（如果显示ground truth）
    if show_gt and gt_mask is not None:
        axes[2].imshow(image)
        
        # 处理ground truth mask
        if isinstance(gt_mask, torch.Tensor):
            gt_mask_np = gt_mask.cpu().numpy()
        else:
            gt_mask_np = np.array(gt_mask)
        if gt_mask_np.ndim == 3:
            gt_mask_np = gt_mask_np.squeeze(0)
        gt_mask_bool = gt_mask_np > 0.5
        
        # 绘制ground truth mask
        if gt_mask_bool.sum() > 0:
            color_mask_gt = np.zeros_like(image)
            color_mask_gt[:, :, 0] = 0.0
            color_mask_gt[:, :, 1] = 1.0  # 绿色通道
            color_mask_gt[:, :, 2] = 0.0
            axes[2].imshow(color_mask_gt, alpha=0.3 * gt_mask_bool.astype(float))
            
            # 可选：绘制ground truth bounding box
            if show_bbox:
                if gt_bbox is not None:
                    gt_bbox = np.array(gt_bbox).astype(int)
                    if len(gt_bbox) == 4:
                        rect_gt = Rectangle(
                            (gt_bbox[0], gt_bbox[1]),
                            gt_bbox[2] - gt_bbox[0],
                            gt_bbox[3] - gt_bbox[1],
                            linewidth=3,
                            edgecolor='green',
                            facecolor='none',
                            label='真实框'
                        )
                        axes[2].add_patch(rect_gt)
                else:
                    # 从mask中提取bbox
                    gt_mask_tensor = torch.from_numpy(gt_mask_bool.astype(float))
                    gt_bbox_from_mask = get_bbox_from_mask(gt_mask_tensor)
                    gt_bbox_from_mask = gt_bbox_from_mask.numpy().astype(int)
                    if gt_bbox_from_mask[2] > gt_bbox_from_mask[0] and gt_bbox_from_mask[3] > gt_bbox_from_mask[1]:
                        rect_gt = Rectangle(
                            (gt_bbox_from_mask[0], gt_bbox_from_mask[1]),
                            gt_bbox_from_mask[2] - gt_bbox_from_mask[0],
                            gt_bbox_from_mask[3] - gt_bbox_from_mask[1],
                            linewidth=3,
                            edgecolor='green',
                            facecolor='none',
                            label='真实框'
                        )
                        axes[2].add_patch(rect_gt)
        
        axes[2].set_title('Ground Truth overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        if gt_mask_bool.sum() > 0 and show_bbox:
            axes[2].legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化结果已保存到: {save_path}")
    plt.close()


def visualize_dataset_samples(model, dataset, num_samples=5, output_dir='visual/results', 
                              show_gt=True, show_bbox=False, device='cuda'):
    """
    从数据集中采样并可视化
    
    Args:
        model: 训练好的模型
        dataset: 数据集
        num_samples: 要可视化的样本数量
        output_dir: 输出目录
        show_gt: 是否显示ground truth
        device: 设备
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    model = model.to(device)
    
    # 随机选择样本
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            samples, targets = dataset[sample_idx]
            
            # 准备输入
            img = samples['img'].unsqueeze(0).to(device)
            word_ids = samples['word_ids'].unsqueeze(0).to(device)
            word_masks = samples['word_masks'].unsqueeze(0).to(device)
            sentence = samples['text']
            
            # 模型推理
            pred_mask = model(img, word_ids, word_masks)
            if pred_mask.ndim == 4:
                pred_mask = pred_mask.squeeze(1)
            pred_mask = pred_mask[0]  # [H, W]
            
            # 获取ground truth
            gt_mask = None
            gt_bbox = None
            if show_gt:
                gt_mask = targets['mask']
                if 'boxes' in targets and targets['boxes'] is not None:
                    gt_bbox = targets['boxes']
            
            # 保存路径
            img_path = targets.get('img_path', f'sample_{sample_idx}')
            save_name = f"vis_{idx:03d}_{os.path.basename(str(img_path))}.png"
            save_path = os.path.join(output_dir, save_name)
            
            # 可视化
            visualize_inference_with_bbox(
                img[0].cpu(),
                pred_mask.cpu(),
                sentence,
                save_path,
                gt_mask=gt_mask,
                gt_bbox=gt_bbox,
                show_gt=show_gt,
                show_bbox=show_bbox
            )


def visualize_single_image(model, image_path, sentence, output_path, device='cuda', 
                           img_size=320, show_bbox=False, ck_bert=None):
    """
    对单张图片进行推理和可视化
    
    Args:
        model: 训练好的模型
        image_path: 图片路径
        sentence: 文本描述
        output_path: 输出路径
        device: 设备
        img_size: 图像大小
        ck_bert: BERT模型路径
    """
    from transformers import BertTokenizer
    
    if ck_bert is None:
        raise ValueError("ck_bert 参数不能为 None")
    
    model.eval()
    model = model.to(device)
    
    # 加载和预处理图像
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 处理文本
    tokenizer = BertTokenizer.from_pretrained(ck_bert)
    tokens = tokenizer.encode(sentence, add_special_tokens=True, max_length=30, 
                              truncation=True, padding='max_length')
    word_ids = torch.tensor([tokens]).to(device)
    word_masks = (word_ids != tokenizer.pad_token_id).long().to(device)
    
    # 模型推理
    with torch.no_grad():
        pred_mask = model(img_tensor, word_ids, word_masks)
        if pred_mask.ndim == 4:
            pred_mask = pred_mask.squeeze(1)
        pred_mask = pred_mask[0]  # [H, W]
    
    # 可视化
    visualize_inference_with_bbox(
        img_tensor[0].cpu(),
        pred_mask.cpu(),
        sentence,
        output_path,
        show_gt=False,
        show_bbox=show_bbox
    )


def main():
    # Step 1: 先解析 config 文件路径（类似 get_args.py 的方式）
    parser = argparse.ArgumentParser(description='模型推理可视化')
    parser.add_argument('--config', type=str, default='configs/main_refersam_bert.yaml',
                       help='配置文件路径')
    
    # 可视化相关参数（这些是可视化脚本特有的，不在配置文件中）
    parser.add_argument('--mode', type=str, choices=['dataset', 'single'], 
                       default='dataset', help='可视化模式: dataset或single')
    parser.add_argument('--num_samples', type=int, default=5, 
                       help='从数据集中可视化的样本数量（仅dataset模式）')
    parser.add_argument('--image_path', type=str, default=None,
                       help='单张图片路径（仅single模式）')
    parser.add_argument('--sentence', type=str, default=None,
                       help='文本描述（仅single模式）')
    parser.add_argument('--output_img_path', type=str, default='visual/results',
                       help='输出目录')
    parser.add_argument('--show_gt', action='store_true', default=True,
                       help='是否显示ground truth（仅dataset模式，默认True）')
    parser.add_argument('--no-show_gt', dest='show_gt', action='store_false',
                       help='不显示ground truth')
    parser.add_argument('--show_bbox', action='store_true', default=False,
                       help='是否显示bounding box（默认False，语义分割任务通常不需要）')
    parser.add_argument('--dataset_name', type=str, default='refcoco',
                       choices=['refcoco', 'refcoco+', 'refcocog', 'grefcoco', 'referit'],
                       help='数据集名称（仅dataset模式）')
    parser.add_argument('--split', type=str, default='val',
                       help='数据集split（仅dataset模式）')
    parser.add_argument('--sample_idx', type=int, default=None,
                       help='指定数据集的样本索引（可选）')
    parser.add_argument('--use_model_origin', action='store_true', default=False,
                       help='使用model_origin模型（默认False，使用model模型）')
    
    # 先解析 config 参数
    args_config_only, remaining = parser.parse_known_args()
    
    # Step 2: 加载 YAML 配置文件，自动添加所有配置参数
    if os.path.exists(args_config_only.config):
        with open(args_config_only.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # 自动将配置文件中的所有键值对添加为参数
        for key, value in config_dict.items():
            arg_type = type(value) if value is not None else str
            parser.add_argument(f'--{key}', type=arg_type, default=value)
    
    # Step 3: 添加一些额外的参数（不在配置文件中，但可能需要）
    parser.add_argument('--deepspeed_config', type=str, default=None,
                       help='DeepSpeed配置文件路径')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')
    parser.add_argument('--use_enhanced_loss', action='store_true',
                       help='使用增强损失函数')
    parser.add_argument('--loss_config_path', type=str, default=None,
                       help='损失配置文件路径')
    parser.add_argument('--sentence_aggregation', type=str, 
                       choices=['best', 'mean', 'mean_iou', 'median'],
                       default='mean', help='Sentence aggregation method (not used in visualization)')
    
    # Step 4: 解析所有参数（命令行优先）
    args = parser.parse_args()
    
    # 直接使用 args，不需要创建 model_args
    model_args = args
    
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("正在加载模型...")
    if args.use_model_origin:
        # 使用 model_origin
        print("使用 model_origin 模型")
        from model_origin.builder import refersam as refersam_origin
        
        # model_origin 的 refersam 函数会自动加载 checkpoint（如果 pretrained 不为 None）
        pretrained = hasattr(model_args, 'pre_train_path') and model_args.pre_train_path is not None
        model = refersam_origin(pretrained=pretrained, args=model_args)
        
        # 确保模型在正确的设备上
        if next(model.parameters()).device != device:
            model = model.to(device)
        model.eval()
        use_fp16 = False
        use_bf16 = False
        model_engine = None
    else:
        # 使用 model (enhanced)
        print("使用 model (enhanced) 模型")
        loss_config_path = getattr(model_args, 'loss_config_path', None) if getattr(model_args, 'use_enhanced_loss', False) else None
        model, use_fp16, use_bf16, model_engine = load_model_with_checkpoint(
            model_func=refersam,
            pretrained=True,
            args=model_args,
            loss_config_path=loss_config_path,
            device=device
        )
    print("模型加载完成")
    
    if args.mode == 'dataset':
        # 数据集模式
        print(f"正在加载数据集: {args.dataset_name}")
        
        if args.dataset_name == 'refcoco':
            dataset = ReferDataset(
                refer_data_root=getattr(model_args, 'data_root', None),
                dataset='refcoco',
                splitBy='unc',
                bert_tokenizer=getattr(model_args, 'tokenizer_type', 'bert'),
                max_tokens=30,
                split=args.split,
                eval_mode=False,
                size=getattr(model_args, 'img_size', 320),
                precision=getattr(model_args, 'precision', 'fp32')
            )
        elif args.dataset_name == 'grefcoco':
            dataset = GRefDataset(
                refer_data_root=getattr(model_args, 'data_root', None),
                dataset='grefcoco',
                splitBy='unc',
                bert_tokenizer=getattr(model_args, 'tokenizer_type', 'bert'),
                max_tokens=30,
                split=args.split,
                eval_mode=False,
                size=getattr(model_args, 'img_size', 320),
                precision=getattr(model_args, 'precision', 'fp32')
            )
        elif args.dataset_name == 'referit':
            dataset = ReferitDataset(
                root=getattr(model_args, 'data_referit_root', None),
                split=args.split,
                max_tokens=30,
                size=getattr(model_args, 'img_size', 320)
            )
        else:
            raise ValueError(f"不支持的数据集: {args.dataset_name}")
        
        print(f"数据集大小: {len(dataset)}")
        
        # 如果指定了样本索引，只可视化该样本
        if args.sample_idx is not None:
            indices = [args.sample_idx]
        else:
            indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
        
        os.makedirs(args.output_img_path, exist_ok=True)
        
        model.eval()
        model = model.to(device)
        
        with torch.no_grad():
            for idx, sample_idx in enumerate(indices):
                print(f"正在处理样本 {idx+1}/{len(indices)} (索引: {sample_idx})...")
                samples, targets = dataset[sample_idx]
                
                # 准备输入
                img = samples['img'].unsqueeze(0).to(device)
                word_ids = samples['word_ids'].unsqueeze(0).to(device)
                word_masks = samples['word_masks'].unsqueeze(0).to(device)
                sentence = samples['text']
                
                # 模型推理
                pred_mask = model(img, word_ids, word_masks)
                if pred_mask.ndim == 4:
                    pred_mask = pred_mask.squeeze(1)
                pred_mask = pred_mask[0]  # [H, W]
                
                # 获取ground truth
                gt_mask = None
                gt_bbox = None
                if args.show_gt:
                    gt_mask = targets['mask']
                    if 'boxes' in targets and targets['boxes'] is not None:
                        gt_bbox = targets['boxes']
                
                # 保存路径
                img_path = targets.get('img_path', f'sample_{sample_idx}')
                save_name = f"vis_{idx:03d}_{os.path.basename(str(img_path))}.png"
                save_path = os.path.join(args.output_img_path, save_name)
                
                # 可视化
                visualize_inference_with_bbox(
                    img[0].cpu(),
                    pred_mask.cpu(),
                    sentence,
                    save_path,
                    gt_mask=gt_mask,
                    gt_bbox=gt_bbox,
                    show_gt=args.show_gt,
                    show_bbox=args.show_bbox
                )
        
        print(f"\n✅ 所有可视化结果已保存到: {args.output_img_path}")
    
    elif args.mode == 'single':
        # 单张图片模式
        if args.image_path is None or args.sentence is None:
            raise ValueError("single模式需要提供--image_path和--sentence参数")
        
        if args.output_img_path.endswith('.png'):
            output_path = args.output_img_path
        else:
            os.makedirs(args.output_img_path, exist_ok=True)
            output_path = os.path.join(args.output_img_path, 
                                      f"vis_{os.path.basename(args.image_path)}")
        
        print(f"正在处理单张图片: {args.image_path}")
        visualize_single_image(
            model,
            args.image_path,
            args.sentence,
            output_path,
            device=device,
            img_size=getattr(model_args, 'img_size', 320),
            show_bbox=args.show_bbox,
            ck_bert=getattr(model_args, 'ck_bert', None)
        )
        print(f"✅ 可视化结果已保存到: {output_path}")


if __name__ == '__main__':
    main()

