import argparse
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
import torch
from PIL import Image
from dataset.ReferDataset import ReferDataset
from get_args import get_args
from model.builder import refersam
from model.segment_anything.build_sam import sam_model_registry
import torch.nn.functional as F
"""
可以加载单张图片成为gt,主要用于单张图片的可视化
"""
def visualize_sample(samples, targets, model=None, save_path='debug_output.png'):
    img_tensor = samples['img'].to("cuda")
    # img_tensor = TF.resize(img_tensor, size=[1024, 1024])  # (C, H, W)

    mask = targets['mask'].to("cuda")
    word_id = samples['word_ids'].to("cuda")
    word_mask = samples['word_masks'].to("cuda")
    sentence = samples['text']
    img_path = targets['img_full_path']

    # 模型推理
    model.eval()
    with torch.no_grad():
        pred_mask = model(img_tensor.unsqueeze(0), word_id.unsqueeze(0), word_mask.unsqueeze(0))
        # resized_mask_tensor = F.interpolate(pred_mask.unsqueeze(1).float(), size=(480, 480), mode='bilinear', align_corners=False)
        pred_mask = pred_mask.squeeze(0).cpu().numpy()  # [H, W]
    # 处理图像
    if isinstance(img_tensor, torch.Tensor):
        img = TF.to_pil_image(img_tensor.cpu())
    else:
        img = img_tensor

    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = np.array(mask)

    if mask_np.ndim == 3 and mask_np.shape[0] == 1:
        mask_np = mask_np.squeeze(0)

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 5, 1)
    plt.imshow(img)
    plt.title(f'Image\n{img_path}')
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.imshow(img)
    plt.imshow(mask_np, cmap='jet', alpha=0.5)
    plt.title(f'GT Overlay\n"{sentence}"')
    plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.imshow(img)
    plt.imshow(pred_mask, cmap='jet', alpha=0.5)
    plt.title(f'Pred Overlay\n"{sentence}"')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved visualization to {save_path}")
    plt.close()

if __name__ == '__main__':
    args = get_args()
    dataset = ReferDataset(
        refer_data_root=args.data_root,
        dataset='refcoco',
        splitBy='unc',
        bert_tokenizer=args.tokenizer_type,
        max_tokens=30, 
        split='train',
        eval_mode=False,
        size=320,
        precision=args.precision,
        # image_transforms=None
    )

    samples, targets = dataset.__getitem__(1)


    # print("Creating ReferSAM model...")
    # # Initialize models and criterion
    # print("Initializing models...")
    # sam = sam_model_registry[args.sam_type](checkpoint=args.checkpoint)
    # text_model = BertModel.from_pretrained(args.ck_bert)
    # criterion = SegMaskLoss(num_points=112*112, oversample_ratio=3.0, importance_sample_ratio=0.75)

    # # Create model
    # print("Creating ReferSAM model...")
    # model = ReferSAM(
    #     sam_model=sam,
    #     text_encoder=text_model,
    #     args=args,
    #     num_classes=1,
    #     criterion=criterion
    # )
    from model.enhanced_builder import load_model_with_checkpoint
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 使用统一的模型加载函数
    loss_config_path = getattr(args, 'loss_config_path', None) if hasattr(args, 'use_enhanced_loss') and args.use_enhanced_loss else None
    model, use_fp16, use_bf16, model_engine = load_model_with_checkpoint(
        model_func=refersam,
        pretrained=True,
        args=args,
        loss_config_path=loss_config_path,
        device=device
    )
    # model 已经是 eval 模式

    visualize_sample(samples, targets, model=model,save_path='sample_debug.png')
