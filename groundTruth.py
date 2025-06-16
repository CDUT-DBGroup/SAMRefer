import argparse
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
import torch
from PIL import Image
from dataset.ReferDataset import ReferDataset
def visualize_sample(samples, targets, save_path='debug_output.png'):
    img_tensor = samples['img']
    mask = targets['mask']
    sentence = samples['text']
    img_path = targets['img_full_path']

    
    # 从 Tensor 转为 PIL Image（确保图像维度是 [C, H, W]）
    if isinstance(img_tensor, torch.Tensor):
        img = TF.to_pil_image(img_tensor.cpu())
    else:
        img = img_tensor
    # img = Image.open(img_path).convert('RGB')

    # 如果 mask 是 torch.Tensor，则转为 numpy array
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = np.array(mask)

    if mask_np.ndim == 3 and mask_np.shape[0] == 1:
        mask_np = mask_np.squeeze(0)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f'Image\n{img_path}')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(mask_np, cmap='jet', alpha=0.5)
    plt.title(f'Mask Overlay\n"{sentence}"')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved visualization to {save_path}")
    plt.close()

if __name__ == '__main__':
    args = argparse.Namespace(
        data_root='/public/home/2023020919/vision_paper/paper_data/coco_data',
        output_dir='output/refersam_bert',
        model_type='vit_b',
        checkpoint='/public/home/2023020919/vision_paper/weight/sam/sam_vit_b_01ec64.pth',
        tokenizer_type='bert',
        precision='fp32',
        clip_path=None, 
        ck_bert='/public/home/2023020919/vision_paper/samrefer/bert-base-uncased'
    )
    dataset = ReferDataset(
        refer_data_root=args.data_root,
        dataset='refcoco',
        splitBy='unc',
        bert_tokenizer=args.tokenizer_type,
        max_tokens=30, 
        split='train',
        eval_mode=False,
        size=480,
        precision=args.precision,
        image_transforms=None
    )

    samples, targets = dataset.__getitem__(1)
    visualize_sample(samples, targets, save_path='sample_debug.png')
