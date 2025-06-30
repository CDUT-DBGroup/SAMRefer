import random
import sys
sys.path.append('')
import pickle
import torch
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import os, pickle, cv2
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
import scipy.io as sio
from transformers import BertTokenizer, BertModel
from transformers import CLIPTextModel, CLIPTokenizer
from pycocotools import mask as cocomask
import torchvision.transforms as transforms
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def load_referit_gt_mask(mask_path):
    mat = sio.loadmat(mask_path)
    mask = (mat['segimg_t'] == 0)
    return mask

def save_tmp_mask(input_path, save_name):
    m1 = load_referit_gt_mask(input_path)
    cv2.imwrite(save_name, m1*255)


class ReferitDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", loader=pil_loader, max_tokens=20, negative_samples=0, pseudo_path=None, test_split=None, size=320, clip=False):
        annt_path = os.path.join(root, 'annotations', split + '.pickle')
        print(annt_path, '----', split)
        with open(annt_path, 'rb') as f:
            self.annotations = pickle.load(f, encoding='latin1')
        self.files = list(self.annotations.keys())
        print('num of data:{}'.format(len(self.files)))
        self.image_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),  # 避免插值产生非整数标签
            transforms.PILToTensor(),       # 保持 int 类型，不归一化
            transforms.Lambda(lambda x: x.squeeze().long())  # 若是单通道，去掉通道维度并转为 LongTensor
        ])
        # Initialize tokenizers
        self.clip = clip
        if self.clip:
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased/')
        self.loader = loader
        self.split = split
        self.img_folder = os.path.join(root, 'images')
        self.size = size
        self.max_tokens = max_tokens 
        self.negative_samples = negative_samples

        self.pseudo_path = pseudo_path
        self.test_split = test_split
        self.all_refs = {} 
        index_g = 0
        for index in range(len((self.files))):
            item = str(self.files[index])
            ann = self.annotations[item]['annotations']
            for ref in ann:
                self.all_refs[index_g] = ref 
                index_g += 1 

    def __getitem__(self, index):
        item = self.all_refs[index]
        img_path = os.path.join(self.img_folder, str(item['image_id']) + '.jpg')
        img = Image.open(img_path).convert("RGB")
        orig_size = np.array([img.height, img.width])
        img_full_path = os.path.abspath(img_path)
        img = self.image_transform(img)

        query = item['query']
        if self.clip:
            tokens = self.tokenizer(
                query,
                max_length=self.max_tokens,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            word_id = tokens['input_ids'].squeeze(0)
            word_mask = tokens['attention_mask'].squeeze(0)
        else:
            tokens = self.tokenizer(
                query,
                max_length=self.max_tokens,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            word_id = tokens['input_ids'].squeeze(0)
            word_mask = tokens['attention_mask'].squeeze(0)

        bbox = item["bbox"]
        bbox = np.array(bbox)
        mask = cocomask.decode(item["segmentation"])
        mask = np.sum(mask, axis=2)
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask)
        mask = self.mask_transform(mask)
        mask = torch.tensor(np.asarray(mask), dtype=torch.float32)

        samples = {
            "img": img,
            "orig_size": orig_size,
            "text": query,
            "word_ids": word_id,
            "word_masks": word_mask,
        }
        targets = {
            "mask": mask,
            "img_path": item["image_id"],
            "sentences": query,
            "boxes": bbox,
            "orig_size": orig_size,
            "img_full_path": img_full_path,
        }
        return samples, targets

    def __len__(self):
        return len(self.all_refs)



if __name__ == "__main__":
    import cv2
    # args = vars(parser.parse_args())
    d_train  = ReferitDataset(root="/root/autodl-tmp/paper_data/referit", split="train", max_tokens=30)
    d_test = ReferitDataset(root="/root/autodl-tmp/paper_data/referit", split="test", max_tokens=30)
    # d_test.__getitem__(1)
    d_train.__getitem__(1)
    ds = torch.utils.data.DataLoader(d_test,
                                     batch_size=1,
                                     num_workers=0,
                                     shuffle=False,
                                     drop_last=False)
    
    for idx,(img, samples, image_sizes, img_path) in enumerate(ds):
        img_id = img_path[0].split('/')[-1].split('.')[0]
        img = img.cuda()
        j = 0 
        for sen in samples.keys():
            item = samples[sen]
            sentences, bbox = item['sentences'], item['bbox']
            bbox = bbox[0]
            word_id = item['word_id'].cuda() 
            target = item['mask'].cuda() 
            o_H,o_W = target.shape[-2:]
            batch_size = word_id.shape[0]

            output = model(img, word_id)
            pred = F.interpolate(output, (o_H,o_W), align_corners=True, mode='bilinear').squeeze(0)

            # pdb.set_trace() 
            pred /= F.adaptive_max_pool2d(pred, (1, 1)) + 1e-5
            pred = pred.squeeze(0)
            t_cam = pred.clone()
            pred = pred.gt(1e-9)
            target = target.squeeze(0).squeeze(0)     
            

