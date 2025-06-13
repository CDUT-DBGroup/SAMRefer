import random
import sys
sys.path.append('')
from args import get_parser
import pickle
import torch
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import os, pickle, cv2
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

import CLIP.clip as clip 
import scipy.io as sio

from pycocotools import mask as cocomask

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

import torchvision.transforms as transforms
def get_flicker_transform(args):
    Isize = args.size 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    resize = (Isize, Isize)
    tflist = [transforms.Resize(resize)]

    transform_train = transforms.Compose(tflist + [
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])

    transform_test = transforms.Compose([
                             transforms.Resize(resize),
                             transforms.ToTensor(),
                             normalize
                             ])

    return transform_train, transform_test


class ImageLoader_train(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, split="train", loader=pil_loader, max_tokens=20, negative_samples=0, pseudo_path=None, test_split=None, size=320):
        annt_path = os.path.join(root, 'annotations', split + '.pickle')
        print(annt_path, '----', split)
        with open(annt_path, 'rb') as f:
            self.annotations = pickle.load(f, encoding='latin1')
        self.files = list(self.annotations.keys())
        print('num of data:{}'.format(len(self.files)))
        self.transform = transform
        self.loader = loader
        self.split = split
        self.img_folder = os.path.join(root, 'images')
        self.size = size
        self.max_tokens = max_tokens 
        self.negative_samples = negative_samples

        self.pseudo_path = pseudo_path
        self.test_split = test_split
        self.all_refs = {} 
        # self.imgid2_refs = {} 
        # self.refid2index = {} 
        index_g = 0
        for index in range(len((self.files))):
            item = str(self.files[index])
            ann = self.annotations[item]['annotations']

            # self.imgid2_refs[ann[0]['image_id']] = ann 

            for ref in ann:
                self.all_refs[index_g] = ref 
                # self.refid2index[ref['ref_id']] = index_g

                index_g += 1 
        # print(index_g, '===')

    def __getitem__(self, index):
        item = self.all_refs[index]
        image_id = item["image_id"]
        pseudo_gt = None
        if self.pseudo_path is not None:
                # pseudo_filename = f'{index}_{choice_sent}_{this_img_id[0]}.npy'
            pseudo_filename = f'{index}_{image_id}.npy'
            pseudo_info = np.load(os.path.join(self.pseudo_path, pseudo_filename), allow_pickle=True).item()
            pseudo_gt = pseudo_info['mask'] * 1.0
            pseudo_gt = pseudo_gt.sum(0)
            pseudo_gt = F.resize(Image.fromarray(pseudo_gt), (self.size, self.size),
                                    interpolation=InterpolationMode.NEAREST)
            pseudo_gt = torch.tensor(np.asarray(pseudo_gt), dtype=torch.int64).unsqueeze(0)
                # pseudo_gt[pseudo_gt>0] = 1   
        else:
            pseudo_gt = None


        if self.negative_samples>0:
            def get_random_indices_except_current(total_length, current_index, num_samples=5):
                # 生成一个从0到total_length-1的索引列表
                all_indices = list(range(total_length))
                
                # 从所有索引中移除当前索引
                available_indices = [index for index in all_indices if index != current_index]
                
                # 确保有足够多的元素来选择
                if len(available_indices) < num_samples:
                    raise ValueError("The list is too short to pick the required number of unique samples after excluding the current index.")
                
                # 从剩余的索引中随机抽取num_samples个索引
                return random.sample(available_indices, num_samples)
            random_indices = get_random_indices_except_current(len(self.all_refs), index, num_samples=self.negative_samples)
            neg_sents = []  # 文本描述词
            neg_word_ids = []  # 单词转为数字之后的映射
            for neg_index in random_indices:
                neg_word_ids.append(torch.tensor(clip.tokenize(self.all_refs[neg_index]['query']).squeeze(0)[:self.max_tokens]).unsqueeze(0))
                neg_sents.append(self.all_refs[neg_index]['query'])
            neg_word_ids = torch.cat(neg_word_ids, dim=0)
        img_path = os.path.join(self.img_folder, str(item['image_id']) + '.jpg')
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)   

        query = item['query']
        word_id = clip.tokenize(query).squeeze(0)[:self.max_tokens]
        word_id = np.array(word_id)
        samples = {  # 原始图片
            "img": img,
            "word_ids": word_id,
        }
        if self.negative_samples > 0:  # 如果是负样本的话需要加上负样本
            samples['neg_sents'] = neg_sents
            samples['neg_word_ids'] = neg_word_ids

        
        out = {}
        bbox=torch.tensor(0)
        mask=torch.tensor(0)
        # bbox = item["bbox"]
        # bbox = np.array(bbox)
        # # image_sizes = img.size
        # mask = cocomask.decode(item["segmentation"])
        # mask = np.sum(mask, axis=2)
        # mask = mask.astype(np.uint8)
        if self.test_split is not None:
            bbox = item["bbox"]
            bbox = np.array(bbox)
            # image_sizes = img.size
            mask = cocomask.decode(item["segmentation"])
            mask = np.sum(mask, axis=2)
            mask = mask.astype(np.uint8)
            # 加的都是当前的内容
            ann = self.annotations[str(image_id)]['annotations']
            for i in range(0, len(ann)):
                tmp = {}
                bbox = ann[i]['bbox']
                if (bbox[0][3]-bbox[0][1]) * (bbox[0][2]-bbox[0][0]) > 0.05 * image_sizes[0] * image_sizes[1]:
                    tmp['sentences'] = ann[i]['query']
                    tmp['word_id'] = clip.tokenize(ann[i]['query']).squeeze(0)[:self.max_tokens]
                    tmp['bbox'] = np.array(bbox)
                    ####### get target mask 
                    rle = ann[i]['segmentation']
                    mask = cocomask.decode(rle)
                    mask = np.sum(mask, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
                    mask = mask.astype(np.uint8) # convert to np.uint8
                    # img, mask = self.image_transforms(img, mask) 
                    #######
                    tmp['mask'] = mask 
                    out[str(i)] = tmp 

        targets = { # target为分割后的图片
            "target":mask,
            "img_path": img_path,
            "image_id": image_id,
            "sentences": query,
            "boxes": bbox,
            # "other":out  # 同一种类的其他内容
            # "orig_size": np.array([h, w]),
            # "img_path_full": img_path_full
        }
        if pseudo_gt is not None:
            targets['pseudo_gt'] = pseudo_gt

        return samples, targets

    def __len__(self):
        return len(self.all_refs) * 1


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, split="train", loader=pil_loader, max_tokens=20):
        annt_path = os.path.join(root, 'annotations', split + '.pickle')
        print(annt_path, '----')
        with open(annt_path, 'rb') as f:
            self.annotations = pickle.load(f, encoding='latin1')
        self.files = list(self.annotations.keys())
        print('num of data:{}'.format(len(self.files)))
        self.transform = transform
        self.loader = loader
        self.split = split
        self.img_folder = os.path.join(root, 'images')

        self.max_tokens = max_tokens 

    def __getitem__(self, index):
        item = str(self.files[index])
        img_path = os.path.join(self.img_folder, item + '.jpg')
        img = Image.open(img_path).convert("RGB")
        image_sizes = (img.height, img.width)

        img = self.transform(img)

        ann = self.annotations[item]['annotations']
        
        out = {}
        for i in range(0, len(ann)):
            tmp = {}
            bbox = ann[i]['bbox']

            if (bbox[0][3]-bbox[0][1]) * (bbox[0][2]-bbox[0][0]) > 0.05 * image_sizes[0] * image_sizes[1]:
                tmp['sentences'] = ann[i]['query']
                tmp['word_id'] = clip.tokenize(ann[i]['query']).squeeze(0)[:self.max_tokens]
                tmp['bbox'] = np.array(bbox)
                ####### get target mask 
                rle = ann[i]['segmentation']
                mask = cocomask.decode(rle)
                mask = np.sum(mask, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
                mask = mask.astype(np.uint8) # convert to np.uint8
                # img, mask = self.image_transforms(img, mask) 
                #######
                tmp['mask'] = mask 
                out[str(i)] = tmp 
        return img, out, image_sizes, img_path

    def __len__(self):
        return len(self.files) 




from dataset.transform import get_transform
def get_refit_dataset(args, train='train',test='test'):
    datadir = args.refer_data_root
    transform_train, transform_test = get_flicker_transform(args)
    ds_train = ImageLoader_train(datadir, split=train, transform=transform_train, max_tokens=args.max_query_len, negative_samples=args.negative_samples, pseudo_path=args.pseudo_path)  
    ds_test = ImageLoader(datadir, split=test, transform=transform_test, max_tokens=args.max_query_len)
    return ds_train, ds_test

if __name__ == "__main__":
    import argparse
    import cv2
    parser = get_parser()
    # parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-Isize', '--Isize', default=224, help='image size', required=False)
    args = parser.parse_args()
    args.refer_data_root = '/public/home/2023020919/vision_paper/paper_data/referit'
    args.negative_samples = 3
    # args = vars(parser.parse_args())
    d_train,d_test = get_refit_dataset(args=args)
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
            

