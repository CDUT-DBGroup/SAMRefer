import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image
import pdb
import copy
from random import choice

from transformers import BertTokenizer, CLIPTokenizer
from torchvision import transforms
from get_args import get_args
from dataset.refer_refzom import REFER
import copy
import random
import torch
from collections import defaultdict

import torch
import torch.distributed as dist


def squeeze_and_long(x):
    """辅助函数：将tensor压缩并转换为long类型，用于mask transform
    这个函数必须在模块级别定义，以便在多进程DataLoader中可以被pickle序列化
    """
    return x.squeeze().long()
from torch.utils.data.distributed import DistributedSampler

# from args import get_parser
# import random
# # Dataset configuration initialization
# parser = get_parser()
# args = parser.parse_args()


# class Referzom_Dataset(data.Dataset):

#     def __init__(self,
#                  args,
#                  image_transforms=None,
#                  target_transforms=None,
#                  split='train',
#                  eval_mode=False):

#         self.classes = []
#         self.image_transforms = image_transforms
#         self.target_transform = target_transforms
#         self.split = split
#         self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)
#         self.dataset_type = args.dataset
#         self.max_tokens = 20
#         ref_ids = self.refer.getRefIds(split=self.split)
#         self.img_ids = self.refer.getImgIds()

#         all_imgs = self.refer.Imgs
#         self.imgs = list(all_imgs[i] for i in self.img_ids)
#         self.ref_ids = ref_ids

#         self.input_ids = []
#         self.input_ids_masked = []
#         self.attention_masks = []
#         self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

#         self.eval_mode = eval_mode

#         self.zero_sent_id_list = []
#         self.one_sent_id_list = []
#         self.all_sent_id_list = []
#         self.sent_2_refid = {}
#         for r in ref_ids:
#             ref = self.refer.loadRefs(r)

#             source_type = ref[0]['source']

#             for sent_dict in ref[0]['sentences']:
#                 sent_id = sent_dict['sent_id']

#                 self.sent_2_refid[sent_id] = r
#                 self.all_sent_id_list.append(sent_id)
#                 if source_type=='zero':
#                     self.zero_sent_id_list.append(sent_id)
#                 else:
#                     self.one_sent_id_list.append(sent_id)

#         for r in ref_ids:
#             ref = self.refer.Refs[r]

#             sentences_for_ref = []
#             sentences_for_ref_masked = []
#             attentions_for_ref = []

#             for i, el in enumerate(ref['sentences']):
#                 sentence_raw = el['raw']
#                 attention_mask = [0] * self.max_tokens
#                 padded_input_ids = [0] * self.max_tokens
#                 padded_input_ids_masked = [0] * self.max_tokens

#                 blob = TextBlob(sentence_raw.lower())
#                 chara_list = blob.tags
#                 mask_ops = []
#                 mask_ops1 = []
#                 for word_i, (word_now, chara) in enumerate(chara_list):
#                     if (chara == 'NN' or chara == 'NNS') and word_i < 19 and word_now.lower():
#                         mask_ops.append(word_i)
#                         mask_ops1.append(word_now)
#                 mask_ops2 = self.get_adjacent_word(mask_ops)


#                 input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

#                 # truncation of tokens
#                 input_ids = input_ids[:self.max_tokens]

#                 padded_input_ids[:len(input_ids)] = input_ids
#                 attention_mask[:len(input_ids)] = [1]*len(input_ids)
#                 if len(mask_ops) == 0:
#                     attention_remask = attention_mask
#                     input_ids_masked = input_ids
#                 else:
#                     could_mask = choice(mask_ops2)
#                     input_ids_masked = copy.deepcopy(input_ids)
#                     for i in could_mask:
#                         input_ids_masked[i + 1] = 0
#                 padded_input_ids_masked[:len(input_ids_masked)] = input_ids_masked

#                 sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
#                 sentences_for_ref_masked.append(torch.tensor(padded_input_ids_masked).unsqueeze(0))
#                 attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

#             self.input_ids.extend(sentences_for_ref)
#             self.input_ids_masked.extend(sentences_for_ref_masked)
#             self.attention_masks.extend(attentions_for_ref)


#     def get_classes(self):
#         return self.classes

#     def __len__(self):
#         return len(self.all_sent_id_list)
    
#     def get_adjacent_word(self, mask_list):
#         output_mask_list = []
#         length = len(mask_list)
#         i = 0
#         while i < length:
#             begin_pos = i
#             while i+1 < length and mask_list[i+1] == mask_list[i] + 1:
#                 i += 1
#             end_pos = i+1
#             output_mask_list.append(mask_list[begin_pos:end_pos])
#             i = end_pos

#         return output_mask_list

#     def __getitem__(self, index):
        
#         sent_id = self.all_sent_id_list[index]
#         this_ref_id = self.sent_2_refid[sent_id]

#         this_img_id = self.refer.getImgIds(this_ref_id)
#         this_img = self.refer.Imgs[this_img_id[0]]

#         img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")

#         ref = self.refer.loadRefs(this_ref_id)
#         if self.dataset_type == 'ref-zom':
#             source_type = ref[0]['source']
#         else:
#             source_type = 'not_zero'

#         ref_mask = np.array(self.refer.getMask(ref[0])['mask'])

#         annot = np.zeros(ref_mask.shape)
#         annot[ref_mask == 1] = 1
#         annot = Image.fromarray(annot.astype(np.uint8), mode="P")


#         if self.image_transforms is not None:

#             if self.split == 'train':
#                 img, target = self.image_transforms(img, annot)
#             elif self.split == 'val':
#                 img, target = self.image_transforms(img, annot)
#             else:
#                 img, target = self.image_transforms(img, annot)

#         if self.eval_mode:
#             embedding = []
#             embedding_masked = []
#             att = []
#             for s in range(len(self.input_ids[index])):
#                 e = self.input_ids[index][s]
#                 # e1 = self.input_ids_masked[index][s]
#                 a = self.attention_masks[index][s]
#                 embedding.append(e.unsqueeze(-1))
#                 embedding_masked.append(e.unsqueeze(-1))
#                 att.append(a.unsqueeze(-1))
            
#             tensor_embeddings = torch.cat(embedding, dim=-1)
#             tensor_embeddings_masked = torch.cat(embedding_masked, dim=-1)
#             attention_mask = torch.cat(att, dim=-1)
#         else:
#             choice_sent = np.random.choice(len(self.input_ids[index]))
#             tensor_embeddings = self.input_ids[index][choice_sent]
#             tensor_embeddings_masked = self.input_ids_masked[index][choice_sent]
#             attention_mask = self.attention_masks[index][choice_sent]

#         return img, target, source_type, tensor_embeddings, tensor_embeddings_masked, attention_mask




# class Refzom_DistributedSampler(DistributedSampler):
#     def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
#         super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
#         self.one_id_list = dataset.one_sent_id_list

#         self.zero_id_list = dataset.zero_sent_id_list
#         self.sent_ids_list = dataset.all_sent_id_list
#         if self.shuffle==True:
#             random.shuffle(self.one_id_list)
#             random.shuffle(self.zero_id_list)

#         self.sent_id = self.insert_evenly(self.zero_id_list,self.one_id_list)
#         self.indices = self.get_positions(self.sent_ids_list, self.sent_id)
        
#     def get_positions(self, list_a, list_b):
#         position_dict = {value: index for index, value in enumerate(list_a)}
#         positions = [position_dict[item] for item in list_b]

#         return positions
    
#     def insert_evenly(self, list_a, list_b):
#         len_a = len(list_a)
#         len_b = len(list_b)
#         block_size = len_b // len_a

#         result = []
#         for i in range(len_a):
#             start = i * block_size
#             end = (i + 1) * block_size
#             result.extend(list_b[start:end])
#             result.append(list_a[i])

#         remaining = list_b[(len_a * block_size):]
#         result.extend(remaining)

#         return result
    
#     def __iter__(self):
        
#         indices_per_process = self.indices[self.rank::self.num_replicas]
#         return iter(indices_per_process)

class ReferzomDataset(data.Dataset):
    def __init__(self,
                 refer_data_root='data',
                 dataset='ref-zom',
                 splitBy='final',
                 bert_tokenizer='bert-base-uncased/',
                 max_tokens=30,
                 split='train',
                 eval_mode=False,
                 size=480,
                 precision='fp32',
                 return_all_sentences=False):
        self.clip = 'clip' in bert_tokenizer
        self.split = split
        self.dataset_type = dataset
        self.eval_mode = eval_mode
        self.return_all_sentences = return_all_sentences  # 是否返回所有描述用于选择最优
        self.max_tokens = max_tokens
        self.size = size

        # Precision setting
        if precision == "bf16":
            self.torch_dtype = torch.bfloat16
        elif precision == "fp16":
            self.torch_dtype = torch.half
        else:
            self.torch_dtype = torch.float32

        # Init tokenizer
        if self.clip:
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased/')

        # Load REFER
        self.refer = REFER(refer_data_root, dataset, splitBy)
        self.ref_ids = self.refer.getRefIds(split=self.split)

        self.input_ids = []
        self.word_masks = []
        self.all_sentences = []
        self.img_ids = []
        self.bboxes = []

        for ref_id in self.ref_ids:
            ref = self.refer.Refs[ref_id]
            img_id = ref['image_id']
            self.img_ids.append(img_id)

            ref_sent_ids = []
            ref_masks = []
            raw_sentences = []

            for sent in ref['sentences']:
                sentence = sent['raw']
                tokens = self.tokenizer(
                    sentence,
                    max_length=self.max_tokens,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                ref_sent_ids.append(tokens['input_ids'].squeeze(0))
                ref_masks.append(tokens['attention_mask'].squeeze(0))
                raw_sentences.append(sentence)

            self.input_ids.append(ref_sent_ids)
            self.word_masks.append(ref_masks)
            self.all_sentences.append(raw_sentences)

        # Define transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor(),
            transforms.Lambda(squeeze_and_long)
        ])

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        ref_id = self.ref_ids[index]
        img_id = self.img_ids[index]
        img_meta = self.refer.Imgs[img_id]
        if img_meta['file_name'].startswith('COCO_train2014'):
            img_path = os.path.join(self.refer.IMAGE_DIR, img_meta['file_name'])
        else:
            img_path = os.path.join(self.refer.IMAGE_DIR.replace('train2014', 'val2014'), img_meta['file_name'])
        image = Image.open(img_path).convert("RGB")

        ref = self.refer.loadRefs(ref_id)[0]
        mask = self.refer.getMask(ref)['mask']  # dict with 'mask': numpy array
        # bbox = self.refer.getRefBox(ref_id)
        # bbox = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])

        h, w = mask.shape
        annot = Image.fromarray(mask.astype(np.uint8), mode="P")

        # Transform image and mask
        image = self.image_transform(image)
        annot = self.mask_transform(annot).float()

        # Choose a sentence
        if self.return_all_sentences and self.eval_mode:
            # 返回所有描述，用于在验证时选择最优描述
            num_sentences = len(self.input_ids[index])
            all_word_ids = [self.input_ids[index][i].clone().to(dtype=torch.long) for i in range(num_sentences)]
            all_word_masks = [self.word_masks[index][i].clone().to(dtype=torch.long) for i in range(num_sentences)]
            all_sentences = self.all_sentences[index]
            
            # 使用第一个描述作为默认（用于兼容性）
            word_ids = all_word_ids[0]
            word_masks = all_word_masks[0]
            sentence = all_sentences[0]
        elif self.eval_mode:
            sent_idx = 0
            word_ids = self.input_ids[index][sent_idx].to(dtype=torch.long)
            word_masks = self.word_masks[index][sent_idx].to(dtype=torch.long)
            sentence = self.all_sentences[index][sent_idx]
            all_word_ids = []
            all_word_masks = []
            all_sentences = []
        else:
            sent_idx = random.randint(0, len(self.input_ids[index]) - 1)
            word_ids = self.input_ids[index][sent_idx].to(dtype=torch.long)
            word_masks = self.word_masks[index][sent_idx].to(dtype=torch.long)
            sentence = self.all_sentences[index][sent_idx]
            all_word_ids = []
            all_word_masks = []
            all_sentences = []

        samples = {
            "img": image,
            "orig_size": np.array([h, w]),
            "text": sentence,
            "word_ids": word_ids,
            "word_masks": word_masks,
        }
        
        # 如果返回所有描述，添加到samples中
        if self.return_all_sentences and self.eval_mode and len(all_word_ids) > 0:
            samples["all_word_ids"] = all_word_ids
            samples["all_word_masks"] = all_word_masks
            samples["all_sentences"] = all_sentences

        targets = {
            "mask": annot,
            "img_path": str(img_meta['file_name']),
            "sentences": sentence,
            # "boxes": None,
            "orig_size": np.array([h, w]),
            "img_full_path": img_path,
        }

        return samples, targets
    
if __name__=="__main__":
    args = get_args()
    dataset = ReferzomDataset(
        refer_data_root=args.data_root,
        dataset='ref-zom',
        splitBy='final',
        bert_tokenizer=args.tokenizer_type,
        max_tokens=30,
        split='train',
    )
    print(dataset.__getitem__(0))
