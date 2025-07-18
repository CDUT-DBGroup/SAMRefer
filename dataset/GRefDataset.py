import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image
import random
from transformers import BertTokenizer, CLIPTokenizer
from torchvision import transforms
from get_args import get_args
from dataset.gref import G_REFER

class GRefDataset(data.Dataset):
    def __init__(self,
                 refer_data_root='data',
                 dataset='grefcoco',
                 splitBy='unc',
                 bert_tokenizer='bert-base-uncased',
                 max_tokens=30,
                 split='train',
                 eval_mode=False,
                 size=480,
                 precision='fp32'):
        self.clip = 'clip' in bert_tokenizer
        self.split = split
        self.dataset_type = dataset
        self.eval_mode = eval_mode
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
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Load G_REFER
        self.refer = G_REFER(refer_data_root, dataset, splitBy)
        self.ref_ids = self.refer.getRefIds(split=self.split)

        self.input_ids = []
        self.word_masks = []
        self.all_sentences = []
        self.img_ids = []

        for ref_id in self.ref_ids:
            ref = self.refer.Refs[ref_id]
            img_id = ref['image_id']
            self.img_ids.append(img_id)

            ref_sent_ids = []
            ref_masks = []
            raw_sentences = []

            for sent in ref['sentences']:
                # G_REFER使用'sent'字段
                sentence = sent['sent']
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
            transforms.Lambda(lambda x: x.squeeze().long())
        ])

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        # 跳过空input_ids样本，最多尝试10次
        max_attempts = 10
        attempt = 0
        current_index = index
        
        while len(self.input_ids[current_index]) == 0 and attempt < max_attempts:
            current_index = random.randint(0, len(self.input_ids) - 1)
            attempt += 1
            
        # 如果所有尝试都失败，创建一个默认的tensor
        if len(self.input_ids[current_index]) == 0:
            # 创建一个默认的tensor
            default_tensor = torch.zeros(self.max_tokens, dtype=torch.long)
            default_mask = torch.zeros(self.max_tokens, dtype=torch.long)
            default_sentence = ""

        ref_id = self.ref_ids[current_index]
        img_id = self.img_ids[current_index]
        img_meta = self.refer.Imgs[img_id]

        img_path = os.path.join(self.refer.IMAGE_DIR, img_meta['file_name'])
        image = Image.open(img_path).convert("RGB")

        ref = self.refer.loadRefs(ref_id)[0]
        # G_REFER的getMaskByRef方法返回多个mask，需要合并
        mask_data = self.refer.getMaskByRef(ref, merge=True)
        
        # 检查mask_data是否为空
        if mask_data.get('empty', False):
            # 如果为空，创建一个全零mask
            img_meta = self.refer.Imgs[img_id]
            mask = np.zeros((img_meta['height'], img_meta['width']), dtype=np.uint8)
            h, w = mask.shape
        else:
            mask = mask_data['mask']  # numpy array
            h, w = mask.shape
            
        annot = Image.fromarray(mask.astype(np.uint8), mode="P")

        # Transform image and mask
        image = self.image_transform(image)
        annot = self.mask_transform(annot).float()

        # Choose a sentence
        if len(self.input_ids[current_index]) == 0:
            # 使用默认值
            word_ids = default_tensor
            word_masks = default_mask
            sentence = default_sentence
        else:
            if self.eval_mode:
                sent_idx = 0
            else:
                sent_idx = random.randint(0, len(self.input_ids[current_index]) - 1)

            word_ids = self.input_ids[current_index][sent_idx].to(dtype=torch.long)
            word_masks = self.word_masks[current_index][sent_idx].to(dtype=torch.long)
            sentence = self.all_sentences[current_index][sent_idx]

        # 确保word_ids和word_masks是torch.long类型
        if isinstance(word_ids, torch.Tensor):
            word_ids = word_ids.to(dtype=torch.long)
        else:
            word_ids = torch.tensor(word_ids, dtype=torch.long)
            
        if isinstance(word_masks, torch.Tensor):
            word_masks = word_masks.to(dtype=torch.long)
        else:
            word_masks = torch.tensor(word_masks, dtype=torch.long)
            
        # 确保orig_size是numpy数组
        orig_size = np.array([h, w], dtype=np.int64)
        
        samples = {
            "img": image,
            "orig_size": orig_size,
            "text": sentence,
            "word_ids": word_ids,
            "word_masks": word_masks,
        }

        targets = {
            "mask": annot,
            "img_path": str(img_meta['file_name']),
            "sentences": sentence,
            "orig_size": orig_size,
            "img_full_path": img_path,
        }

        return samples, targets

if __name__=="__main__":
    args = get_args()
    dataset = GRefDataset(
        refer_data_root=args.data_root,
        dataset='grefcoco',
        splitBy='unc',
        bert_tokenizer=args.tokenizer_type,
        max_tokens=30,
        split='train',
    )
    print(dataset.__getitem__(0)) 