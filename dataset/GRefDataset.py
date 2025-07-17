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
        # 跳过空input_ids样本
        if len(self.input_ids[index]) == 0:
            # 随机采样一个新index，递归调用
            new_index = random.randint(0, len(self.input_ids) - 1)
            return self.__getitem__(new_index)

        ref_id = self.ref_ids[index]
        img_id = self.img_ids[index]
        img_meta = self.refer.Imgs[img_id]

        img_path = os.path.join(self.refer.IMAGE_DIR, img_meta['file_name'])
        image = Image.open(img_path).convert("RGB")

        ref = self.refer.loadRefs(ref_id)[0]
        # G_REFER的getMaskByRef方法返回多个mask，需要合并
        mask_data = self.refer.getMaskByRef(ref, merge=True)
        mask = mask_data['mask']  # numpy array
        h, w = mask.shape
        annot = Image.fromarray(mask.astype(np.uint8), mode="P")

        # Transform image and mask
        image = self.image_transform(image)
        annot = self.mask_transform(annot).float()

        # Choose a sentence
        if self.eval_mode:
            sent_idx = 0
        else:
            sent_idx = random.randint(0, len(self.input_ids[index]) - 1)

        word_ids = self.input_ids[index][sent_idx].to(dtype=torch.long)
        word_masks = self.word_masks[index][sent_idx].to(dtype=torch.long)
        sentence = self.all_sentences[index][sent_idx]

        samples = {
            "img": image,
            "orig_size": np.array([h, w]),
            "text": sentence,
            "word_ids": word_ids,
            "word_masks": word_masks,
        }

        targets = {
            "mask": annot,
            "img_path": img_meta['file_name'],
            "sentences": sentence,
            "orig_size": np.array([h, w]),
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