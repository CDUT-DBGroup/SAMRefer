import os
# from dataset.RefTR_Dataset import denorm 
import torch.utils.data as data
import torch

from PIL import Image
from dataset.refer import REFER
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
import numpy as np
from transformers import BertTokenizer, BertModel
from transformers import CLIPTextModel, CLIPTokenizer
# sentence = 'new_sentenc'
sentence = 'sentences'
def pil_resize(img, size, order):
    if size[0] == img.shape[0] and size[1] == img.shape[1]:
        return img

    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST

    return np.asarray(Image.fromarray(img).resize(size[::-1], resample))


def pil_rescale(img, scale, order):
    height, width = img.shape[:2]
    target_size = (int(np.round(height * scale)), int(np.round(width * scale)))
    return pil_resize(img, target_size, order)


class ReferDataset(data.Dataset):
    def __init__(self,
                 refer_data_root='data',
                 dataset='refcoco',
                 splitBy='unc',
                 bert_tokenizer='clip',
                 image_transforms=None,
                 max_tokens=30,
                 split='train',
                 eval_mode=True,
                 size=480,
                 scales=False,
                 negative_samples=0,
                 positive_samples=1,
                 pseudo_path=None,
                 precision='fp32') -> None:
        """
        parameters:
            refer_data_root: root directory of the dataset
            dataset: dataset name (refcoco, refcoco+, refcocog)
            splitBy: split method (unc, google, umd)
            bert_tokenizer: tokenizer type ('clip' or 'bert')
            image_transforms: transforms apply to image and mask
            max_tokens: maximum length of text tokens (30 as per paper)
            split: ['train','val','testA','testB']
            eval_mode: whether in training or evaluating
            size: image size (480x480 as per paper)
            precision: model precision ('fp32', 'fp16', or 'bf16')
        """
        self.clip = ('clip' in bert_tokenizer)
        self.negative_samples = negative_samples
        self.positive_samples = positive_samples
        self.classes = []
        self.image_transforms = image_transforms
        self.split = split
        self.refer = REFER(refer_data_root, dataset, splitBy)
        self.scales = scales
        self.size = size
        self.pseudo_path = pseudo_path
        self.precision = precision

        # Set torch dtype based on precision
        if precision == "bf16":
            self.torch_dtype = torch.bfloat16
        elif precision == "fp16":
            self.torch_dtype = torch.half
        else:
            self.torch_dtype = torch.float32

        print('\nPreparing dataset .....')
        print(dataset, split)
        print(refer_data_root, dataset, splitBy)
        print(f'pseudo_path = {pseudo_path}')

        self.max_tokens = max_tokens

        # Initialize tokenizers
        if self.clip:
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)
        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)

        self.ref_ids = ref_ids
        self.eval_mode = eval_mode

        self.input_ids = []
        self.word_masks = []
        self.all_sentences = []
        self.refid2index = {}

        for index, r in enumerate(self.ref_ids):
            self.refid2index[r] = index
            ref = self.refer.Refs[r]
            sentences_for_ref = []
            attentions_for_ref = []
            sentence_raw_for_re = []

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['sent']

                if self.clip:
                    # CLIP tokenization
                    tokens = self.tokenizer(
                        sentence_raw,
                        max_length=self.max_tokens,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    word_id = tokens['input_ids'].squeeze(0)
                    word_mask = tokens['attention_mask'].squeeze(0)
                else:
                    # BERT tokenization
                    tokens = self.tokenizer(
                        sentence_raw,
                        max_length=self.max_tokens,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    word_id = tokens['input_ids'].squeeze(0)
                    word_mask = tokens['attention_mask'].squeeze(0)

                sentences_for_ref.append(word_id)
                attentions_for_ref.append(word_mask)
                sentence_raw_for_re.append(sentence_raw)

            self.input_ids.append(sentences_for_ref)
            self.word_masks.append(attentions_for_ref)
            self.all_sentences.append(sentence_raw_for_re)
        print('Dataset prepared!')

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img_path = os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])
        img = Image.open(img_path).convert("RGB")

        ref = self.refer.loadRefs(this_ref_id)[0]

        bbox = self.refer.Anns[ref['ann_id']]['bbox']
        bbox = np.array(bbox, dtype=int)
        bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]

        ref_mask = np.array(self.refer.getMask(ref)['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1
        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.image_transforms is not None:
            h, w = ref_mask.shape
            img, target = self.image_transforms(img, annot)
        else:
            target = annot

        if self.eval_mode:
            # In eval mode, we still only use one sentence to ensure consistent batch sizes
            choice_sent = 0  # Use the first sentence for evaluation
            word_ids = self.input_ids[index][choice_sent]
            word_masks = self.word_masks[index][choice_sent]
            sentences = self.all_sentences[index][choice_sent]
        else:
            choice_sent = np.random.choice(len(self.input_ids[index]))
            word_ids = self.input_ids[index][choice_sent]
            word_masks = self.word_masks[index][choice_sent]
            sentences = self.all_sentences[index][choice_sent]

        img_path_full = this_img['file_name']
        img_path = int(img_path_full.split('.')[0].split('_')[-1])

        # Convert tensors to specified dtype
        img = img.to(dtype=self.torch_dtype)
        word_ids = word_ids.to(dtype=torch.long)
        word_masks = word_masks.to(dtype=torch.long)
        target = target.to(dtype=torch.float32)  # Changed to float32 for loss computation

        samples = {
            "img": img,
            "text": sentences,
            "word_ids": word_ids,
            "word_masks": word_masks,
        }

        targets = {
            "mask": target,  # Changed to match SegMaskLoss requirements
            "img_path": img_path,
            "sentences": sentences,
            "boxes": bbox,
            "orig_size": np.array([h, w]),
            "img_path_full": img_path_full
        }

        return samples, targets


if __name__ == '__main__':
    from transform import get_transform
    import numpy as np
    import json
    from torch.utils.data import DataLoader

    refcoco_train = ReferDataset(dataset='refcoco', splitBy='unc', split='train', refer_data_root='/vision_paper/paper_data/coco_data/', eval_mode=False,negative_samples=3,
                                 image_transforms=get_transform(320, train=False))
    refcoco_train.__getitem__(1)
    # train_loader = DataLoader(refcoco_train,
    #                           batch_size=12,
    #                           num_workers=2,
    #                           pin_memory=True,
    #                           sampler=None)
    # print('要进入循环了')
    # for idx, (img, target, bbox, word_ids, word_mask, _, raw_sentences) in enumerate(train_loader):
    #     print(idx, img.shape)
    #
    #     if idx > 10: break
