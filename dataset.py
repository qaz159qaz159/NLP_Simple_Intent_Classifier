from typing import List, Dict

import torch
from torch.utils.data import Dataset

import utils
from utils import Vocab

vocab = Vocab


class SeqClsDataset(Dataset):
    def __init__(self,
                 data: List[Dict],
                 vocab: Vocab,
                 label_mapping: Dict[str, int],
                 max_len: int,
                 ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        """
        :param samples: ((__getitem__), (__getitem__), ...)
        :return:
        """
        # TODO: implement collate_fn
        content = [samples[i]['text'] for i in range(len(samples))]
        target = [self.label2idx(samples[i]['intent']) for i in range(len(samples))]
        for i in range(len(content)):
            content[i] = content[i].split()
        content = self.vocab.encode_batch(batch_tokens=content, to_len=self.max_len)
        content, target = torch.LongTensor(content), torch.LongTensor(target)

        return content, target
        raise NotImplementedError

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
