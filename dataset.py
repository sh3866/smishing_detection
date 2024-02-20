import torch
import random
import numpy as np
from typing import Union, List, Dict
from torch.utils.data import Dataset, DataLoader

class SMSDataset(Dataset):
    def __init__(self,
                 data: Dict[str, List],
                 tokenizer = None,
                 label_encoder=None,
                 max_length:int=512
                 ):

        self.data = data['TEXT']
        self.label = data['LABEL_ID']
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_encoder = label_encoder
                    
    def __len__(self):
        return len(self.label)
    
    def encode_sample(self, sample: Union[str, List[str]]):
        return self.tokenizer(sample,
                              padding=True,
                              truncation=True,
                              max_length = self.max_length,
                              return_tensors='pt')
    
    def __getitem__(self, idx):
        elements = {'data': self.data[idx],
                    'label': self.label[idx]}

        return elements

    def collate_fn(self, samples:Dict[str, List]):
        datas = [ s['data'] for s in samples]
        labels = [ s['label'] for s in samples]
        
        datas = self.encode_sample(datas)
        labels = torch.tensor(labels)

        elements = { k:v for k, v in datas.items()}
        elements['labels'] = labels

        return elements
        
    def random_seed(self, seed:int = 42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
    def id2label(self, label:Union[List[int], int]):
        return self.label_encoder.inverse_transform(label)