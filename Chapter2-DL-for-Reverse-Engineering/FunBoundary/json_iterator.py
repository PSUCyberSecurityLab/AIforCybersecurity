import numpy as np
import config
import pickle
import glob
import random
import json
from torch.utils.data import DataLoader, Dataset
import torch

class DGLDataset(Dataset):
    def __init__(self, Sequence):
        self.Sequence = Sequence

    def __len__(self):
        return len(self.Sequence)

    def __getitem__(self, idx):
        return self.Sequence[idx]


class SimpleBatch:
    def __init__(self, data):
        self.data = data

    def pin_memory(self):
        return self.data

def collate_wrapper(batch):
    return SimpleBatch(batch)

def split(Samples, train_val_split=0.1):
    Samples = Samples[:10000]
    random.shuffle(Samples)

    cross_validation_val = Samples[:int(train_val_split*len(Samples))]
    cross_validation_train = Samples[int(train_val_split*len(Samples)):]
    return cross_validation_train, cross_validation_val
    

def generator(jsons, BATCH_SIZE, NUM_WORKS):
    Sequences = []
    for f_name in jsons[:100]:
        with open(f_name) as inf:
            for line in inf:
                s_info = json.loads(line.strip())
                sequence = s_info[0]
                label = torch.tensor(s_info[1])
                Sequences.append([sequence,label])
                # print(s_info)
    cross_validation_train,cross_validation_val = split(Sequences)
    train_dataloader = DataLoader(cross_validation_train, 
                                    batch_size=BATCH_SIZE, 
                                    num_workers=NUM_WORKS, 
                                    collate_fn=collate_wrapper,
                                    pin_memory=False, shuffle=True)
    val_dataloader = DataLoader(cross_validation_val, 
                                batch_size=BATCH_SIZE, 
                                num_workers=NUM_WORKS, 
                                collate_fn=collate_wrapper,
                                pin_memory=False, shuffle=False)
    dataloaders = {'train':None, 'val':None}
    dataloaders['train'] = train_dataloader
    dataloaders['val'] = val_dataloader
    return dataloaders
