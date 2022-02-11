import os
from config import *
from torch import nn
from scipy.ndimage.filters import gaussian_filter1d
from torch.autograd import Variable
import torch
import numpy as np
import eval_utils as utils
import glob
import time
import pdb
from json_iterator import generator

import torch.nn as nn
if torch.cuda.is_available():
    device = torch.device("cuda")#'cuda'
    print("***run on GPU***")
    CUDA_DEVICE = device
else:
    device = torch.device('cpu')
    print("***run on CPU***")



class NN(nn.Module):
    def __init__(self, hidden_size, sequence_len, num_classes, device):
        super(NN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.sequence_len = sequence_len
        self.num_classes = num_classes

        # Bi-LSTM
		# Forward and backward
        self.lstm_cell_forward = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm_cell_backward = nn.LSTMCell(self.hidden_size, self.hidden_size)

        # LSTM layer
        self.lstm_cell = nn.LSTMCell(self.hidden_size * 2, self.hidden_size * 2)

        # Linear layer
        self.linear = nn.Linear(self.hidden_size * 2, self.num_classes)


    def forward(self, x):
        hs_forward = torch.zeros(1, self.hidden_size).to(self.device)
        cs_forward = torch.zeros(1, self.hidden_size).to(self.device)
        hs_backward = torch.zeros(1, self.hidden_size).to(self.device)
        cs_backward = torch.zeros(1, self.hidden_size).to(self.device)

        # LSTM
        hs_lstm = torch.zeros(1, self.hidden_size * 2).to(self.device)
        cs_lstm = torch.zeros(1, self.hidden_size * 2).to(self.device)

        # Weights initialization
        torch.nn.init.kaiming_normal_(hs_forward)
        torch.nn.init.kaiming_normal_(cs_forward)
        torch.nn.init.kaiming_normal_(hs_backward)
        torch.nn.init.kaiming_normal_(cs_backward)
        torch.nn.init.kaiming_normal_(hs_lstm)
        torch.nn.init.kaiming_normal_(cs_lstm)

        # Prepare the shape for LSTM Cells
        out = x.view(20,1,128)

        forward = []
        backward = []
        fbhiddens = torch.zeros(0,self.hidden_size * 2).to(self.device)

        # Unfolding Bi-LSTM
        # Forward
        for i in range(self.sequence_len):
            # pdb.set_trace()
            hs_forward, cs_forward = self.lstm_cell_forward(out[i], (hs_forward, cs_forward))
            forward.append(hs_forward)
            
        # Backward
        for i in reversed(range(self.sequence_len)):
            hs_backward, cs_backward = self.lstm_cell_backward(out[i], (hs_backward, cs_backward))
            backward.append(hs_backward)

        # LSTM
        for fwd, bwd in zip(forward, backward):
            input_tensor = torch.cat((fwd, bwd), 1)
            hs_lstm, cs_lstm = self.lstm_cell(input_tensor, (hs_lstm, cs_lstm))
            # fbhiddens.append(hs_lstm)
            
            fbhiddens = torch.cat((fbhiddens, hs_lstm), 0)
        
        # Last hidden state is passed through a linear layer
        out = self.linear(fbhiddens)
        return out

    
class Classifier(object):
    def __init__(self):
        self.mode = "train"
        self.device = device
        self.palmtree = utils.UsableTransformer(model_path="./palmtree/transformer.ep19", vocab_path="./palmtree/vocab")
        # self.palmtree = self.palmtree.to(self.device)
        self.dataloaders = generator(glob.glob('Data/x86_O1/bbjsons/*.json'), BATCH_SIZE, NUM_WORKS)
        self.model = NN(128,20,2, device)
        self.model = self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=10,
                                                                    verbose=True, min_lr=1e-6)

    def iterate(self, phase, dataloader, epoch):
        total_loss = 0.0
       
        
        for batch_idx, Sequences in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print("-- %s on %d-st batch" % (phase, batch_idx))
            self.optimizer.zero_grad()
            batch_acc = 0.0
            batch_loss = 0.0
            batch_TP, batch_TN, batch_FP, batch_FN = 0, 0, 0, 0

            
            for sample in Sequences.pin_memory():
                if sample is None: # failed building DGL
                    continue
                sequence = sample[0]
                labels = sample[1].to(self.device)
                # print(sequence)
                embeddings = self.palmtree.encode(sequence)
                embeddings = torch.tensor(embeddings).to(self.device)
                logits = self.model(embeddings)
                loss = self.criterion(logits, labels)
                batch_loss += loss

                logits = logits.detach().cpu().argmax(dim=1)
                labels = labels.detach().cpu()

                batch_TP += torch.sum((labels == 1) & (logits == labels)).data.numpy()
                batch_TN += torch.sum((labels == 0) & (logits == labels)).data.numpy()
                batch_FN += torch.sum((labels == 1) & (logits != labels)).data.numpy()
                batch_FP += torch.sum((labels == 0) & (logits != labels)).data.numpy()


            if phase == 'train':
                batch_loss.backward()
                self.optimizer.step()

           
            total_loss += float(batch_loss.item()) 
            p = batch_TP / (batch_TP + batch_FP + 0.0001)
            r = batch_TP / (batch_TP + batch_FN + 0.0001)
            batch_F1 = 2 * r * p / (r + p + 0.0001)
            batch_acc = (batch_TP + batch_TN)/(batch_TP + batch_TN + batch_FP + batch_FN)
            if phase == 'train':
                print("train acc:%f, F1:%f, loss:%f" % (batch_acc, batch_F1, batch_loss))
            else:
                print("validation acc:%f, F1:%f, loss:%f" % (batch_acc, batch_F1, batch_loss))
        return total_loss


    def train(self): 
        train_dataloader = self.dataloaders['train']
        val_dataloader = self.dataloaders['val']
        print("# batches of sequence for training", len(train_dataloader))
        print("# batches of sequence for validation", len(val_dataloader))
        self.model.train()
        for epoch in range(1, 50):
            train_loss = self.iterate('train', train_dataloader,epoch)
            with torch.no_grad():
                val_loss = self.iterate('val', val_dataloader, epoch)
                self.scheduler.step(val_loss)

    
    def run(self):
        if self.mode == 'train':
            self.train()
        elif self.mode == 'test':
            self.test()
        elif self.mode == 'ctest':
            self.case_by_case_test()

classifier = Classifier()
classifier.run()


