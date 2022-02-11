import tensorflow as tf
import numpy as np
import config
import pickle
from binary_structure import *

class func_iterator(tf.keras.utils.Sequence):
    def __init__(self, x_train, y_train, y, batch_size, shuffle=True):
        self.x_train = x_train
        self.y_train = y_train
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.idx = list(range(x_train.shape[0]))
        self.on_epoch_end()

    def __len__(self) :
        return (np.ceil(len(self.idx) / float(self.batch_size))).astype(np.int)
    
    def __getitem__(self, idx) :
        o1 = np.vstack([self.x_train[idx] for idx in self.idx[idx * self.batch_size : (idx+1) * self.batch_size]])
        o2 = np.vstack([self.y_train[idx] for idx in self.idx[idx * self.batch_size : (idx+1) * self.batch_size]])
        o3 = np.vstack([self.y[idx] for idx in self.idx[idx * self.batch_size : (idx+1) * self.batch_size]])
        return o1, o2, o3

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)

class func_pair_iterator_highhalf(tf.keras.utils.Sequence):
    def __init__(self, batch_size, shuffle=True):
        with open('data/func_pairs_seq.pickle','rb') as jf:
            self.o1, self.o3 = pickle.load(jf)
        self.o1 = self.o1[int(len(self.o1)/2):]
        self.o3 = self.o3[int(len(self.o3)/2):]
        self.idx = list(range(len(self.o1)))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self) :
        return (np.ceil(len(self.idx) / float(self.batch_size))).astype(np.int)
    
    def __getitem__(self, idx) :
        
        o1 = np.vstack([self.o1[idx][0] for idx in self.idx[idx * self.batch_size : (idx+1) * self.batch_size]])
        y_tmp = np.vstack([self.o1[idx][1] for idx in self.idx[idx * self.batch_size : (idx+1) * self.batch_size]])
        o1_y = np.zeros((self.batch_size, 1, 2), dtype='int32')
        for id in range(len(y_tmp)):
            o1_y[id,0, y_tmp[id]] = 1

        o3 = np.vstack([self.o3[idx][0] for idx in self.idx[idx * self.batch_size : (idx+1) * self.batch_size]])
        y_tmp = np.vstack([self.o3[idx][1] for idx in self.idx[idx * self.batch_size : (idx+1) * self.batch_size]])
        o3_y = np.zeros((self.batch_size,1, 2), dtype='int32')
        for id in range(len(y_tmp)):
            o3_y[id,0, y_tmp[id]] = 1
        return o1, o1_y, o3, o3_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)
    

class func_pair_iterator_lowhalf(tf.keras.utils.Sequence):
    def __init__(self, batch_size, shuffle=True):
        with open('data/func_pairs_seq.pickle','rb') as jf:
            self.o1, self.o3 = pickle.load(jf)
        self.o1 = self.o1[0:int(len(self.o1)/2)]
        self.o3 = self.o3[0:int(len(self.o3)/2)]
        self.idx = list(range(len(self.o1)))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self) :
        return (np.ceil(len(self.idx) / float(self.batch_size))).astype(np.int)
    
    def __getitem__(self, idx) :
        
        o1 = np.vstack([self.o1[idx][0] for idx in self.idx[idx * self.batch_size : (idx+1) * self.batch_size]])
        y_tmp = np.vstack([self.o1[idx][1] for idx in self.idx[idx * self.batch_size : (idx+1) * self.batch_size]])
        o1_y = np.zeros((self.batch_size, 1, 2), dtype='int32')
        for id in range(len(y_tmp)):
            o1_y[id,0, y_tmp[id]] = 1

        o3 = np.vstack([self.o3[idx][0] for idx in self.idx[idx * self.batch_size : (idx+1) * self.batch_size]])
        y_tmp = np.vstack([self.o3[idx][1] for idx in self.idx[idx * self.batch_size : (idx+1) * self.batch_size]])
        o3_y = np.zeros((self.batch_size,1, 2), dtype='int32')
        for id in range(len(y_tmp)):
            o3_y[id,0, y_tmp[id]] = 1
        return o1, o1_y, o3, o3_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)

class pair_txt_iterator(tf.keras.utils.Sequence):
    def __init__(self, filenames, batch_size, shuffle=True):
        self.filenames = filenames
        self.idx = list(range(len(self.filenames[0])))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self) :
        return (np.ceil(len(self.idx) / float(self.batch_size))).astype(np.int)
    
    def get_info(self, fns):
        batch_x = []
        batch_shape = []
        start = 0
        for x in fns:
            batch_x.append((np.loadtxt(x).reshape(-1, config.sequence_length)))
            batch_shape.append([start, start + batch_x[-1].shape[0]])
            start += batch_x[-1].shape[0]
        return (np.vstack(batch_x), np.array(batch_shape))

    def __getitem__(self, idx) :
        for _idx in self.idx[idx * self.batch_size : (idx+1) * self.batch_size]:
            if self.filenames[0][_idx].split('/')[-1] != self.filenames[1][_idx].split('/')[-1]:
                print(self.filenames[0][_idx], self.filenames[1][_idx])
                exit()
        fns = [self.filenames[0][idx] for idx in self.idx[idx * self.batch_size : (idx+1) * self.batch_size]]
        x86 = self.get_info(fns)
        fns = [self.filenames[1][idx] for idx in self.idx[idx * self.batch_size : (idx+1) * self.batch_size]]
        arm = self.get_info(fns)
        return x86, arm

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)


class pair_iterator(tf.keras.utils.Sequence):
    def padding(self, tmp):
        length = len(tmp) if len(tmp) % config.sequence_length == 0 else config.sequence_length - len(tmp) % config.sequence_length + len(tmp)
        res = np.ones(length) * (config.max_features-1)
        res[:len(tmp)] = tmp
        return np.reshape(res, (-1, config.sequence_length))

    def __init__(self, batch_size, shuffle=True):
        '''
        data = pickle.load(open('data/pairs.bb','rb'))
        pairs = data['bb_pairs']
        self.arm_data = []
        self.x86_data = []
        for pair in pairs:
            self.arm_data.append(self.padding(pair['arm_bb'].get_binary()))    
            self.x86_data.append(self.padding(pair['intel_bbs'].get_binary()))
        with open('data/pairs.pickle','wb') as jf:
            pickle.dump((self.arm_data, self.x86_data), jf)
        '''
        self.all_data = pickle.load(open('data/pairs.pickle','rb'))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()    

    def __len__(self) :
        return (np.ceil(len(self.all_data[0]) / float(self.batch_size))).astype(np.int)
        
    def __getitem__(self, idx):
        arm = self.all_data[0][idx * self.batch_size : (idx+1) * self.batch_size]
        x86 = self.all_data[1][idx * self.batch_size : (idx+1) * self.batch_size]

        arm_batch_shape = []
        x86_batch_shape = []
        arm_start = 0
        x86_start = 0
        for i in range(self.batch_size):
            arm_batch_shape.append([arm_start, arm_start + arm[i].shape[0]])
            arm_start += arm[i].shape[0]
            x86_batch_shape.append([x86_start, x86_start + x86[i].shape[0]])
            x86_start += x86[i].shape[0]
        return (np.vstack(arm), np.array(arm_batch_shape)),(np.vstack(x86), np.array(x86_batch_shape))

    def on_epoch_end(self):
        pass


class text_iterator(tf.keras.utils.Sequence) :
    def __init__(self, filenames, batch_size, shuffle=True) :
        self.filenames = filenames
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self) :
        return (np.ceil(len(self.filenames) / float(self.batch_size))).astype(np.int)
    
    def __getitem__(self, idx) :
        fns = self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = []
        batch_x = []
        batch_shape = []
        query = []
        start = 0
        for x in fns:
            batch_x.append((np.loadtxt(x).reshape(-1, config.sequence_length)))
            batch_y.append(config.malware_label if x.find('Virus') != -1 else config.benign_label)
            batch_shape.append([start, start + batch_x[-1].shape[0]])
            start += batch_x[-1].shape[0]
            query.extend([batch_y[-1]]*batch_x[-1].shape[0])
        return (np.vstack(batch_x), np.array(batch_shape), np.array(query)), np.array(batch_y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.filenames)

    def get_labels(self):
        y = []
        for x in self.filenames:
            y.append(config.malware_label if x.find('Virus') != -1 else config.benign_label)
        return y
