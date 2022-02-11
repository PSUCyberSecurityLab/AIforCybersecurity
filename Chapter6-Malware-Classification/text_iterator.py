import tensorflow as tf
import numpy as np
import config

class text_iterator(tf.keras.utils.Sequence) :
    def __init__(self, filenames, y, batch_size, shuffle=True) :
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
            batch_x.append(np.loadtxt(x))
            batch_y.append(config.malware_label if x.find('Virus') != -1 else config.benign_label)
            batch_shape.append([start, start + batch_x[-1].shape[0]])
            start += batch_x[-1].shape[0]
            query.extend([batch_y[-1]]*batch_x[-1].shape[0])
        return (np.array([seq for sublist in batch_x for seq in sublist]), np.array(batch_shape), np.array(query)), np.array(batch_y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.filenames)

    def get_labels(self):
        y = []
        for x in self.filenames:
            y.append(config.malware_label if x.find('Virus') != -1 else config.benign_label)
        return y
