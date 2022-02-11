#!/usr/bin/python3
import numpy
import random
import tensorflow.keras as keras


class EventSequence(keras.utils.Sequence):
    """
        Function: to construct an event sequence from raw sample
    """
    class Config(object):
        def __init__(self, seqlen=512, batch=2048, shuffle=False, static_length=False, verbose=False):
            self.seqlen = seqlen
            self.batch = batch
            self.shuffle = shuffle
            self.verbose = verbose
            self.static_length = static_length

    def __init__(self, config=None, sequence=None):
        self.config = config if config is not None else EventSequence.Config()
        self.sequences = []
        self.seqinfo = []
        self.shuffled_indices = []
        self.float_size = 0
        self.idx = 0
        self.maxlen = 0
        self.types = {}
        self.codebook = None
        self.ready = False
        self.append(sequence)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        pass

    def next(self):
        if not self.ready:
            self.reset()
        if self.idx >= len(self):
            self.reset()
            raise StopIteration
        self.idx += 1
        return self.__getitem__(self.idx - 1)

    def batch(self):
        batchsize = self.config.batch
        self.config.batch = len(self.sequences)
        ret = self.next()
        self.config.batch = batchsize
        self.reset()
        return ret

    def on_epoch_end(self):
        self.reset()

    def reset(self):
        self.idx = 0
        self.ready = True
        if self.config.shuffle:
            self.shuffled_indices = numpy.arange(len(self.sequences))
            random.shuffle(self.shuffled_indices)

    def append(self, sequence, seqinfo=None):
        if seqinfo is None:
            seqinfo = {}
        if sequence is None:
            return
        for i in range(0, len(sequence)):
            if isinstance(sequence[i], str):
                sequence[i] = [sequence[i]]
            self.float_size = max(self.float_size, len(sequence[i]) - 1)
            event = sequence[i][0]
            if event not in self.types:
                self.types[event] = 0
            self.types[event] += 1
        self.sequences.append(sequence)
        seqinfo['len'] = len(sequence)
        self.seqinfo.append(seqinfo)
        self.maxlen = max(self.maxlen, len(sequence))
        self.ready = False

    def getStats(self):
        lens = [len(seq) for seq in self.sequences]
        ret = {
            'n_seq': len(lens),
            'avg_len': numpy.mean(lens),
            'max_len': self.maxlen,
            'min_len': 0 if len(lens) == 0 else min(lens),
            'std_len': numpy.std(lens),
        }
        return ret
