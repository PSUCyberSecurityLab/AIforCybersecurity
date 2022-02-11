#!/usr/bin/python3
import copy
import gc
import gzip
import json
import os
import random
import numpy
import progressbar
import spacy
import tensorflow
import tensorflow.keras as keras

from models import mimick, sequence, util


class Codebook(util.Codebook):
    """
        Function: to construct a sequence with add-ons: begin-sequence, end-sequence, unknown
    """
    pass


class DablogSequence(sequence.EventSequence):
    """
        Function: to construct a sequence for DabLog inherited EventSequence
    """
    def __init__(self, config=None, sequence=None):
        super().__init__(config, sequence)
        self.nlp = spacy.load('en_core_web_md')
        self.unknowns = 0

    def batch(self, mode='autoencoder'):
        batchsize = self.config.batch
        self.config.batch = len(self.sequences)
        ret = self.next(mode)
        self.config.batch = batchsize
        self.reset()
        return ret

    def next(self, mode='autoencoder'):
        if not self.ready:
            self.reset()
        if self.idx >= len(self):
            self.reset()
            raise StopIteration
        self.idx += 1
        return self.__getitem__(self.idx - 1, mode)

    def __getitem__(self, idx, mode='autoencoder'):
        if mode == 'deeplog':
            return self.getItemDeeplog(idx)
        if mode == 'autoencoder':
            return self.getItemAutoencoder(idx)
        if mode == 'word2vec_fit':
            return self.getItemWord2Vec(idx, epochs=256)
        if mode == 'word2vec_predict':
            return self.getItemWord2Vec(idx, epochs=1)

    def getItemDeeplog(self, idx):
        if not self.ready:
            self.reset()
        XL, XF, YL, INFO = [], [], [], []
        begin = idx * self.config.batch
        end = (idx + 1) * self.config.batch
        maxlen = 0 if not self.config.static_length else self.config.seqlen
        for i in range(begin, end):
            idx = self.shuffled_indices[i] if self.config.shuffle else i
            sequence = [[self.codebook.BEGIN] + [0.0] * self.float_size]
            sequence += self.sequences[idx]
            sequence += [[self.codebook.END] + [0.0] * self.float_size]
            maxlen = max(maxlen, min(self.config.seqlen, len(sequence)))
            # build inputs
            for j in range(1, len(sequence)):
                xl = [self.codebook.enc(event[0]) for event in sequence[max(0, j - self.config.seqlen): j]]
                xf = [event[1:] for event in sequence[max(0, j - self.config.seqlen): j]]
                yl = [self.codebook.enc(sequence[j][0])]
                XL.append(xl)
                XF.append(xf)
                YL.append(yl)
                seqinfo = copy.copy(self.seqinfo[idx])
                seqinfo['idx'] = j
                INFO.append(seqinfo)
        # add paddings and transform
        for i in range(0, len(XL)):
            if len(XL[i]) < maxlen:
                padlen = maxlen - len(XL[i])
                XL[i] = [self.codebook.enc(self.codebook.PAD)] * padlen + XL[i]
                XF[i] = [[0.0] * self.float_size] * padlen + XF[i]
        # finalize
        XL = numpy.array(XL)
        XF = numpy.array(XF)
        YL = keras.utils.to_categorical(YL, num_classes=len(self.codebook))
        if self.config.verbose:
            print('Batch Shape:', XL.shape, XF.shape, YL.shape)
        if self.float_size > 0:
            return [XL, XF], YL, INFO
        return XL, YL, INFO

    def getItemAutoencoder(self, idx):
        if not self.ready:
            self.reset()
        XL, XF, YL, INFO = [], [], [], []
        begin = idx * self.config.batch
        end = (idx + 1) * self.config.batch
        maxlen = 0 if not self.config.static_length else self.config.seqlen
        for i in range(begin, end):
            idx = self.shuffled_indices[i] if self.config.shuffle else i
            sequence = [[self.codebook.BEGIN] + [0.0] * self.float_size]
            sequence += self.sequences[idx]
            sequence += [[self.codebook.END] + [0.0] * self.float_size]
            maxlen = max(maxlen, min(self.config.seqlen, len(sequence)))
            # build inputs
            for j in range(min(self.config.seqlen, len(sequence)), len(sequence) + 1):
                xl = [self.codebook.enc(event[0]) for event in sequence[max(0, j - self.config.seqlen): j]]
                xf = [event[1:] for event in sequence[max(0, j - self.config.seqlen): j]]
                XL.append(xl)
                XF.append(xf)
                seqinfo = copy.copy(self.seqinfo[idx])
                seqinfo['idx'] = j
                INFO.append(seqinfo)
        # add paddings and transform
        for i in range(0, len(XL)):
            if len(XL[i]) < maxlen:
                padlen = maxlen - len(XL[i])
                XL[i] = [self.codebook.enc(self.codebook.PAD)] * padlen + XL[i]
                XF[i] = [[0.0] * self.float_size] * padlen + XF[i]
        # finalize
        XL = numpy.array(XL)
        XF = numpy.array(XF)
        YL = [xl[:: -1] for xl in XL]
        for i in range(0, len(YL)):
            for j in range(0, len(YL[i])):
                if YL[i][j] >= len(self.codebook) - self.unknowns:
                    YL[i][j] = self.codebook.enc(self.codebook.UNKNOWN)
        YL = keras.utils.to_categorical(YL, num_classes=len(self.codebook) - self.unknowns)
        if self.config.verbose: 
            print('Batch Shape:', XL.shape, XF.shape, YL.shape)
        if self.float_size > 0:
            return [XL, XF], YL, INFO
        return XL, YL, INFO

    def getItemWord2Vec(self, idx, epochs=1):
        if not self.ready:
            self.reset()
        # find-and-add unknown events
        for event in self.types:
            if event not in self.codebook.encode:
                self.codebook.add(event)
                self.unknowns += 1
                print('    \033[91mAdd Unknown Event to Codebook:\033[0m', self.codebook.enc(event), event)
        # build vectors
        vectors, maxlen = [], 0
        for event in [self.codebook.decode[i] for i in range(0, len(self.codebook))]:
            vectors.append([])
            for c in '.:*,_/=':
                event = event.replace(c, ' ')
            event = self.nlp(event)
            maxlen = max(maxlen, len(event))
            for word in event:
                vectors[-1].append(word.vector)
        # add paddings
        X, dim = [], len(vectors[0][0])
        for epoch in range(0, epochs):
            for i in range(0, len(vectors)):
                if len(vectors[i]) < maxlen:
                    padding, padlen = [], maxlen - len(vectors[i])
                    for j in range(0, padlen):
                        padding.append([random.random() for _ in range(0, dim)])
                    X.append(padding + vectors[i])
                else:
                    X.append(vectors[i])
        # finalize
        X = numpy.array(X)
        return X


class Dablog(object):
    @staticmethod
    def SetupGPU(gpulist=None):
        gpus = tensorflow.config.experimental.list_physical_devices('GPU')
        if gpulist is None:
            lgpus = gpus
        elif len(gpulist) == 1 and gpulist[0] == -1:
            lgpus = []
        else:
            lgpus = [gpus[i] for i in gpulist]
        tensorflow.config.experimental.set_visible_devices(devices=lgpus, device_type='GPU')
        for i in range(0, len(lgpus)):
            tensorflow.config.experimental.set_memory_growth(lgpus[i], True)
        print('\033[91mUsing GPU\n', '\n'.join([str(gpu) for gpu in lgpus]), '\033[0m')

    class Config(object):
        def __init__(self, hidden_layer=2, hidden_unit=64, epochs=256, batch_size=256, rank_threshold=0.05,
                     distance_threshold=0.05, use_repeat_vector=True, use_mimick_embedding=True, filepath=None,
                     use_gzip=True,
                     verbose=False):
            self.verbose = verbose
            self.hidden_layer = hidden_layer
            self.hidden_unit = hidden_unit
            self.epochs = epochs
            self.batch_size = batch_size
            self.rank_threshold = rank_threshold
            self.distance_threshold = distance_threshold
            self.use_repeat_vector = use_repeat_vector
            self.use_mimick_embedding = use_mimick_embedding
            self.filepath = filepath
            self.use_gzip = use_gzip

    def __init__(self, config=None):
        self.filepath = None
        self.model = None
        self.config = config if config is not None else Dablog.Config()
        self.mimick = None
        self.embedding = {}

    def pretrain(self, trainset, mode='autoencoder'):
        global float_input, merge, model, X, Y, I
        if not self.config.use_mimick_embedding: 
            return
        # initialize variables
        gc.collect()
        n_labels = len(trainset.codebook)
        n_floats = trainset.float_size
        nn_size = self.config.hidden_unit + n_floats
        seqlen = min(trainset.maxlen + 2, trainset.config.seqlen)
        # input label sequence
        label_input = keras.layers.Input(shape=(None,), name='Label_Input')
        label_embed = keras.layers.Embedding(n_labels, self.config.hidden_unit, mask_zero=True, name='Label_Embed')
        label_dense = label_embed(label_input)
        # input float sequence
        if n_floats > 0:
            float_input = keras.layers.Input(shape=(None, n_floats), name='Float_Input')
            float_dense = keras.layers.Dense(n_floats, name='Float_Dense')(float_input)
            merge = keras.layers.concatenate([label_dense, float_dense], axis=2)
        # neural network layer
        if mode == 'deeplog':
            nn = keras.models.Sequential(name='RNN')
            for i in range(1, self.config.hidden_layer):
                nn.add(keras.layers.LSTM(nn_size, activation='relu', return_sequences=True, name='LSTM_' + str(i)))
            nn.add(keras.layers.LSTM(nn_size, name='LSTM_' + str(self.config.hidden_layer)))
            nn.add(keras.layers.Dense(n_labels, activation='softmax', name='Softmax'))
            # build model
            if n_floats == 0:
                model = keras.models.Model(inputs=label_input, outputs=nn(label_dense))
            else:
                model = keras.models.Model(inputs=[label_input, float_input], outputs=nn(merge))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
            if self.config.verbose: 
                model.summary()
                nn.summary() 
            # train deeplog
            X, Y, I = trainset.batch(mode='deeplog')
        elif mode == 'autoencoder':
            # autoencoder layer
            nn = keras.models.Sequential(name='RNN')
            nn_sizes = [int(nn_size / i) for i in range(1, self.config.hidden_layer + 1)]
            nn_sizes = [0] + nn_sizes + nn_sizes[:: -1]
            for i in range(1, self.config.hidden_layer):
                nn.add(keras.layers.LSTM(nn_sizes[i], activation='relu', return_sequences=True, name='LSTM_' + str(i)))
            i = self.config.hidden_layer
            if self.config.use_repeat_vector:  # bi-direction
                nn.add(keras.layers.LSTM(nn_sizes[i], activation='relu', return_sequences=False, name='LSTM_' + str(i)))
                nn.add(keras.layers.RepeatVector(seqlen))
            else:
                nn.add(keras.layers.LSTM(nn_sizes[i], activation='relu', return_sequences=True, name='LSTM_' + str(i)))
            
            for i in range(self.config.hidden_layer + 1, 2 * self.config.hidden_layer + 1):
                nn.add(keras.layers.LSTM(nn_sizes[i], activation='relu', return_sequences=True, name='LSTM_' + str(i)))
            nn.add(keras.layers.TimeDistributed(keras.layers.Dense(n_labels, activation='softmax'), name='Softmax'))
            # model
            if n_floats == 0:
                model = keras.models.Model(inputs=label_input, outputs=nn(label_dense))
            else:
                model = keras.models.Model(inputs=[label_input, float_input], outputs=nn(merge))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
            if self.config.verbose: 
                model.summary()
                nn.summary()
            # train dablog
            X, Y, I = trainset.batch(mode='autoencoder')

        callbacks = [keras.callbacks.EarlyStopping(monitor='loss', patience=64)]
        model.fit(X, Y, shuffle=True, verbose=self.config.verbose, batch_size=self.config.batch_size,
                  epochs=self.config.epochs, callbacks=callbacks)
        # get embedding
        embedding = list(label_embed.get_weights()[0])
        # destroy network
        keras.backend.clear_session()
        del model
        del X
        del Y
        del I
        gc.collect()
        # build and train mimick
        X, Y = trainset.batch(mode='word2vec_fit'), []
        for i in range(0, len(X) // len(embedding)): 
            Y.extend(embedding)
        Y = numpy.array(Y)
        self.mimick = mimick.Word2VecMimickEmbedding(X.shape[2], Y.shape[1], filepath=self.config.filepath, verbose=self.config.verbose)
        self.mimick.train(X, Y)

    def load_mimick(self, filepath=None):
        if filepath is None:
            filepath = self.filepath
        self.mimick = mimick.Word2VecMimickEmbedding(0, 0, filepath=filepath, verbose=self.config.verbose)
        self.mimick.load()

    def train(self, trainset):
        return self.fit(trainset)

    def fit(self, trainset):
        if self.model is None:
            # preprocess
            gc.collect()
            self.n_labels = len(trainset.codebook)
            self.n_floats = trainset.float_size
            self.nn_size = self.config.hidden_unit + self.n_floats
            self.seqlen = min(trainset.maxlen + 2, trainset.config.seqlen)
            # input label sequence
            if self.mimick is None:
                self.label_input = keras.layers.Input(shape=(None,), name='Label_Input')
                self.label_embed = keras.layers.Embedding(self.n_labels, self.config.hidden_unit, mask_zero=True, name='Label_Embed')
                self.label_dense = self.label_embed(self.label_input)
            else: 
                self.label_input = keras.layers.Input(shape=(None, self.config.hidden_unit), name='Label_Input')
                self.label_dense = keras.layers.Dense(self.config.hidden_unit)(self.label_input)
                # self.label_dense = keras.layers.Dense (self.config.hidden_unit) (self.label_dense)
            # input float sequence
            if self.n_floats > 0:
                self.float_input = keras.layers.Input(shape=(None, self.n_floats), name='Float_Input')
                self.float_dense = keras.layers.Dense(self.n_floats, name='Float_Dense')(self.float_input)
                if self.mimick is None:
                    self.merge = keras.layers.concatenate([self.label_dense, self.float_dense], axis=2)
                else:
                    self.merge = keras.layers.concatenate([self.label_input, self.float_dense], axis=2)
            # neural network layer
            self.nn = keras.models.Sequential(name='RNN')
            nn_sizes = [int(self.nn_size / i) for i in range(1, self.config.hidden_layer + 1)]
            nn_sizes = [0] + nn_sizes + nn_sizes[:: -1] # reverse
            for i in range(1, self.config.hidden_layer):
                self.nn.add(keras.layers.LSTM(nn_sizes[i], activation='relu', return_sequences=True, name='LSTM_' + str(i)))

            hidden_layer = self.config.hidden_layer
            if self.config.use_repeat_vector:
                self.nn.add(keras.layers.LSTM(nn_sizes[hidden_layer], activation='relu', return_sequences=False, name='LSTM_' + str(hidden_layer)))
                self.nn.add(keras.layers.RepeatVector(self.seqlen))
            else:
                self.nn.add(keras.layers.LSTM(nn_sizes[hidden_layer], activation='relu', return_sequences=True, name='LSTM_' + str(hidden_layer)))

            for i in range(self.config.hidden_layer + 1, 2 * self.config.hidden_layer + 1):
                self.nn.add(keras.layers.LSTM(nn_sizes[i], activation='relu', return_sequences=True, name='LSTM_' + str(i)))

            self.nn.add(keras.layers.TimeDistributed(keras.layers.Dense(self.n_labels, activation='softmax'), name='Softmax'))
            # model
            if self.n_floats == 0:
                if self.mimick is None:
                    self.model = keras.models.Model(inputs=self.label_input, outputs=self.nn(self.label_dense))
                else:
                    self.model = keras.models.Model(inputs=self.label_input, outputs=self.nn(self.label_input))
            else:
                self.model = keras.models.Model(inputs=[self.label_input, self.float_input], outputs=self.nn(self.merge))
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
            if self.config.verbose:
                self.model.summary()
                self.nn.summary()

        X, Y, I, V = self.getBatch(trainset)
        del X
        gc.collect()
        return self.model.fit(V, Y, shuffle=True, verbose=self.config.verbose, batch_size=self.config.batch_size,
                              epochs=self.config.epochs, callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=64)])

    def getBatch(self, dataset):
        if self.mimick is not None:
            # update embedding
            self.embedding = self.mimick.predict(dataset.batch(mode='word2vec_predict'))
            # extract batch
            X, Y, I = dataset.batch()
            XL = X[0] if self.n_floats > 0 else X
            shape = XL.shape
            XL = XL.tolist()
            # replace batch with vectors
            for seq in range(0, shape[0]):
                for evt in range(0, shape[1]):
                    XL[seq][evt] = self.embedding[XL[seq][evt]]
            V = [numpy.array(XL), X[1]] if self.n_floats > 0 else numpy.array(XL)
            print('Batch Shape: [', shape, '--->', V.shape, ']', X[1].shape, Y.shape)
        else:
            X, Y, I = dataset.batch()
            V = X
        return X, Y, I, V

    def test(self, testset):
        return self.predict(testset)

    def predict(self, testset):
        # temporarily overwrite / save-and-restore config for getBatch
        global bar
        config = testset.config.seqlen, testset.config.static_length
        testset.config.seqlen, testset.config.static_length = self.seqlen, True
        X, Y, I, V = self.getBatch(testset)
        # testset.config.seqlen, testset.config.static_length = config
        # test
        if self.config.verbose:
            print('Testing')
        P = self.model.predict(V, batch_size=self.config.batch_size, verbose=True)
        del V
        gc.collect()
        if self.config.verbose:
            print('Done Testing')
        # check the misses and return anomalies
        dij = util.EmbeddingDistance(self.getEmbeddings())
        filepath = os.path.join(self.config.filepath, 'predict')
        fout = gzip.open(filepath + '.gz', 'wt') if self.config.use_gzip else open(filepath, 'w')
        n_miss = 0
        if self.config.verbose:
            print('Deriving Anomalies')
            bar = progressbar.ProgressBar(term_width=100, maxval=P.shape[0], widgets=[
                progressbar.Bar('=', '[', ']'), ' ',
                progressbar.Percentage(), ' ',
                progressbar.Timer(), ' ',
                progressbar.ETA()])
            bar.start()
        for seq in range(0, P.shape[0]):
            if self.config.verbose and seq & 1023 == 0:
                bar.update(seq)
            append_to_result = False
            for evt in range(0, P.shape[1]):
                # first condition: the label is not top-predicted
                # get the rank of label by finding the corresponding prob in the sorted probs
                x_label = X[seq][P.shape[1] - 1 - evt]
                y_label = numpy.argmax(Y[seq][evt])
                sortProbs = sorted(P[seq][evt], reverse=True)
                rank = float(sortProbs.index(P[seq][evt][y_label])) / P.shape[2]
                # conditions
                if rank > self.config.rank_threshold:
                    append_to_result = True
                    break
                elif self.config.distance_threshold < 1:  # second condition: the label is far away from top-predicts
                    prob_threshold = sortProbs[min(int(self.config.rank_threshold * P.shape[2]) + 1, P.shape[2] - 1)]
                    dists = [dij[x_label][j] if P[seq][evt][j] > prob_threshold else 1.0 for j in range(0, P.shape[2])]
                    if len(dists) > 0 and min(dists) > self.config.distance_threshold:
                        append_to_result = True
                        break
            if append_to_result:  # append prediction-miss (append once for a sequence)
                n_miss += 1
                fout.write(json.dumps({'X': X[seq].tolist(), 'P': P[seq].tolist(), 'I': I[seq]}) + '\n')
        if self.config.verbose:
            bar.finish()
        fout.close()
        return n_miss, self.readfile(filepath)

    def getEmbeddings(self, codebook=None):
        if len(self.embedding) == 0: 
            self.embedding = self.label_embed.get_weights()[0]
        if codebook is not None:
            return {codebook.dec(i): self.embedding[i] for i in range(0, len(codebook))}
        return {i: self.embedding[i] for i in range(0, len(self.embedding))}

    @staticmethod
    def split(miss):
        ret, X, P, I = [], miss['X'], miss['P'], miss['I']
        for idx in range(0, len(X)):
            x_idx, y_idx = idx, len(X) - 1 - idx
            x, y = X[x_idx], numpy.argmax(P[y_idx])
            if x != y:
                ret.append({'X': X, 'P': P[y_idx], 'I': I, 'idx': x_idx, 'x': x, 'y': y})
        print(X, [int(numpy.argmax(P[len(X) - 1 - idx])) for idx in range(0, len(X))])
        return ret

    @staticmethod
    def readfile(filepath):
        for line in open(filepath, 'r'):
            yield json.loads(line)
