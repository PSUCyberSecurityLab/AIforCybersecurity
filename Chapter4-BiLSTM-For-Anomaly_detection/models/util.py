#!/usr/bin/python3
import json
import matplotlib
import matplotlib.pyplot
import numpy
import os
import tensorflow.keras as keras

matplotlib.use('Agg')


class Codebook(object):
    """
        Function: to construct a sequence with add-ons: begin-sequence, end-sequence, unknown
    """
    PAD = '<pad>'
    BEGIN = '<sequence>'
    END = '</sequence>'
    UNKNOWN = '<unknown>'

    def __init__(self, codebook=None, threshold=0.01):
        self.encode, self.decode, self.types = None, None, None
        if codebook is not None:
            if isinstance(codebook, dict):
                codebook = set(codebook.keys())
            if isinstance(codebook, list):
                codebook = set(codebook)
            if isinstance(codebook, set):
                for preserved in [self.PAD, self.BEGIN, self.END, self.UNKNOWN]:
                    if preserved in codebook:
                        codebook.remove(preserved)
                self.types = set(codebook)
                self.decode = {i: k for i, k in
                               enumerate([self.PAD, self.BEGIN, self.END, self.UNKNOWN] + sorted(self.types))}
                self.encode = {k: i for i, k in self.decode.items()}
                self.types = codebook | {self.PAD, self.BEGIN, self.END, self.UNKNOWN}

    def __len__(self):
        return len(self.encode)

    def __iter__(self):
        for i in range(0, len(self.encode)):
            yield self.encode[i]

    def enc(self, sequence):
        if isinstance(sequence, str):
            return self.encode[sequence] if sequence in self.encode else self.encode[self.UNKNOWN]
        return [self.encode[event] if event in self.encode else self.encode[self.UNKNOWN] for event in sequence]

    def dec(self, sequence, trim=False):
        if not isinstance(sequence, list):
            return self.decode[sequence]
        if trim: sequence = sequence[: self.strlen(sequence, text=False)]
        return [self.decode[label] for label in sequence]

    def categorical(self, label):
        return self.onehot(label)

    def onehot(self, label):
        if isinstance(label, list):
            return numpy.argmax(label)
        if isinstance(label, str):
            label = self.encode[label]
        return keras.utils.to_categorical(label, num_classes=len(self))

    def save(self, filepath):
        if not os.path.isdir(filepath):
            os.mkdir(filepath)
        with open(os.path.join(filepath, 'codebook'), 'w') as fout:
            fout.write(json.dumps(list(self.types)))

    def strlen(self, seq, text=True):
        terminator = [self.PAD, self.BEGIN, self.END, self.UNKNOWN]
        if not text:
            terminator = self.enc(terminator)
        for i in range(0, len(seq)):
            if seq[i] in terminator:
                return i
        return len(seq)

    def add(self, event):
        if event not in self.types:
            code = len(self.types)
            self.decode[code] = event
            self.encode[event] = code
            self.types.add(event)

    @classmethod
    def load(cls, filepath):
        return cls(json.loads(open(os.path.join(filepath, 'codebook'), 'r').read()))


class EmbeddingDistance(numpy.ndarray):
    """
        Function: to calculate the embedding distance of two embeddings
    """

    def __new__(cls, embed):
        dist = [list() for _ in embed]
        for i in embed:
            vec1 = numpy.array(embed[i])
            for j in embed:
                vec2 = numpy.array(embed[j])
                dist[i].append(numpy.linalg.norm(vec1 - vec2))
        dist = numpy.array(dist)
        unit = numpy.max(dist)
        dist = dist / unit
        ret = numpy.asarray(dist).view(cls)
        return ret

    def __init__(self, embed):
        self.embed = embed


class Statistics(object):
    """
        function: calculate F1 score (TP, TN, FP, FN), FPR, FNR, Recall, Accuracy
    """

    def __init__(self, steps, seqlen, normals, abnormals):
        self.steps = steps
        self.P = {
            'TP': [set() for _ in range(0, len(steps))],
            'FP': [set() for _ in range(0, len(steps))]}
        self.P_IDX = {
            'TP': [[0] * seqlen for _ in range(0, len(steps))],
            'FP': [[0] * seqlen for _ in range(0, len(steps))]}
        self.P_EVT = {'TP': {}, 'FP': {}}
        self.P_SEQ = {'TP': {}, 'FP': {}}
        self.normals = normals
        self.abnormals = abnormals
        self.metrics = {}

    def add(self, i, subject, label=None, evt_idx=None, seq_idx=None):
        if isinstance(subject, dict):
            label = subject['label']
            evt_idx = subject['evt_idx']
            seq_idx = subject['seq_idx']
            subject = subject['subject']
        if isinstance(label, str):
            label = True if label in ['Abnormal', 'abnormal'] else False
        P = 'TP' if label else 'FP'
        self.P[P][i].add(subject)
        self.P_IDX[P][i][evt_idx] += 1
        if subject not in self.P_EVT[P]:
            self.P_EVT[P][subject] = [set() for _ in range(0, len(self.steps))]
            self.P_SEQ[P][subject] = [set() for _ in range(0, len(self.steps))]
        self.P_EVT[P][subject][i].add(seq_idx + evt_idx)
        self.P_SEQ[P][subject][i].add(seq_idx)

    def output(self, output):
        # output resulting anomalous sequences
        with open('.'.join([output, 'anom', 'results']), 'w') as fout:
            for i, s in enumerate(self.steps):
                line = {'step': s, 'subjects': {}}
                for P in ['TP', 'FP']:
                    for subject in self.P_SEQ[P]:
                        if len(self.P_SEQ[P][subject][i]) == 0:
                            continue
                        line['subjects'][subject] = {
                            'label': 'Normal' if P == 'FP' else 'Abnormal',
                            'seqid': sorted(list(self.P_SEQ[P][subject][i])), }
                fout.write(json.dumps(line) + '\n')
        # metrics
        TP = [len(s) for s in self.P['TP']]
        FP = [len(s) for s in self.P['FP']]
        TN = [self.normals - s for s in FP]
        FN = [self.abnormals - s for s in TP]
        with open('.'.join([output, 'metric']), 'w') as fout:
            fout.write(json.dumps({'TP': TP}))  # print ('TP', TP)
            fout.write(json.dumps({'FP': FP}))  # print ('FP', FP)
            fout.write(json.dumps({'TN': TN}))  # print ('TN', TN)
            fout.write(json.dumps({'FN': FN}))  # print ('FN', FN)
        # calculate metrics
        steps = self.steps
        self.metrics['FallOut (FPR)'] = [float(FP[i]) / (FP[i] + TN[i]) for i in range(0, len(steps))]
        self.metrics['Miss Rate (FNR)'] = [float(FN[i]) / (FN[i] + TP[i]) for i in range(0, len(steps))]
        self.metrics['Recall (TPR)'] = [float(TP[i]) / (TP[i] + FN[i]) for i in range(0, len(steps))]
        self.metrics['F1 Score'] = [float(2 * TP[i]) / (2 * TP[i] + FP[i] + FN[i]) for i in range(0, len(steps))]
        self.metrics['Accuracy'] = [float(TP[i] + TN[i]) / (TP[i] + FN[i] + FP[i] + TN[i]) for i in
                                    range(0, len(steps))]
        # Special handle for PPV due to division-by-zero
        self.metrics['Precision (PPV)'] = []
        for i in range(0, len(steps)):
            if TP[i] + FP[i] > 0:
                self.metrics['Precision (PPV)'].append(float(TP[i]) / (TP[i] + FP[i]))
            else:
                self.metrics['Precision (PPV)'].append(0.0)

        # AUC (area under ROC)
        def getAUC(steps, m1, m2):
            ret = [0]
            for _ in range(1, len(steps) + 1):
                index = len(steps) - i
                while len(ret) <= len(steps) and steps[len(ret) - 1] < m2[index]:
                    ret.append(ret[-1])
                ret[-1] = m1[index]
            for _ in range(len(ret) - 1, len(steps)):
                ret.append(ret[-1])
            ret = ret[1:]
            return ret

        self.metrics['AUC (Area under ROC)'] = getAUC(steps, self.metrics['Recall (TPR)'],
                                                      self.metrics['FallOut (FPR)'])
        self.metrics['Precision-Recall Curve'] = getAUC(steps, self.metrics['Precision (PPV)'],
                                                        self.metrics['Recall (TPR)'])
        # plot metrics
        fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot(1, 1, 1)
        for metric in sorted(list(self.metrics.keys())):
            ax.plot(steps, self.metrics[metric], '-', label=metric)
        matplotlib.pyplot.legend()
        fig.tight_layout()
        matplotlib.pyplot.savefig('.'.join([output, 'metric', 'png']), dpi=300)
        matplotlib.pyplot.close(fig)
        # plot heatmap
        fig, axs = matplotlib.pyplot.subplots(nrows=2, ncols=1, sharex='all')
        for i, P in enumerate(['TP', 'FP']):
            ax = axs[i]
            P_IDX = numpy.transpose(numpy.array(self.P_IDX[P]))
            max_P_IDX = numpy.max(P_IDX)
            if max_P_IDX > 0.0:
                P_IDX = P_IDX / max_P_IDX
            im = ax.imshow(P_IDX, cmap='gray')
            ax.set_xticks([0, len(self.steps) - 1])
            ax.set_xticklabels(['0.0', '1.0'])
            ax.set_title(P)
        fig.tight_layout()
        matplotlib.pyplot.savefig('.'.join([output, 'heatmap', 'png']), dpi=300)
        matplotlib.pyplot.close(fig)
        # plot evt and seq
        fig, axs = matplotlib.pyplot.subplots(nrows=2, ncols=2, sharex='all')
        maxevtcount, maxseqcount = 0, 0
        for P_SEQ in [self.P_SEQ['TP'], self.P_SEQ['FP']]:
            for subject in P_SEQ:
                for i, s in enumerate(self.steps):
                    maxseqcount = max(maxseqcount, len(P_SEQ[subject][i]))
        for P_EVT in [self.P_EVT['TP'], self.P_EVT['FP']]:
            for subject in P_EVT:
                for i, s in enumerate(self.steps):
                    maxevtcount = max(maxevtcount, len(P_EVT[subject][i]))
        for x, y, P, title, maxcount in [
            [0, 0, self.P_SEQ['TP'], 'TP SEQ (' + str(maxseqcount) + ')', maxseqcount],
            [0, 1, self.P_EVT['TP'], 'TP EVT (' + str(maxevtcount) + ')', maxevtcount],
            [1, 0, self.P_SEQ['FP'], 'FP SEQ (' + str(maxseqcount) + ')', maxseqcount],
            [1, 1, self.P_EVT['FP'], 'FP EVT (' + str(maxevtcount) + ')', maxevtcount]
        ]:
            ax = axs[x][y]
            bitmap = [[0] * len(self.steps) for _ in range(0, len(self.steps))]
            for subject in P:
                for i, s in enumerate(self.steps):
                    bitmap[int(len(P[subject][i]) * len(self.steps) / (maxcount + 1))][i] += 1
            for i in range(0, len(bitmap)):
                for j in range(1, len(bitmap[i])):
                    rj = len(bitmap[i]) - j
                    bitmap[rj - 1][i] += bitmap[rj][i]
            bitmap = numpy.array(bitmap)
            bitmapmax = numpy.max(bitmap)
            if bitmapmax > 0.0:
                bitmap = bitmap / bitmapmax
            im = ax.imshow(bitmap, cmap='gray')
            ax.set_xticks([0, len(self.steps) - 1])
            ax.set_xticklabels(['0.0', '1.0'])
            ax.set_yticks([0, len(self.steps) - 1])
            ax.set_yticklabels(['0.0', '1.0'])
            ax.set_title(title)
        fig.tight_layout()
        matplotlib.pyplot.savefig('.'.join([output, 'seqevt', 'png']), dpi=300)
        matplotlib.pyplot.close(fig)
