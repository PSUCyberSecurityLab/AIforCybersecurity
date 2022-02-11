import argparse
import datetime
import gc
import gzip
import json
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy
import os
import re
import sys
import time
import hypertools
import random

from models.deeplog import Codebook, DeeplogSequence, Deeplog
from models.dablog import Codebook, DablogSequence, Dablog
import models.util as modelUtil
import utils.util as util


def main():
    timestamp = datetime.datetime.now()
    operation = {
        'detect': [SequentialAnomalyDetection, SyntacticDistanceAndStrategyAnalysis, RankBasedAreaUnderCurveAnalysis],
        'seqsa': [SyntacticDistanceAndStrategyAnalysis, ],
        'auroc': [RankBasedAreaUnderCurveAnalysis, ],
        'parse': [ParseFromOriginalData, ],
        'lkass': [ListKeyAndSequenceStatistics, ],
        'statkey': [PrintKeyAndStatistics, ],
        'sdplt': [PlotSyntacticDistance, ],
        'report': [ReportAnomalies, ],
    }
    parser = argparse.ArgumentParser(description='Experiment based on SOSP 2009 dataset')
    parser.add_argument('operation', help='Specify an operation (' + ', '.join(operation.keys()) + ')')
    # Anomaly Detection
    parser.add_argument('-l', '--label', help='Specify a label file (e.g., mylabel.sample_1024.txt',
                        default='label.txt')
    parser.add_argument('-i', '--input', help='Specify a input file or directory that contains data (e.g., mydata/)',
                        default='data')
    parser.add_argument('-o', '--output', help='Specify an output file to write to',
                        default=timestamp.strftime('%Y_%m_%d_%H_%M_%S'))
    parser.add_argument('-m', '--model', help='Specify a deep learning model', default='dablog')
    parser.add_argument('--seqlen', help='Hyperparameter', type=int, default=10)
    parser.add_argument('--epochs', help='Hyperparameter', type=int, default=16)
    parser.add_argument('--logkeys', help='Logkeys Scenarios (0=32, 1=78, 2=104, 3=1129)', type=int, default=2)
    parser.add_argument('--train-normal', help='Number of normal blocks in training set', type=int, default=200000)
    parser.add_argument('--train-abnormal', help='Number of abnormal blocks in training set', type=int, default=0)
    parser.add_argument('--test-normal', help='Number of normal blocks in testing set', type=int, default=200000)
    parser.add_argument('--test-abnormal', help='Number of abnormal blocks in testing set', type=int, default=16838)
    parser.add_argument('--save-model', help='Specify a filepath to which to save the model')
    parser.add_argument('--load-model', help='Specify a filepath from which to load the model')
    parser.add_argument('--sdasa-strategy', help='Specify a strategy [rank|dist|complete|all]', default='all')
    parser.add_argument('--sdasa-distance', help='Specify a distance threshold', type=float, default=0.0)
    parser.add_argument('--use-gpu', help='Specify GPU usage (-1: disable, default: all gpus)', type=int, nargs='+')
    parser.add_argument('--use-mimick', help='Toggle Mimick Embedding', action='store_true')
    parser.add_argument('--stats-step', help='Step for stats (default=0.01)', type=float, default=0.01)
    parser.add_argument('--check-adjacent', help='Check adjacent sequences', type=int, default=0)
    # Parser
    args = parser.parse_args()
    for op in operation[args.operation]:
        gc.collect()
        op(args)


def readSequence(filepath, args):
    """
    :function: Define how to read sequence from a file
    :param filepath: path of raw data sample
    :param args: arguments from command line
    :return: processed sequence
    """
    seq = []
    for line in open(filepath, 'r'):
        obj = json.loads(line)
        evt = obj['event']
        task = evt['task']

        if args.logkeys != 0:
            if 'filepath_type' in evt:
                task += ', filepath_type=' + evt['filepath_type']
            if task in [
                'INFO dfs.DataNode$PacketResponder: Received block BLK of size SIZE from IP',
                'INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: DST is added to BLK size SIZE']:
                if 'size' in evt:
                    task += ', size=' + str(int(evt['size']) // 10000000) + '0 MB'
        if args.logkeys != 0:
            precision = args.logkeys
            # parse IPs
            if 'src_ip' in evt and 'dst_ip' in evt:
                if evt['src_ip'] == evt['dst_ip']:
                    task += ', traffic=localhost'
                elif '.'.join(evt['src_ip'].split('.')[: precision]) == '.'.join(evt['dst_ip'].split('.')[: precision]):
                    task += ', traffic=subnet'
                else:
                    task += ', traffic=intranet'
            elif 'src_ip' in evt:
                src_ip = '.'.join(evt['src_ip'].split('.')[: precision])
                if precision > 2:
                    src_ip = src_ip[: src_ip.rfind('.') + 2]
                task += ', src_ip=' + src_ip
            elif 'dst_ip' in evt:
                dst_ip = '.'.join(evt['dst_ip'].split('.')[: precision])
                if precision > 2:
                    dst_ip = dst_ip[: dst_ip.rfind('.') + 2]
                task += ', dst_ip=' + dst_ip
        seq.append(task)
    return seq


def SequentialAnomalyDetection(args):
    global trainset, Model
    if args.model in ['deeplog']:
        trainset = DeeplogSequence(DeeplogSequence.Config(seqlen=args.seqlen, verbose=True))
        Model = Deeplog
    if args.model in ['autoencoder', 'dablog']:
        trainset = DablogSequence(DablogSequence.Config(seqlen=args.seqlen, verbose=True))
        Model = Dablog

    Model.SetupGPU(args.use_gpu)
    # read labels
    normals, abnormals, count, bar = [], [], 0, util.ProgressBar('Read Anomaly Labels', 575139)
    with open(args.label, 'r', encoding="utf-8") as file:
        count = 0
        reader = csv.reader(file)
        for row in reader:
            label, block = row[1], row[0]
            count += 1
            bar.update(count)

            if label == 'Normal':
                normals.append(block)
            else:
                abnormals.append(block)
    
    bar.finish()
    # build training dataset
    args.train_normal, args.train_abnormal = min(args.train_normal, len(normals)), min(args.train_abnormal, len(abnormals))
    if args.train_normal > 0:
        bar = util.ProgressBar('Read Normal Blocks For Training', args.train_normal)
        for i in range(0, args.train_normal):
            bar.update(i)
            try:
                trainset.append(readSequence(os.path.join(args.input, normals[i] + '.log'), args),
                                seqinfo={'blk': normals[i], 'label': 'Normal'})
            except KeyboardInterrupt:
                exit(0)
        bar.finish()

    if args.train_abnormal > 0:
        bar = util.ProgressBar('Read Abnormal Blocks For Training', args.train_abnormal)
        for i in range(0, args.train_abnormal):
            bar.update(i)
            try:
                trainset.append(readSequence(os.path.join(args.input, abnormals[i] + '.log'), args),
                                seqinfo={'blk': abnormals[i], 'label': 'Abnormal'})
            except KeyboardInterrupt:
                exit(0)
        bar.finish()
    # build universal codebook
    if args.load_model:
        codebook = Codebook.load(args.load_model)
    else:
        codebook = Codebook(trainset.types)
    trainset.codebook = codebook
    print('Trainset Stats \n' + trainset.getStats())
    # build model
    os.makedirs(args.output, exist_ok=True)
    config = Model.Config(epochs=args.epochs, use_mimick_embedding=args.use_mimick, filepath=args.output,
                          rank_threshold=args.stats_step, distance_threshold=args.stats_step, verbose=True)
    model = Model(config)
    try:
        model.pretrain(trainset)
        model.train(trainset)
    except KeyboardInterrupt:
        pass
    # test 
    testset = DablogSequence(trainset.config)
    del (trainset)
    gc.collect()
    args.test_normal, args.test_abnormal = min(args.test_normal, len(normals)), min(args.test_abnormal, len(abnormals))
    if args.test_normal > 0:
        bar = util.ProgressBar('Read Normal Blocks For Testing', args.test_normal)
        for index in range(0, args.test_normal):
            i = len(normals) - args.test_normal + index
            bar.update(index)
            try:
                testset.append(readSequence(os.path.join(args.input, normals[i] + '.log'), args),
                               seqinfo={'blk': normals[i], 'label': 'Normal'})
            except KeyboardInterrupt:
                exit(0)
            except:
                continue
        bar.finish()
    if args.test_abnormal > 0:
        bar = util.ProgressBar('Read Abnormal Blocks For Testing', args.test_abnormal)
        for index in range(0, args.test_abnormal):
            i = len(abnormals) - args.test_abnormal + index
            bar.update(index)
            try:
                testset.append(readSequence(os.path.join(args.input, abnormals[i] + '.log'), args),
                               seqinfo={'blk': abnormals[i], 'label': 'Abnormal'})
            except KeyboardInterrupt:
                exit(0)
            except:
                continue
        bar.finish()
    testset.codebook = codebook
    print('Testset Stats \n' + json.dumps(testset.getStats(), indent=4))
    n_miss, misses = model.test(testset)
    codebook = testset.codebook
    print('Prototype Misses:', n_miss)
    del testset
    gc.collect()
    # write to directory
    myoutputfile = os.path.join(args.output, 'metadata')
    myoutput = open(myoutputfile, 'w')
    myoutput.write(json.dumps({
        'model': args.model, 'misses': n_miss,
        'seqlen': args.seqlen, 'epochs': args.epochs, 'logkeys': args.logkeys,
        'train-normal': args.train_normal, 'train-abnormal': args.train_abnormal,
        'test-normal': args.test_normal, 'test-abnormal': args.test_abnormal, }) + '\n')
    myoutput.write(json.dumps(codebook.decode) + '\n')
    # embed: ndarray and float32 are not json-serializable
    embed = model.getEmbeddings(codebook)
    for k in embed:
        embed[k] = list(embed[k])
        for i in range(0, len(embed[k])):
            embed[k][i] = float(embed[k][i])
    myoutput.write(json.dumps(embed) + '\n')
    with open(os.path.join(args.output, 'embed'), 'w') as fout:
        fout.write(json.dumps(embed) + '\n')
    # bar = util.ProgressBar ('Write Misses Into File ' + myoutputfile, n_miss)
    # for i, miss in enumerate (misses):
    #     bar.update (i)
    #     myoutput.write (json.dumps (miss) + '\n')
    # bar.finish ()
    myoutput.close()
    # reset input file for later analysis
    args.input = args.output


def ReportAnomalies(args):
    # default reporting threshold 
    global datapoint, key, X, Y, data, labels, model
    report_threshold = 0.075
    # parse input and metadata
    fin = open(os.path.join(args.input, 'metadata'), 'r')
    metadata = json.loads(fin.readline())
    decode = json.loads(fin.readline())
    encode = {decode[i]: int(i) for i in decode}
    embeds = json.loads(fin.readline())
    embed = {i: embeds[decode[str(i)]] for i in range(0, len(decode))}

    dij = modelUtil.EmbeddingDistance(embed)
    if metadata['model'] in ['deeplog']:
        model = Deeplog
    if metadata['model'] in ['autoencoder', 'dablog']:
        model = Dablog
    fin.close()
    normals, abnormals = metadata['test-normal'], metadata['test-abnormal']
    TP, FP = set(), set()
    normalDP, abnormalDP = {}, {}
    # parse each line
    bar = util.ProgressBar('Report Anomalies', metadata['misses'])
    try:
        fin = open(os.path.join(args.input, 'predict'), 'r')
    except:
        fin = gzip.open(os.path.join(args.input, 'predict.gz'), 'r')
    fout = open(os.path.join(args.input, 'reconstruct'), 'w')
    for progress in range(0, metadata['misses']):
        bar.update(progress)
        line = fin.readline()
        if len(line) == 0:
            break
        line = json.loads(line)
        mispredict = False
        # parse misses
        misses = model.split(line)
        for miss in misses:
            X, P, I, idx, x, y = miss['X'], miss['P'], miss['I'], miss['idx'], miss['x'], miss['y']
            sprob = sorted(P, reverse=True)
            rank = sprob.index(P[x]) / len(sprob)
            # stratey == 'rank'
            if rank <= report_threshold:
                continue
            else:
                mispredict = True
        if not mispredict:
            continue
        # reconstruct
        if metadata['model'] in ['deeplog']:
            X, Y, I = line['X'], int(numpy.argmax(line['P'])), line['I']
            key, datapoint = str(Y), embed[Y]
            I['ndiff'] = 1
            I['dist'] = dij[X[-1]][Y]
        if metadata['model'] in ['autoencoder', 'dablog']:
            X, Y, I = line['X'], numpy.argmax(line['P'], 1).tolist(), line['I']
            key, datapoint = str(Y), [embed[y] for y in Y]
            I['ndiff'] = sum(X[i] != Y[0 - i - 1] for i in range(0, len(X)))
            I['dist'] = numpy.linalg.norm([dij[X[i]][Y[0 - i - 1]] for i in range(0, len(X))])
            # if I ['ndiff'] < 3: continue
        # update metric
        blk = line['I']['blk']
        if line['I']['label'] in ['Abnormal', 'abnormal']:
            TP.add(blk)
            abnormalDP[key] = datapoint
        else:  # is normal
            FP.add(blk)
            normalDP[key] = datapoint
        # write to file 
        info = line['I']
        info = ' '.join([info['blk'], 'len', str(info['len']), 'is', info['label'], '@', str(info['idx']), 'diff',
                         str(info['ndiff']), 'dist', str(info['dist'])])
        fout.write(info + ': ' + json.dumps({'X': X, 'Y': Y}) + '\n')

    bar.finish()
    fin.close()
    fout.close()
    # metric
    TP, FP = len(TP), len(FP)
    precision = TP / (TP + FP)
    recall = TP / abnormals
    f1 = 2 * precision * recall / (precision + recall)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F-1 Score:', f1)
    # hyperplot embeddings of sequences of events
    if True:
        normalKeys, abnormalKeys = list(normalDP.keys()), list(abnormalDP.keys())
        random.shuffle(normalKeys)
        random.shuffle(abnormalKeys)
        keys = normalKeys[: 256] + abnormalKeys[: 256]
        hue, datapoints = [], {}
        for key in keys:
            if key in normalDP:
                datapoints[key] = normalDP[key]
            else:
                datapoints[key] = abnormalDP[key]
        for key in keys:
            if key in normalDP and key in abnormalDP:
                hue.append('Uncertain')
            elif key in normalDP and key not in abnormalDP:
                hue.append('Normal')
            elif key not in normalDP and key in abnormalDP:
                hue.append('Abnormal')
        for tag in ['Uncertain', 'Normal', 'Abnormal']:
            print(tag + ':', sum([hue[i] == tag for i in range(0, len(hue))]))
        if metadata['model'] in ['deeplog']:
            labels = [decode[key] for key in keys]
            data = [datapoints[key] for key in keys]
        elif metadata['model'] in ['autoencoder', 'dablog']:
            # labels = [decode [key] for key in keys]
            labels = [0] * len(keys)
            ones = numpy.array([1.0] * len(embed[0]))
            zeros = numpy.array([0.0] * len(embed[0]))
            data = []
            for key in keys:
                vector = []
                for vec in datapoints[key]:
                    vector.append(numpy.linalg.norm(vec - zeros))
                data.append(vector)
        hypertools.plot(numpy.array(data), '.', labels=labels, hue=hue, explore=True, legend=True)


def RankBasedAreaUnderCurveAnalysis(args):
    # parse input and metadata
    global model
    fin = open(os.path.join(args.input, 'metadata'), 'r')
    metadata = json.loads(fin.readline())
    decode = json.loads(fin.readline())
    embeds = json.loads(fin.readline())
    fin.close()
    if metadata['model'] in ['deeplog']:
        model = Deeplog
    if metadata['model'] in ['autoencoder', 'dablog']:
        model = Dablog
    normals, abnormals = metadata['test-normal'], metadata['test-abnormal']
    # parse each line
    subjectranks = {}
    bar = util.ProgressBar('Rank-Based Area Under Curve Analysis', metadata['misses'])
    try:
        fin = open(os.path.join(args.input, 'predict'), 'r')
    except:
        fin = gzip.open(os.path.join(args.input, 'predict.gz'), 'r')
    for progress in range(0, metadata['misses']):
        bar.update(progress)
        line = fin.readline()
        if len(line) == 0:
            break
        misses = model.split(json.loads(line))
        for miss in misses:
            X, P, I, idx, x, y = miss['X'], miss['P'], miss['I'], miss['idx'], miss['x'], miss['y']
            obj = {'label': I['label'], 'evt_idx': idx, 'seq_idx': I['idx'], 'subject': I['blk'], 'len': I['len']}
            subject = obj['subject']
            sprob = sorted(P, reverse=True)
            rank = sprob.index(P[x]) / len(sprob)
            if subject not in subjectranks:
                subjectranks[subject] = {'subject': subject, 'label': obj['label'], 'rank': rank}
            else:
                subjectranks[subject]['rank'] = max(subjectranks[subject]['rank'], rank)
    fin.close()
    bar.finish()
    # write subject ranks
    with open(os.path.join(args.input, 'subjectranks'), 'w') as fout:
        for subject in subjectranks: fout.write(json.dumps(subjectranks[subject]) + '\n')
    # sort subject ranks
    ranks, rankstat, allTPs = {}, {}, set()
    for subject in subjectranks:
        rank = subjectranks[subject]['rank']
        label = subjectranks[subject]['label']
        if rank not in ranks:
            ranks[rank] = {}
        if label not in ranks[rank]:
            ranks[rank][label] = 0
        if label not in rankstat:
            rankstat[label] = []
        if label in ['Abnormal', 'abnormal']:
            allTPs.add(subject)
        ranks[rank][label] += 1
        rankstat[label].append(subjectranks[subject]['rank'])
    ranks = sorted(list(ranks.items()), reverse=True, key=lambda obj: obj[0])  # sort by rank
    # write absolute FNs
    with open(os.path.join(args.input, 'false_negatives'), 'w') as fout:
        abnormalsubjects = set()
        for line in open(args.label, 'r'):
            label, subject = line.split()
            if label == '1':
                abnormalsubjects.add(subject)
        fns = abnormalsubjects - allTPs
        for fn in fns:
            fout.write(fn + '\n')
    # rank stats
    with open(os.path.join(args.input, 'rankstats'), 'w') as fout:
        obj = {}
        for label in rankstat:
            obj[label + '-mean'] = numpy.mean(rankstat[label])
            obj[label + '-std'] = numpy.std(rankstat[label])
            obj[label + '-max'] = max(rankstat[label])
            obj[label + '-min'] = min(rankstat[label])
        fout.write(json.dumps(obj) + '\n')
    # derive rank-based metric
    with open(os.path.join(args.input, 'rank.metric'), 'w') as fout:
        steps = [i for i in numpy.arange(args.stats_step, 1 + args.stats_step, args.stats_step)]
        TP, FP = [0] * len(steps), [0] * len(steps)
        for rank in ranks:
            index = int(rank[0] * len(steps))
            rank = rank[1]  # becomes {'normal': count, 'abnormal': count}
            for label in rank:
                for i in range(0, index):
                    if label in ['Abnormal', 'abnormal']:
                        TP[i] += rank[label]
                    else:
                        FP[i] += rank[label]
        TN = [normals - fp for fp in FP]
        FN = [abnormals - tp for tp in TP]
        fout.write(json.dumps({'TP': TP}))  # print ('TP', TP)
        fout.write(json.dumps({'FP': FP}))  # print ('FP', FP)
        fout.write(json.dumps({'TN': TN}))  # print ('TN', TN)
        fout.write(json.dumps({'FN': FN}))  # print ('FN', FN)
    # derive AUC ( tpr/fpr = tp/(tp+fn) / fp/(fp+tn)) 
    with open(os.path.join(args.input, 'auroc'), 'w') as fout:
        tpr = [0.0]
        fpr = [0.0]
        auc = 0.0
        tp, fp = 0, 0
        for rank in ranks:
            rank = rank[1]  # becomes {'normal': count, 'abnormal': count}
            for label in rank:
                if label in ['Abnormal', 'abnormal']:
                    tp += rank[label]
                else:
                    fp += rank[label]
            fn = abnormals - tp
            tn = normals - fp
            tpr.append(tp / (tp + fn))
            fpr.append(fp / (fp + tn))
            addarea = (tpr[-1] + tpr[-2]) * (fpr[-1] - fpr[-2]) / 2
            auc += addarea
        auc += tpr[-1] * (1.0 - fpr[-1])
        obj = {'auc': auc, 'tpr': tpr, 'fpr': fpr}
        fout.write(json.dumps(obj) + '\n')


def SyntacticDistanceAndStrategyAnalysis(args):
    global dij, model
    strategy = args.sdasa_strategy
    gapdist = args.sdasa_distance
    if strategy == 'all':
        args.sdasa_strategy = 'rank'
        SyntacticDistanceAndStrategyAnalysis(args)
        args.sdasa_strategy = 'dist'
        SyntacticDistanceAndStrategyAnalysis(args)
        # args.sdasa_strategy = 'complete'; SyntacticDistanceAndStrategyAnalysis (args)
        args.sdasa_strategy = strategy
        return
    # parse input and metadata
    fin = open(os.path.join(args.input, 'metadata'), 'r')
    metadata = json.loads(fin.readline())
    decode = json.loads(fin.readline())
    encode = {decode[i]: int(i) for i in decode}
    embeds = json.loads(fin.readline())
    embed = {i: embeds[decode[str(i)]] for i in range(0, len(decode))}
    fin.close()
    if metadata['model'] in ['deeplog']:
        model = Deeplog
    if metadata['model'] in ['autoencoder', 'dablog']:
        model = Dablog
    normals, abnormals = metadata['test-normal'], metadata['test-abnormal']
    steps = [i for i in numpy.arange(args.stats_step, 1 + args.stats_step, args.stats_step)]
    # calculate pair wise distance and universe unit
    if strategy != 'rank':
        dij = modelUtil.EmbeddingDistance(embed)
        unknown = encode[modelUtil.Codebook.UNKNOWN]
    # parse each line
    positives = {}  # [list () for _ in range (0, len (steps))]
    bar = util.ProgressBar('Syntatic Distance and Strategy Analysis (' + strategy + ')', metadata['misses'])
    try:
        fin = open(os.path.join(args.input, 'predict'), 'r')
    except:
        fin = gzip.open(os.path.join(args.input, 'predict.gz'), 'r')
    for progress in range(0, metadata['misses']):
        bar.update(progress)
        line = fin.readline()
        if len(line) == 0:
            break
        misses = model.split(json.loads(line))
        for miss in misses:
            X, P, I, idx, x, y = miss['X'], miss['P'], miss['I'], miss['idx'], miss['x'], miss['y']
            obj = {'label': I['label'], 'evt_idx': idx, 'seq_idx': I['idx'], 'subject': I['blk'], 'len': I['len']}
            subject = obj['subject']
            if subject not in positives:
                positives[subject] = [list() for _ in range(0, len(steps))]
            sprob = sorted(P, reverse=True)
            rank = sprob.index(P[x]) / len(sprob)
            # strategy: rank
            if strategy == 'rank':
                for i, s in enumerate(steps):
                    if rank <= s:
                        continue
                    else:
                        positives[subject][i].append(obj)
            # strategy : distance or complete
            else:  # distance analysis
                # sort by prob, but if probs are the same then compare distance
                spred = sorted([i for i in range(0, len(P))], reverse=True, key=lambda i: P[i] * 1000000 - dij[x][i])
                sdist = [dij[x][i] for i in spred]
                j, pred, reported = 0, spred[0], False
                jrank = sprob.index(P[pred]) / len(sprob)
                for i, s in enumerate(steps):
                    if strategy == 'complete' and rank <= s:
                        continue  # check rank and continue if hit
                    while jrank <= s and j < len(list(spred)):
                        pred = spred[j]
                        jrank = sprob.index(P[pred]) / len(sprob)
                        j += 1
                    dist = min(sdist[0: j + 1])
                    if dist > s + gapdist:
                        positives[subject][i].append(obj)
    # add positive cases to stats
    stats = modelUtil.Statistics(steps, metadata['seqlen'] + 1, metadata['test-normal'], metadata['test-abnormal'])
    for subject in positives:
        for i in range(0, len(positives[subject])):
            # extra check on adjecent sequences if applicable
            seqs = set([obj['seq_idx'] for obj in positives[subject][i]])
            # if len (seqs) > 1: print (seqs)
            for obj in positives[subject][i]:
                idx = obj['seq_idx']
                if obj['len'] < args.seqlen:
                    stats.add(i, obj)
                elif not any([idx + adj not in seqs for adj in range(0, args.check_adjacent + 1)]):
                    stats.add(i, obj)
    fin.close()
    bar.finish()
    stats.output(os.path.join(args.output, '.'.join([strategy, str(gapdist)])))


def ListKeyAndSequenceStatistics(args):
    # read labels
    global sequence
    progress, bar = 0, util.ProgressBar('List Sequences and Labels', 575139)
    seqset, evtset = {}, {}
    # parse each line from data
    for line in open(args.label, 'r'):
        progress += 1
        bar.update(progress)
        label, block = line.split()
        label = 'Normal' if int(label) == 0 else 'Abnormal'
        try:
            sequence = readSequence(os.path.join(args.input, block + '.log'), args)
        except KeyboardInterrupt:
            exit(0)
        except:
            print(block, 'is broken')
            continue
        for evt in sequence:  # parse event
            if evt not in evtset:
                evtset[evt] = {'occurrence': 0, 'Normal': 0, 'Abnormal': 0}
            evtset[evt]['occurrence'] += 1
            evtset[evt][label] += 1
        for i in range(0, max(1, len(sequence) - args.seqlen)):  # parse sequence
            seq = sequence[i: min(i + args.seqlen, len(sequence))]
            if len(seq) < args.seqlen:
                seq = [Codebook.PAD] * (args.seqlen - len(seq)) + seq
            seqstr = json.dumps(seq)
            if seqstr not in seqset:
                seqset[seqstr] = {'Normal': 0, 'Abnormal': 0}
            seqset[seqstr][label] += 1
    bar.finish()
    # write to key file
    count = sum([evtset[evt]['occurrence'] for evt in evtset])
    sort = sorted(list(evtset.keys()), reverse=True, key=lambda k: evtset[k]['occurrence'])
    with open('.'.join([args.output, str(args.logkeys), str(len(sort)), 'keys']), 'w') as fout:
        for key in sort:
            obj = evtset[key]
            occ = evtset[key]['occurrence']
            fout.write(json.dumps({'key': key, 'occurrence': occ, 'percentage': occ / count,
                                   'normals': obj['Normal'], 'abnormals': obj['Abnormal']}) + '\n')
    # write to seq file
    count = sum([seqset[seq]['Normal'] + seqset[seq]['Abnormal'] for seq in seqset])
    with open('.'.join([args.output, str(args.logkeys), str(args.seqlen), 'seqs']), 'w') as fout:
        for seqstr in seqset:
            seq = seqset[seqstr]
            normals = seq['Normal']
            abnormals = seq['Abnormal']
            occurrence = normals + abnormals
            obj = {'sequence': json.loads(seqstr),
                   'occurrence': occurrence, 'percentage': occurrence / count,
                   'normals': normals, 'abnormals': abnormals, }
            fout.write(json.dumps(obj) + '\n')


def PrintKeyAndStatistics(args):
    global sequence
    keys = []
    for line in open(args.output, 'r'):
        keys.append(json.loads(line))
    for i in range(0, len(keys)):
        print(i + 1, end=', ')
        print(keys[i]['percentage'], end=', ')
        print(keys[i]['normals'], end=', ')
        print(keys[i]['abnormals'], end=', ')
        print(sum([obj['percentage'] for obj in keys[0: i + 1]]), end=', ')
        print(keys[i]['key'])
    ktoi = {keys[i]['key']: i for i in range(0, len(keys))}
    TP, FP = [0] * 100, [0] * 100
    # TP, FP = [0] * len (keys), [0] * len (keys)
    progress, bar = 0, util.ProgressBar('List Sequences and Labels', 575139)
    for line in open(args.label, 'r'):
        progress += 1
        bar.update(progress)
        label, block = line.split()
        label = 'Normal' if int(label) == 0 else 'Abnormal'
        try:
            sequence = readSequence(os.path.join(args.input, block + '.log'), args)
        except KeyboardInterrupt:
            exit(0)
        except:
            print(block, 'is broken')
            continue
        maxindex = max([ktoi[evt] for evt in sequence])
        maxpercentile = int((maxindex * len(TP)) / len(keys))
        for i in range(0, maxpercentile):
            # for i in range (0, maxindex):
            if label == 'Normal':
                FP[i] += 1
            else:
                TP[i] += 1
    bar.finish()
    TN, FN = [0] * 100, [0] * 100
    for i in range(0, len(TP)):
        print(i + 1, end=', ')
        tp, fp = TP[i], FP[i]
        tn, fn = 575139 - fp, 16838 - tp
        TN[i], FN[i] = tn, fn
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        accuracy = (tp + tn) / (tp + fn + fp + tn)
        print('f1:' + str(f1), end=', ')
        print('acc:' + str(accuracy), end=', ')
        print('fpr:' + str(fpr))
    with open(args.output + '__rank.0.0.metric', 'w') as fout:
        fout.write(json.dumps({'TP': TP}))
        fout.write(json.dumps({'FP': FP}))
        fout.write(json.dumps({'TN': TN}))
        fout.write(json.dumps({'FN': FN}))


def PlotSyntacticDistance(args):
    embed = json.load(open(args.input, 'r'))
    tags = ['<pad>', '<sequence>', '</sequence>', '<unknown>']
    if args.output == 'embed':
        data = [embed[key] for key in embed]
        # labels = [key if key in tags else None  for key in embed]
        labels = [key for key in embed]
        # hue = [tags.index (key) + 1 if key in tags  else 0 for key in embed]
        # hypertools.plot (numpy.array (data), '.', hue=numpy.array (hue), labels=labels)
        # hypertools.plot (numpy.array (data), '.', labels=labels, n_clusters=2)
        hypertools.plot(numpy.array(data), '.', labels=labels, n_clusters=10, explore=True)
    elif args.output == 'seq':
        # redefine unknown embeddings
        dists, unknown = {}, numpy.array([1.0] * len(embed[tags[-1]]))
        # unknown = numpy.mean ([embed [evt] for evt in embed])
        normals, abnormals, uncertains = [], [], []
        data, labels = [], []
        # calculate pair wise distance and universe unit
        for key in embed:
            if key in embed:
                vec = numpy.array(embed[key])
            else:
                vec = unknown
            dists[key] = numpy.linalg.norm(vec - unknown)
        # parse each anomalous sequences
        for line in open(args.label, 'r'):
            obj = json.loads(line)
            if obj['normal_ratio'] == 1.0:
                normals.append(obj)
            elif obj['abnormal_ratio'] == 1.0:
                abnormals.append(obj)
            else:
                uncertains.append(obj)
        # generate plotting data and plot
        random.shuffle(normals)
        random.shuffle(abnormals)
        random.shuffle(uncertains)
        for hue, dataset in [['Abnormal', abnormals], ['Normals', normals], ['Uncertain', uncertains]]:
            for i in range(0, min(4096, len(dataset))):
                obj = dataset[i]
                seq = obj['sequence']
                vec = [dists[evt] if evt in dists else 0.0 for evt in seq]
                data.append(vec)
                labels.append(hue)
        labels[0] = None  # workaround of hypertools keeps showing numbers instead of texts in its legend
        hypertools.plot(numpy.array(data), '.', hue=labels, legend=True)


def ParseFromOriginalData(args):
    def parseEvent(splits):
        event = {}

        if 'ERROR dfs.DataNode$DataXceiver:' == ' '.join(splits[: 2]):
            event['task'] = 'ERROR dfs.DataNode$DataXceiver: DST:DataXceiver: EXCEPTION: block BLK ' + ' '.join(
                splits[6:])
            event['block_id'] = splits[5]
            event['exception'] = splits[3]
            event['dst_ip'], event['dst_port'], _, _ = splits[2].replace('/', '').split(':')
        elif 'INFO dfs.DataBlockScanner: Verification succeeded for' == ' '.join(splits[: 5]):
            event['task'] = 'INFO dfs.DataBlockScanner: Verification succeeded for BLK'
            event['block_id'] = splits[-1]
        elif 'INFO dfs.DataNode$BlockReceiver: Changing block file offset of block' == ' '.join(splits[: 8]):
            event['task'] = 'INFO dfs.DataNode$BlockReceiver: Changing block file offset of block BLK from PARAM1 to PARAM2 meta file offset to PARAM3'
            event['block_id'] = splits[8]
            event['param'] = [splits[10], splits[12], splits[-1]]
        elif 'INFO dfs.DataNode$BlockReceiver: Exception in receiveBlock for block' == ' '.join(splits[: 7]):
            event['task'] = 'INFO dfs.DataNode$BlockReceiver: Exception in receiveBlock for block BLK EXCEPTION'
            event['block_id'] = splits[7]
            event['exception'] = ' '.join(splits[8:])
        elif 'INFO dfs.DataNode$BlockReceiver:' == ' '.join(splits[: 2]) and 'writing block' == ' '.join(splits[3: 5]):
            event['task'] = 'INFO dfs.DataNode$BlockReceiver: SRC:Exception writing block BLK to mirror DST'
            event['block_id'] = splits[5]
            event['src_ip'], event['src_port'], _ = splits[2].replace('/', '').split(':')
            event['dst_ip'], event['dst_port'] = splits[-1].replace('/', '').split(':')
        elif 'INFO dfs.DataNode$BlockReceiver: Receiving empty packet for block' == ' '.join(splits[: 7]):
            event['task'] = 'INFO dfs.DataNode$BlockReceiver: Receiving empty packet for block BLK'
            event['block_id'] = splits[-1]
        elif 'INFO dfs.DataNode$DataTransfer:' == ' '.join(splits[: 2]):
            event['task'] = 'INFO dfs.DataNode$DataTransfer: SRC:Transmitted block BLK to DST'
            event['block_id'] = splits[4]
            event['src_ip'], event['src_port'], _ = splits[2].replace('/', '').split(':')
            event['dst_ip'], event['dst_port'] = splits[-1].replace('/', '').split(':')
        elif 'INFO dfs.DataNode$DataXceiver:' == ' '.join(splits[: 2]) and 'Served' == splits[3]:
            event['task'] = 'INFO dfs.DataNode$DataXceiver: SRC Served block BLK to DST'
            event['block_id'] = splits[6]
            event['src_ip'], event['src_port'] = splits[2].replace('/', '').split(':')
            event['dst_ip'] = splits[-1].replace('/', '')
        elif 'INFO dfs.DataNode$DataXceiver: Received block' == ' '.join(splits[: 4]):
            event['task'] = 'INFO dfs.DataNode$DataXceiver: Received block BLK src: SRC dest: DST of size SIZE'
            event['block_id'] = splits[4]
            event['src_ip'], event['src_port'] = splits[4 + 2].replace('/', '').split(':')
            event['dst_ip'], event['dst_port'] = splits[4 + 4].replace('/', '').split(':')
            event['size'] = splits[4 + 7]
        elif 'INFO dfs.DataNode$DataXceiver: Receiving block' == ' '.join(splits[: 4]):
            event['task'] = 'INFO dfs.DataNode$DataXceiver: Receiving block BLK from SRC to DST'
            event['block_id'] = splits[4]
            event['src_ip'], event['src_port'] = splits[4 + 2].replace('/', '').split(':')
            event['dst_ip'], event['dst_port'] = splits[4 + 4].replace('/', '').split(':')
        elif 'INFO dfs.DataNode$DataXceiver: writeBlock' == ' '.join(splits[: 3]) and 'exception' in splits:
            event['task'] = 'INFO dfs.DataNode$DataXceiver: writeBlock BLK received exception EXCEPTION'
            event['block_id'] = splits[3]
            event['exception'] = ' '.join(splits[6:])
        elif 'INFO dfs.DataNode:' == ' '.join(splits[: 2]) and 'Starting thread to transfer block' == ' '.join(
                splits[3: 8]):
            event['task'] = 'INFO dfs.DataNode: SRC Starting thread to transfer block BLK to DST(s)'
            event['block_id'] = splits[8]
            event['dst_list'] = splits[10:]
        elif 'INFO dfs.DataNode$PacketResponder: PacketResponder' == ' '.join(splits[: 3]) and 'Exception' in splits:
            event['task'] = 'INFO dfs.DataNode$PacketResponder: PacketResponder BLK CODE Exception EXCEPTION'
            event['block_id'] = splits[3]
            event['code'] = splits[4]
            event['exception'] = ' '.join(splits[6:])
        elif 'INFO dfs.DataNode$PacketResponder: PacketResponder' == ' '.join(
                splits[: 3]) and 'Exception' not in splits:
            event['task'] = 'INFO dfs.DataNode$PacketResponder: PacketResponder CODE for block BLK ' + splits[-1]
            event['block_id'] = splits[6]
            event['code'] = splits[3]
        elif 'INFO dfs.DataNode$PacketResponder: Received block' == ' '.join(splits[: 4]):
            event['task'] = 'INFO dfs.DataNode$PacketResponder: Received block BLK of size SIZE from IP'
            event['block_id'] = splits[4]
            event['size'] = splits[7]
            event['src_ip'] = splits[-1].replace('/', '')
        elif 'INFO dfs.FSDataset: Deleting block' == ' '.join(splits[: 4]):
            event['task'] = 'INFO dfs.FSDataset: Deleting block BLK file FILEPATH'
            event['block_id'] = splits[4]
            event['filepath'] = splits[4 + 2]
        elif 'INFO dfs.FSDataset: Reopen Block' == ' '.join(splits[: -1]):
            event['task'] = 'INFO dfs.FSDataset: Reopen Block BLK'
            event['block_id'] = splits[-1]
        elif 'INFO dfs.FSNamesystem: BLOCK* ask' == ' '.join(splits[: 4]) and 'to delete' == ' '.join(splits[5: 7]):
            event['task'] = 'INFO dfs.FSNamesystem: BLOCK* ask DST to delete BLK (s)'
            event['block_id'] = splits[7] if len(splits) == 8 else splits[7:]
            event['dst_ip'], event['dst_port'] = splits[4].replace('/', '').split(':')
        elif 'INFO dfs.FSNamesystem: BLOCK* ask' == ' '.join(splits[: 4]) and 'to replicate' == ' '.join(splits[5: 7]):
            event['task'] = 'INFO dfs.FSNamesystem: BLOCK* ask DST to replicate BLK to datanode(s) SRC'
            event['block_id'] = splits[7]
            event['src_ip'], event['src_port'] = splits[-1].replace('/', '').split(':')
            event['dst_ip'], event['dst_port'] = splits[4].replace('/', '').split(':')
        elif 'INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: addStoredBlock request received for' == ' '.join(
                splits[: 8]):
            event['task'] = 'INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: addStoredBlock request received but does not belong to any file'
            event['block_id'] = splits[8]
            event['dst_ip'], event['dst_port'] = splits[8 + 2].replace('/', '').split(':')
            event['size'] = splits[8 + 4]
        elif 'INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated:' == ' '.join(splits[: 6]):
            event['task'] = 'INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: DST is added to BLK size SIZE'
            event['block_id'] = splits[6 + 4]
            event['dst_ip'], event['dst_port'] = splits[6].replace('/', '').split(':')
            event['size'] = splits[6 + 6]
        elif 'INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock:' == ' '.join(splits[: 4]):
            event['task'] = 'INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock FILEPATH BLK'
            event['block_id'] = splits[-1]
            event['filepath'] = ' '.join(splits[4: -1])
        elif 'INFO dfs.FSNamesystem: BLOCK* NameSystem.delete:' == ' '.join(splits[: 4]):
            event['task'] = 'INFO dfs.FSNamesystem: BLOCK* NameSystem.delete BLK on DST'
            event['block_id'] = splits[4]
            event['dst_ip'], event['dst_port'] = splits[4 + 6].replace('/', '').split(':')
        elif 'INFO dfs.FSNamesystem: BLOCK* Removing block' == ' '.join(splits[: 5]):
            event['task'] = 'INFO dfs.FSNamesystem: BLOCK* Removing block BLK from neededReplications as it does not belong to any file'
            event['block_id'] = splits[5]
        elif 'WARN dfs.DataBlockScanner: Adding an already existing block' == ' '.join(splits[: -1]):
            event['task'] = 'WARN dfs.DataBlockScanner: Adding an already existing block BLK'
            event['block_id'] = splits[-1]
        elif 'WARN dfs.DataNode$DataTransfer:' == ' '.join(splits[: 2]):
            event['task'] = 'WARN dfs.DataNode$DataTransfer: SRC:Failed to transfer BLK to DST got EXCEPTION'
            event['block_id'] = splits[5]
            event['src_ip'], event['src_port'], _ = splits[2].replace('/', '').split(':')
            event['dst_ip'], event['dst_port'] = splits[7].replace('/', '').split(':')
            event['exception'] = ' '.join(splits[9:])
        elif 'WARN dfs.DataNode$DataXceiver:' == ' '.join(splits[: 2]):
            event['task'] = 'WARN dfs.DataNode$DataXceiver: SRC:Got exception while serving BLK to DST'
            event['block_id'] = splits[6]
            event['src_ip'], event['src_port'], _ = splits[2].replace('/', '').split(':')
            event['dst_ip'], event['dst_port'] = splits[8].replace('/', '').split(':')
        elif 'WARN dfs.FSDataset: Unexpected error trying to delete block' == ' '.join(splits[: 8]):
            event['task'] = 'WARN dfs.FSDataset: Unexpected error trying to delete block BLK BlockInfo not found in volumeMap.'
            event['block_id'] = splits[8][: -1]
        elif 'WARN dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: Redundant addStoredBlock request received for' == ' '.join(
                splits[: 9]):
            event['task'] = 'WARN dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: Redundant addStoredBlock request received for BLK on DST size SIZE'
            event['block_id'] = splits[9]
            event['dst_ip'], event['dst_port'] = splits[9 + 2].replace('/', '').split(':')
            event['size'] = splits[9 + 4]
        elif 'WARN dfs.PendingReplicationBlocks$PendingReplicationMonitor: PendingReplicationMonitor timed out block' == ' '.join(
                splits[: -1]):
            event['task'] = 'WARN dfs.PendingReplicationBlocks$PendingReplicationMonitor: PendingReplicationMonitor timed out block BLK'
            event['block_id'] = splits[-1]
        # check if any event is not handled
        if len(event) == 0:
            sys.stderr.write(' '.join(['Event not handled:'] + splits))
        # replace exception/filepath with regular expression
        if 'exception' in event:
            event['exception_type'] = re.sub('[\-0-9]+', 'NUM', event['exception'])
        if 'filepath' in event:
            event['filepath_type'] = re.sub('[\-0-9]+', 'NUM', event['filepath'])
        return event

    # output
    output = open(args.output + 'json.log', 'w')
    for line in open(args.input, 'r'):
        # <Date> <Time> <Pid> <Level> <Component>: <Content>
        splits = line.split()
        # timestamp range from 081109-203518 to 081111-111628
        timestamp = time.mktime(datetime.datetime.strptime(' '.join(splits[0: 2]), '%y%m%d %H%M%S').timetuple())
        # build object
        obj = {
            'raw': line,
            'timestamp': timestamp,
            'timestamp_str': str(datetime.datetime.fromtimestamp(timestamp)),
            'pid': splits[2],
            'event': parseEvent(splits[3:]), }
        output.write(json.dumps(obj) + '\n')
        args.n = args.n - 1
        if args.n == 0:
            break
    # close
    output.close()


if __name__ == '__main__':
    main()
