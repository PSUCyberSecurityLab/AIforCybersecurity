import argparse
import csv
import datetime
import gc
import gzip
import json
import os
import matplotlib
import numpy

matplotlib.use('Agg')
import utils.util as util
import data.unswnb15.key
from models.dablog import Codebook, DablogSequence, Dablog
from models.deeplog import Codebook, Deeplog, DeeplogSequence

attackers = ['175.45.176.0', '175.45.176.1', '175.45.176.2', '175.45.176.3']
victims = ['149.171.126.10', '149.171.126.11', '149.171.126.12', '149.171.126.13', '149.171.126.14',
           '149.171.126.15', '149.171.126.16', '149.171.126.17', '149.171.126.18', '149.171.126.19']
normals = ['149.171.126.0', '149.171.126.1', '149.171.126.2', '149.171.126.3', '149.171.126.4',
           '149.171.126.5', '149.171.126.6', '149.171.126.7', '149.171.126.8', '149.171.126.9']  # majorly inbound
senders = ['59.166.0.0', '59.166.0.1', '59.166.0.2', '59.166.0.3', '59.166.0.4',
           '59.166.0.5', '59.166.0.6', '59.166.0.7', '59.166.0.8', '59.166.0.9']  # majorly outbound


def main():
    timestamp = datetime.datetime.now()
    operation = {
        'key': [LogKeyStatistics, ],
        'ad': [AnomalyDetection, ],
        'detect': [AnomalyDetection, AnomalyReport, ],
        'auc': [AnomalyReport, ],
        'report': [AnomalyReport, ],
        'freq': [FrequencyModel, ],
    }
    parser = argparse.ArgumentParser(description='Experiment based on UNSW-NB15 dataset')
    parser.add_argument('operation', help='Specify an operation (' + ', '.join(operation.keys()) + ')')
    # Anomaly Detection
    parser.add_argument('-i', '--input', help='Specify a input file or directory that contains data (e.g., mydata/)',
                        default='datasets/unsw-nb15/')
    parser.add_argument('-o', '--output', help='Specify an output file to write to',
                        default=timestamp.strftime('%Y_%m_%d_%H_%M_%S'))
    parser.add_argument('-m', '--model', help='Specify a deep learning model', default='dablog')
    parser.add_argument('--train-file', help='Specify a filename for training', default='flow_from_this.csv')
    parser.add_argument('--test-file', help='Specify a filename for testing', default='flow_to_this.csv')
    parser.add_argument('--seqlen', help='Hyperparameter', type=int, default=10)
    parser.add_argument('--epochs', help='Hyperparameter', type=int, default=32)
    parser.add_argument('--logkeys', help='Logkeys Scenarios (see scenario index in key.py)', type=int, default=0)
    parser.add_argument('--use-gpu', help='Specify GPU usage (-1: disable, default: all gpus)', type=int, nargs='+')
    parser.add_argument('--use-mimick', help='Toggle Mimick Embedding', action='store_true')
    parser.add_argument('--stats-step', help='Step for stats (default=0.01)', type=float, default=0.01)
    parser.add_argument('--window-size', help='Specify the number of minutes for each sequence', type=int, default=15)
    parser.add_argument('--key-divisor', help='Speicfy the number of key divisor', type=int, default=100)
    parser.add_argument('--label-size', help='Specify the number of attack category for anomlies', type=int, default=1)
    parser.add_argument('--check-sequences', help='Specify a file of subjects to check')
    # Parser
    args = parser.parse_args()
    for op in operation[args.operation]:
        gc.collect()
        op(args)


def AnomalyDetection(args):
    global trainset, Model
    if args.model in ['deeplog']:
        Deeplog.SetupGPU(args.use_gpu)
        Model = Deeplog
        trainset = DeeplogSequence(DeeplogSequence.Config(seqlen=args.seqlen, verbose=True))
    if args.model in ['autoencoder', 'dablog']:
        Dablog.SetupGPU(args.use_gpu)
        Model = Dablog
        trainset = DablogSequence(DablogSequence.Config(seqlen=args.seqlen, verbose=True))

    # define how to read sequences from file
    def readSequences(ip, filename):
        sequence = {}
        label = {}
        with open(os.path.join(args.input, ip, filename), 'rt') as fin:
            csvfin = csv.reader(fin, delimiter=',')
            for line in csvfin:
                datetime = data.unswnb15.key.getDateTimeFromLine(line)
                srcip = line[data.unswnb15.key.srcip]
                dstip = line[data.unswnb15.key.dstip]
                dstport = line[data.unswnb15.key.dsport]
                svcport = dstport
                # try: svcport = line [unswnb15.key.proto] if int (dstport) > 1024 else dstport
                # except: pass
                # subject = '-'.join (['from', srcip, 'to', dstip, ':', dstport])
                # subject = '-'.join (['from', srcip, 'to', dstip, ':', svcport])
                subject = '-'.join(['from', srcip, 'to', dstip, 'on', str(datetime.day), str(datetime.hour),
                                    str(datetime.minute // args.window_size)])
                # subject = '-'.join (['from', srcip, 'to', dstip, 'on', str (datetime.day), str (datetime.hour), str (datetime.minute // 60)])
                # subject = '-'.join (['from', srcip, 'to', dstip, 'on', str (datetime.day), str (datetime.hour), str (datetime.minute // 30)])
                # subject = '-'.join (['from', srcip, 'to', dstip, 'on', str (datetime.day), str (datetime.hour), str (datetime.minute // 15)])
                slabel = data.unswnb15.key.getLabelFromLine(line)
                skeystr = data.unswnb15.key.getKeyFromLine(line, args.logkeys, args.key_divisor)
                if subject not in sequence:
                    sequence[subject] = list()
                sequence[subject].append(skeystr)
                if subject not in label:
                    label[subject] = list()
                label[subject].append(slabel)
        ret = []
        for subject in sequence:
            notNoneLabels = [l for l in label[subject] if l is not None]
            ret.append((sequence[subject], subject, ','.join(sorted(set(notNoneLabels))), len(notNoneLabels)))
        return ret

    # build training dataset
    trainips = senders
    bar = util.ProgressBar('Read Normal Sequences for Training', len(trainips))
    for idx, ip in enumerate(trainips):
        bar.update(idx + 1)
        try:
            for sequence, description, label, attacks in readSequences(ip, args.train_file):
                trainset.append(sequence)
        except KeyboardInterrupt as e:
            print(e)
            exit(0)
        except Exception as e:
            print(e)
            continue
    bar.finish()
    # build universal codebook
    codebook = Codebook(trainset.types)
    trainset.codebook = codebook
    print('Trainset Stats \n' + json.dumps(trainset.getStats(), indent=4))
    # build model
    os.makedirs(args.output, exist_ok=True)
    config = Model.Config(epochs=args.epochs, use_mimick_embedding=False, filepath=args.output,
                          rank_threshold=args.stats_step, distance_threshold=1, verbose=True)
    model = Model(config)
    try:
        model.pretrain(trainset)
        model.train(trainset)
    except KeyboardInterrupt:
        pass
    # build testing dataset from normals
    testset = DablogSequence(trainset.config)
    del trainset
    gc.collect()
    bar = util.ProgressBar('Read Normal Blocks for Testing', len(normals))
    testNormals = 0
    subject_lengths = {}
    for idx, ip in enumerate(normals):
        bar.update(idx + 1)
        try:
            for index, (sequence, description, label, attacks) in enumerate(readSequences(ip, args.test_file)):
                subject = '-'.join([ip, str(index)])
                subject_lengths[subject] = len(sequence)
                testset.append(sequence, seqinfo={'subject': subject, 'label': label})
                testNormals += 1
        except KeyboardInterrupt as e:
            print(e)
            exit(0)
        except Exception as e:
            print(e)
            continue
    bar.finish()

    # build testing dataset from victims
    abnormal_labels = {}
    testAbnormals = 0
    bar = util.ProgressBar('Read Abnormal Blocks for Testing', len(victims))
    for idx, ip in enumerate(victims):
        bar.update(idx + 1)
        try:
            for index, (sequence, description, label, attacks) in enumerate(readSequences(ip, args.test_file)):
                subject = '-'.join([ip, str(index)])
                subject_lengths[subject] = len(sequence)
                testset.append(sequence, seqinfo={'subject': subject, 'label': label})
                if len(label) > 0 and attacks > 0:
                    testAbnormals += 1
                    abnormal_labels[subject] = description + ',' + label + ',' + '/'.join(
                        [str(attacks), str(len(sequence))])
                else:
                    testNormals += 1
        except KeyboardInterrupt as e:
            print(e)
            exit(0)
        except Exception as e:
            print(e)
            continue
    bar.finish()
    testset.codebook = codebook
    print('Testset Stats \n' + json.dumps(testset.getStats(), indent=4))
    # write labels to directory                     
    with open(os.path.join(args.output, 'labels'), 'wt') as fout:
        fout.write(json.dumps(abnormal_labels, indent=4) + '\n')
    # write lengths to directory 
    with open(os.path.join(args.output, 'subjectlengths'), 'wt') as fout:
        fout.write(json.dumps(subject_lengths, indent=4) + '\n')
    # test
    print('Testing')
    n_miss, misses = model.test(testset)
    print('Prototype Misses:', n_miss)
    del testset
    gc.collect()
    # write metadata to directory
    with open(os.path.join(args.output, 'metadata'), 'wt') as fout:
        fout.write(json.dumps({
            'model': args.model, 'misses': n_miss,
            'seqlen': args.seqlen, 'epochs': args.epochs, 'logkeys': args.logkeys,
            'test-normal': testNormals, 'test-abnormal': testAbnormals, }) + '\n')
        fout.write(json.dumps(codebook.decode) + '\n')
        # write embed but ndarray and float32 are not json-serializable
        embed = model.getEmbeddings(codebook)
        for k in embed:
            embed[k] = list(embed[k])
            for i in range(0, len(embed[k])): embed[k][i] = float(embed[k][i])
        fout.write(json.dumps(embed) + '\n')
        # reset input file for later analysis
    args.input = args.output


def AnomalyReport(args):
    # parse input and metadata
    global model
    fin = open(os.path.join(args.input, 'metadata'), 'r')
    metadata = json.loads(fin.readline())
    decode = json.loads(fin.readline())
    embeds = json.loads(fin.readline())
    fin.close()
    testNormals, testAbnormals = metadata['test-normal'], metadata['test-abnormal']
    # if metadata['model'] in ['deeplog']: from models.deeplog import Model as model
    if metadata['model'] in ['autoencoder', 'dablog']:
        model = Dablog
    # parse each line
    subjectranks = {}
    bar = util.ProgressBar('Rank-Based Area Under Curve Analysis', metadata['misses'])
    try:
        fin = open(os.path.join(args.input, 'predict'), 'rt')
    except:
        fin = gzip.open(os.path.join(args.input, 'predict.gz'), 'rt')
    for progress in range(0, metadata['misses']):
        bar.update(progress)
        line = fin.readline()
        if len(line) == 0:
            break
        misses = model.split(json.loads(line))
        for miss in misses:
            X, P, I, idx, x, y = miss['X'], miss['P'], miss['I'], miss['idx'], miss['x'], miss['y']
            obj = {
                'label': I['label'],
                'evt_idx': idx,
                'seq_idx': I['idx'],
                'subject': I['subject'],
                'len': I['len'], }
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
    with open(os.path.join(args.input, 'subjectranks'), 'wt') as fout:
        for subject in subjectranks: fout.write(json.dumps(subjectranks[subject]) + '\n')
    # read labels and write absolute FNs
    abnormal_labels = json.load(open(os.path.join(args.input, 'labels'), 'rt'))
    with open(os.path.join(args.input, 'false_negatives'), 'wt') as fout:
        for subject in abnormal_labels:
            if subject not in subjectranks:
                fout.write(subject + ', ' + abnormal_labels[subject] + '\n')
    # sort subject ranks
    ranks, rankstat = {}, {}
    for subject in subjectranks:
        rank = subjectranks[subject]['rank']
        label = subjectranks[subject]['label']
        normality = 'Abnormal' if subject in abnormal_labels else 'Normal'
        if rank not in ranks: ranks[rank] = {}
        if normality not in ranks[rank]:
            ranks[rank][normality] = 0
        if normality not in rankstat:
            rankstat[normality] = []
        ranks[rank][normality] += 1
        rankstat[normality].append(subjectranks[subject]['rank'])
    ranks = sorted(list(ranks.items()), reverse=True, key=lambda obj: obj[0])  # sort by rank
    # rankstats 
    with open(os.path.join(args.input, 'rankstats'), 'wt') as fout:
        obj = {}
        for label in rankstat:
            obj[label + '-mean'] = numpy.mean(rankstat[label])
            obj[label + '-std'] = numpy.std(rankstat[label])
            obj[label + '-max'] = max(rankstat[label])
            obj[label + '-min'] = min(rankstat[label])
        fout.write(json.dumps(obj) + '\n')
    # derive rank-based metric
    with open(os.path.join(args.input, 'rank.metric'), 'wt') as fout:
        steps = [i for i in numpy.arange(args.stats_step, 1 + args.stats_step, args.stats_step)]
        TP, FP = [0] * len(steps), [0] * len(steps)
        for rank in ranks:
            index = int(rank[0] * len(steps))
            rank = rank[1]  # becomes {'normal': count, 'abnormal': count}
            for normality in rank:
                for i in range(0, index):
                    if normality in ['Abnormal', 'abnormal']:
                        TP[i] += rank[normality]
                    else:
                        FP[i] += rank[normality]
        TN = [testNormals - fp for fp in FP]
        FN = [testAbnormals - tp for tp in TP]
        fout.write(json.dumps({'TP': TP}))  # print ('TP', TP)
        fout.write(json.dumps({'FP': FP}))  # print ('FP', FP)
        fout.write(json.dumps({'TN': TN}))  # print ('TN', TN)
        fout.write(json.dumps({'FN': FN}))  # print ('FN', FN)
    # derive AUC ROC ( tpr/fpr = tp/(tp+fn) / fp/(fp+tn))
    auroc = 0.0
    with open(os.path.join(args.input, 'auroc'), 'wt') as fout:
        tpr = [0.0]
        fpr = [0.0]
        tp, fp = 0, 0
        for rank in ranks:
            rank = rank[1]  # becomes {'normal': count, 'abnormal': count}
            for normality in rank:
                if normality in ['Abnormal', 'abnormal']:
                    tp += rank[normality]
                else:
                    fp += rank[normality]
            fn = testAbnormals - tp
            tn = testNormals - fp
            tpr.append((tp / (tp + fn)) if tp > 0.0 else 0.0)
            fpr.append((fp / (fp + tn)) if fp > 0.0 else 0.0)
            addarea = (tpr[-1] + tpr[-2]) * (fpr[-1] - fpr[-2]) / 2
            auroc += addarea
        auroc += tpr[-1] * (1.0 - fpr[-1])
        obj = {'auroc': auroc, 'tpr': tpr, 'fpr': fpr}
        fout.write(json.dumps(obj) + '\n')
    # derive AUC PRC 
    auprc = 0.0
    auf1c = 0.0
    with open(os.path.join(args.input, 'auprc'), 'wt') as fout:
        precision = [0.0]
        recall = [0.0]
        f1 = [0.0]
        tp, fp = 0, 0
        for rank in ranks:
            rank = rank[1]  # becomes {'normal': count, 'abnormal': count}
            for normality in rank:
                if normality in ['Abnormal', 'abnormal']:
                    tp += rank[normality]
                else:
                    fp += rank[normality]
            fn = testAbnormals - tp
            tn = testNormals - fp
            precision.append((tp / (tp + fp)) if tp > 0.0 else 0.0)
            recall.append((tp / (tp + fn)) if tp > 0.0 else 0.0)
            f1.append(((2 * precision[-1] * recall[-1]) / (precision[-1] + recall[-1])) if precision[-1] + recall[
                -1] > 0.0 else 0.0)
            # addarea = (precision [-1] + precision [-2]) * (recall [-1] - recall [-2]) / 2 # this won't work due to sparseness
            addarea = precision[-1] * (recall[-1] - recall[-2])
            auprc += addarea
        auprc += 1.0 * (1.0 - recall[-1])
        auf1c = sum(f1) / len(f1)
        obj = {'auprc': auprc, 'auf1c': auf1c, 'f1': f1, 'precision': precision, 'recall': recall}
        fout.write(json.dumps(obj) + '\n')
    # write auc results
    with open(os.path.join(args.input, 'auc'), 'wt') as fout:
        obj = {'auroc': auroc, 'auprc': auprc, 'auf1c': auf1c}
        fout.write(json.dumps(obj) + '\n')


def LogKeyStatistics(args):
    def getKeyset(iplist):
        ret = {}
        for ip in iplist:
            with open(os.path.join(args.input, ip, 'flows.csv'), 'rt') as fin:
                csvfin = csv.reader(fin, delimiter=',')
                for line in csvfin:
                    # linestr = unswnb15.key.getDirectionFromLine (line, ip) + ',' + unswnb15.key.getKeyFromLine (line, args.logkeys, args.key_divisor)
                    linestr = data.unswnb15.key.getKeyFromLine(line, args.logkeys, args.key_divisor)
                    ret.setdefault(linestr, 0)
                    ret[linestr] += 1
        return ret

    k_attackers, k_victims, k_normals, k_senders = getKeyset(attackers), getKeyset(victims), getKeyset(
        normals), getKeyset(senders)
    s_attackers, s_victims, s_normals, s_senders = set(k_attackers.keys()), set(k_victims.keys()), set(
        k_normals.keys()), set(k_senders.keys())

    print('# of Logkeys from All:', len(s_attackers | s_victims | s_normals | s_senders))
    print('# of Logkeys from Attackers:', len(s_attackers))
    print('# of Logkeys from Victims:', len(s_victims))
    print('# of Logkeys from Senders:', len(s_senders))
    print('# of Logkeys from Normal Receivers:', len(s_normals))
    print('# of LogKeys that are Abnormal:', len(s_attackers | s_victims))
    print('Intersection (abnormal & normal):', len((s_attackers | s_victims) & (s_senders | s_normals)))
    logkeys = {}
    for party in k_attackers, k_victims, k_normals, k_senders:
        for key in party:
            logkeys.setdefault(key, 0)
            logkeys[key] += party[key]
    summention = sum(logkeys.values())
    logkeys = sorted(logkeys.items(), reverse=True, key=lambda item: item[1])
    accumulate = 0
    for index, (key, occurrence) in enumerate(logkeys):
        accumulate += occurrence
        print('{:04d}'.format(index + 1), '{:05.02f}'.format(accumulate * 100 / summention), occurrence,
              '{:05.02f}'.format(occurrence * 100 / summention), key)


def FrequencyModel(args):
    # define how to read sequences from file
    def readSequences(ip, filename):
        sequence = {}
        label = {}
        with open(os.path.join(args.input, ip, filename), 'rt') as fin:
            csvfin = csv.reader(fin, delimiter=',')
            for line in csvfin:
                datetime = data.unswnb15.key.getDateTimeFromLine(line)
                srcip = line[data.unswnb15.key.srcip]
                dstip = line[data.unswnb15.key.dstip]
                dstport = line[data.unswnb15.key.dsport]
                svcport = dstport
                # try: svcport = line [unswnb15.key.proto] if int (dstport) > 1024 else dstport
                # except: pass
                # subject = '-'.join (['from', srcip, 'to', dstip, ':', dstport])
                # subject = '-'.join (['from', srcip, 'to', dstip, ':', svcport])
                subject = '-'.join(['from', srcip, 'to', dstip, 'on', str(datetime.day), str(datetime.hour),
                                    str(datetime.minute // args.window_size)])
                # subject = '-'.join (['from', srcip, 'to', dstip, 'on', str (datetime.day), str (datetime.hour), str (datetime.minute // 60)])
                # subject = '-'.join (['from', srcip, 'to', dstip, 'on', str (datetime.day), str (datetime.hour), str (datetime.minute // 30)])
                # subject = '-'.join (['from', srcip, 'to', dstip, 'on', str (datetime.day), str (datetime.hour), str (datetime.minute // 15)])
                slabel = data.unswnb15.key.getLabelFromLine(line)
                skeystr = data.unswnb15.key.getKeyFromLine(line, args.logkeys, args.key_divisor)
                if subject not in sequence:
                    sequence[subject] = list()
                sequence[subject].append(skeystr)
                if subject not in label:
                    label[subject] = list()
                label[subject].append(slabel)
        ret = []
        for subject in sequence:
            notNoneLabels = [l for l in label[subject] if l is not None]
            ret.append((
                sequence[subject],
                ','.join(sorted(set(notNoneLabels))),
                len(notNoneLabels)))
        return ret

    # sequence frequencies
    subject_sequences = {}
    training_sequences = {}
    normal_sequences = {}
    abnormal_sequences = {}
    normal_keys = {}
    abnormal_keys = {}
    training_keys = {}
    # build training dataset
    global trainset
    if args.model in ['deeplog']:
        trainset = DeeplogSequence(DeeplogSequence.Config(seqlen=args.seqlen, verbose=True))
    if args.model in ['autoencoder', 'dablog']:
        trainset = DablogSequence(DablogSequence.Config(seqlen=args.seqlen, verbose=True))
    trainips = senders
    bar = util.ProgressBar('Read Normal Sequences for Training', len(trainips))
    for idx, ip in enumerate(trainips):
        bar.update(idx + 1)
        try:
            for index, (sequence, label, attacks) in enumerate(readSequences(ip, args.train_file)):
                for i in range(0, max(1, len(sequence) - args.seqlen)):
                    keystring = '\n'.join([seq for seq in sequence[i: i + args.seqlen]])
                    normal_sequences.setdefault(keystring, 0)
                    normal_sequences[keystring] += 1
                    training_sequences.setdefault(keystring, 0)
                    training_sequences[keystring] += 1
                for key in sequence:
                    normal_keys.setdefault(key, 0)
                    normal_keys[key] += 1
                    training_keys.setdefault(key, 0)
                    training_keys[key] += 1
                trainset.append(sequence)
        except KeyboardInterrupt as e:
            print(e)
            exit(0)
        except Exception as e:
            print(e)
            continue
    bar.finish()

    # build universal codebook
    codebook = Codebook(trainset.types)
    trainset.codebook = codebook
    # key frequencies
    frequencies = {}
    for keyset in [trainset.types]:
        for key in keyset:
            frequencies.setdefault(key, 0)
            frequencies[key] += keyset[key]
    sumension = sum([frequencies[k] for k in frequencies])
    for key in frequencies:
        frequencies[key] /= sumension
    # metrics
    TP = [0] * 100
    FP = [0] * 100
    # build testing dataset from normals
    bar = util.ProgressBar('Read Normal Blocks for Testing', len(normals))
    testNormals = 0
    for idx, ip in enumerate(normals):
        bar.update(idx + 1)
        try:
            for index, (sequence, label, attacks) in enumerate(readSequences(ip, args.test_file)):
                subject = '-'.join([ip, str(index)])
                subject_sequences[subject] = sequence
                testNormals += 1
                leastFrequency = 1.0
                for key in sequence:
                    if key in frequencies:
                        leastFrequency = min(leastFrequency, frequencies[key])
                    else:
                        leastFrequency = 0.0
                        break
                reportingIndex = int(100 - leastFrequency * 100)
                for i in range(0, reportingIndex):
                    FP[i] += 1
                for i in range(0, max(1, len(sequence) - args.seqlen)):
                    keystring = '\n'.join([seq for seq in sequence[i: i + args.seqlen]])
                    normal_sequences.setdefault(keystring, 0)
                    normal_sequences[keystring] += 1
                for key in sequence:
                    normal_keys.setdefault(key, 0)
                    normal_keys[key] += 1
        except KeyboardInterrupt as e:
            print(e)
            exit(0)
        except Exception as e:
            print(e)
            continue
    bar.finish()

    # build testing dataset from victims
    testAbnormals = 0
    bar = util.ProgressBar('Read Abnormal Blocks for Testing', len(victims))
    for idx, ip in enumerate(victims):
        bar.update(idx + 1)
        try:
            for index, (sequence, label, attacks) in enumerate(readSequences(ip, args.test_file)):
                subject = '-'.join([ip, str(index)])
                subject_sequences[subject] = sequence
                leastFrequency = 1.0
                for key in sequence:
                    if key in frequencies:
                        leastFrequency = min(leastFrequency, frequencies[key])
                    else:
                        leastFrequency = 0.0
                        break
                    abnormal_keys.setdefault(key, 0)
                    abnormal_keys[key] += 1
                reportingIndex = int(100 - leastFrequency * 100)
                if len(label) > 0 and attacks > 0:
                    testAbnormals += 1
                    for i in range(0, reportingIndex):
                        TP[i] += 1
                    for i in range(0, max(1, len(sequence) - args.seqlen)):
                        keystring = '\n'.join([seq for seq in sequence[i: i + args.seqlen]])
                        abnormal_sequences.setdefault(keystring, 0)
                        abnormal_sequences[keystring] += 1
                else:
                    testNormals += 1
                    for i in range(0, reportingIndex):
                        FP[i] += 1
                    for i in range(0, max(1, len(sequence) - args.seqlen)):
                        keystring = '\n'.join([seq for seq in sequence[i: i + args.seqlen]])
                        normal_sequences.setdefault(keystring, 0)
                        normal_sequences[keystring] += 1
        except KeyboardInterrupt as e:
            print(e)
            exit(0)
        except Exception as e:
            print(e)
            continue

    bar.finish()
    TN = [testNormals - FP[i] for i in range(0, 100)]
    FN = [testAbnormals - TP[i] for i in range(0, 100)]
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, 'keys'), 'wt') as fout:
        fout.write(json.dumps({
            'trains': training_keys,
            'normals': normal_keys,
            'abnormals': abnormal_keys,
        }) + '\n')

    with open(os.path.join(args.output, 'rank.metric'), 'wt') as fout:
        fout.write(json.dumps({'TP': TP}))  # print ('TP', TP)
        fout.write(json.dumps({'FP': FP}))  # print ('FP', FP)
        fout.write(json.dumps({'TN': TN}))  # print ('TN', TN)
        fout.write(json.dumps({'FN': FN}))  # print ('FN', FN)

    with open(os.path.join(args.output, 'sequences'), 'wt') as fout:
        normalset = set(normal_sequences)
        abnormalset = set(abnormal_sequences)
        intersectionset = normalset & abnormalset
        normalset = normalset - intersectionset
        abnormalset = abnormalset - intersectionset
        intersection = {}
        for subject in intersectionset:
            intersection.setdefault(subject, 0)
            intersection[subject] += normal_sequences[subject]
            intersection[subject] += abnormal_sequences[subject]
        fout.write(json.dumps({
            'trains': len(training_sequences),
            'normals': len(normalset),
            'abnormals': len(abnormalset),
            'intersection': len(intersection)}) + '\n')
        fout.write(json.dumps({
            'trains': training_sequences,
            'normals': normal_sequences,
            'abnormals': abnormal_sequences}) + '\n')
        print('avg. occ. of trains:', numpy.mean([training_sequences[seq] for seq in training_sequences]))
        print('avg. occ. of normals:', numpy.mean([normal_sequences[seq] for seq in normalset]))
        print('avg. occ. of abnormals:', numpy.mean([abnormal_sequences[seq] for seq in abnormalset]))

    with open(os.path.join(args.output, 'checks'), 'wt') as fout:
        if args.check_sequences is not None:
            check_sequences = open(args.check_sequences, 'rt').read().split()
            checks = {}
            checked = set()
            subseq_count = 0
            normal_count = 0
            abnormal_count = 0
            train_count = 0
            print('checking', ','.join(check_sequences))
            for subject in check_sequences:
                if subject not in subject_sequences:
                    checks[subject] = 'not found'
                    continue
                sequence = subject_sequences[subject]
                leastFrequency = 1.0
                for key in sequence:
                    if key in frequencies:
                        leastFrequency = min(leastFrequency, frequencies[key])
                    else:
                        leastFrequency = 0.0
                        break
                checks[subject] = [leastFrequency]
                for head in range(0, max(1, len(sequence) - args.seqlen)):
                    subseq_count += 1
                    keystring = '\n'.join([seq for seq in sequence[head: head + args.seqlen]])
                    if keystring in training_sequences:
                        checks[subject].append('train')
                    elif keystring in normalset:
                        checks[subject].append('normal')
                    elif keystring in abnormalset:
                        checks[subject].append('abnormal')
                    if keystring not in checked:
                        if keystring in normal_sequences:
                            normal_count += normal_sequences[keystring]
                        if keystring in abnormal_sequences:
                            abnormal_count += abnormal_sequences[keystring]
                        if keystring in training_sequences:
                            train_count += training_sequences[keystring]
                        checked.add(keystring)

            fout.write(json.dumps(checks, indent=4))
            print('uniq subseq count:', len(checked))
            print('subsequence count:', subseq_count)
            print('train subsequence count:', train_count)
            print('normal subsequence count:', normal_count)
            print('abnormal subsequence count:', abnormal_count)


if __name__ == '__main__':
    main()
