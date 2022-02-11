import multiprocessing
import os
import gc
import argparse

import numpy as np
from sklearn.utils import shuffle
from iteration_utilities import deepflatten

np.random.seed(3)
FLAGS=None

def int2bin(num,padding=False,pad_to=8):
    lst=[]
    while num>1:
        lst.insert(0,num%2)
        num=num//2
    lst.insert(0,num)
    if padding:
        while len(lst)<pad_to:
            lst.insert(0,0)
    return lst


def prep(xList, window_size, window_step):
    X = []
    for i in range(len(xList)):
        line = xList[i]
        n = len(line)
        # segment
        for j in range(0, n-window_size+1, window_step):
            if j+window_size <= len(line):
                X.append(line[j:j+window_size])
            else:
                X.append(line[n-window_size:n])
    return np.array(X)


def worker(id,category,data,window_size):
    bin_data=[]
    N_data=len(data)
    for i in range(N_data):
        x=data[i]
        tmp=list(deepflatten(x))
        tmp=[int2bin(ele,padding=True) for ele in tmp]
        bin_data.append(tmp)
        if i%1000==0:
            print("Done "+category+" set: %d/%d"%(i,N_data))
    bin_data=np.array(bin_data,dtype=int)
    bin_data=bin_data.reshape((bin_data.shape[0],window_size,32,8))
    print(category+" set shape: "+str(bin_data.shape))
    np.save(os.path.join("data",'X_'+category+'-%.3d'%(id)+'.npy'),bin_data,allow_pickle=True)


def thread_function(k, window_size, window_step,benign_bytes, malicious_bytes):
    X_ben_tmp=prep(benign_bytes,window_size,window_step)
    X_mal_tmp=prep(malicious_bytes,window_size,window_step)

    full_set=set()

    # remove duplicates
    tmp = []
    for ele in X_mal_tmp:
        tmp.append(tuple(ele.reshape((window_size*32,)).tolist()))
    full_set = full_set | set(tmp)
    X_mal_tmp = np.array(list(set(tmp)),dtype=np.uint8).reshape((len(tmp),window_size,32))

    tmp = []
    for ele in X_ben_tmp:
        tmp.append(tuple(ele.reshape((window_size*32,)).tolist()))
    full_set = full_set | set(tmp)
    X_ben_tmp = np.array(list(set(tmp)),dtype=np.uint8).reshape((len(tmp),window_size,32))

    print("Original size of dataset for k = %d" % k)
    print("Malicious: %d" % len(X_mal_tmp))
    print("Benign: %d" % len(X_ben_tmp))
    print("Merged size: %d" % len(full_set))

    if len(full_set)!=(X_ben_tmp.shape[0]+X_mal_tmp.shape[0]):
        X_mal = []
        X_ben = []
        print("Double dipping exist (%d!=%d)! Remove double dipping..."%(len(full_set),(X_ben_tmp.shape[0]+X_mal_tmp.shape[0])))
        for item in full_set:
            item=np.array(item,dtype=np.uint8).reshape((window_size,32))
            if (item in X_ben_tmp) and (item in X_mal_tmp):
                continue
            elif item in X_ben_tmp:
                X_ben.append(item)
            elif item in X_mal_tmp:
                X_mal.append(item)
            if (len(X_ben) % 1000 == 0) or (len(X_mal) % 1000 == 0):
                print("############################")
                print(str(k) + " :")
                print("Malicious: %d" % (len(X_mal)))
                print("Benign: %d" % (len(X_ben)))
                print("Progress: %d/%d" %
                    (len(X_mal)+len(X_ben), len(full_set)))
        X_ben=np.array(X_ben)
        X_mal=np.array(X_mal)
    else:
        X_ben=X_ben_tmp
        X_mal=X_mal_tmp

    del(X_ben_tmp)
    del(X_mal_tmp)
    gc.collect()

    print("###########################")
    print("Double dipping removing finished.")
    print("Malicious: %d" % len(X_mal))
    print("Benign: %d" % len(X_ben))

    Y_mal = np.append(np.zeros(shape=(len(X_mal),1)),np.ones(shape=(len(X_mal),1)),1)
    Y_ben = np.append(np.ones(shape=(len(X_ben),1)),np.zeros(shape=(len(X_ben),1)),1)

    print("********************************")
    print("number of benign samples: %d" % len(X_ben))
    print("number of malicious samples: %d" %len(X_mal))

    ratio = len(X_ben)/len(X_mal)
    if ratio > 1.2:
        print("Too many benign data samples!")
        X_ben = shuffle(X_ben, random_state=1)[0:int(len(X_mal))]
        print("Downsample benign data samples to %d" %int(len(X_mal)))
        Y_ben = np.append(np.ones(shape=(len(X_mal),1)),np.zeros(shape=(len(X_mal),1)),1)
    elif (len(X_ben)/len(X_mal)) < 0.8:
        print("Too many malicious data samples!")
        X_mal = shuffle(X_mal, random_state=1)[0:int(len(X_ben))]
        print("Downsample malicious data samples to %d" %int(len(X_ben)))
        Y_mal = np.append(np.zeros(shape=(len(X_mal),1)),np.ones(shape=(len(X_mal),1)),1)

    X = np.append(X_mal,X_ben,0)
    Y = np.append(Y_mal,Y_ben,0)
    X, Y = shuffle(X, Y, random_state=1)

    del(X_mal)
    del(X_ben)
    gc.collect()

    n_data = len(X)
    X_train, X_test = X[:n_data//5*4], X[n_data//5*4:]
    Y_train, Y_test = Y[:n_data//5*4], Y[n_data//5*4:]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],X_test.shape[2], 1)

    del(X)
    del(Y)
    gc.collect()

    np.save(os.path.join("data",'Y_train-'+('%.3d'%(k))+'.npy'),Y_train,allow_pickle=True)
    np.save(os.path.join("data",'Y_test-'+('%.3d'%(k))+'.npy'),Y_test,allow_pickle=True)

    del(Y_train)
    del(Y_test)
    gc.collect()

    worker(k,"train",X_train,window_size)

    del(X_train)
    gc.collect()

    worker(k,"test",X_test,window_size)

    del(X_test)
    gc.collect()

    print("Data set exported done for k=%d"%k)

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '--cpu-num',
        type=int,
        default=1,
        help='Number of CPU cores for parallel execution. For machines with 16 GB or less RAM, 1 is recommended.'
    )
    FLAGS, unparsed =parser.parse_known_args()

    gc.enable()
    pool=multiprocessing.Pool(processes=FLAGS.cpu_num)

    benign_bytes=np.array([])
    benign_bytes=np.load(os.path.join('data','benign_bytes.npy'),allow_pickle=True)
    malicious_bytes=np.load(os.path.join('data','malicious_bytes.npy'),allow_pickle=True)

    window_sizes=[4,6,8,10,12]
    window_steps=[1,2,4,6,8]

    k=0
    job_args=[]
    for window_size in window_sizes:
        for window_step in window_steps:
            x=(k,window_size,window_step,benign_bytes,malicious_bytes)
            job_args.append(x)
            k+=1

    pool.starmap(thread_function,job_args)
    pool.close()
    pool.join()
