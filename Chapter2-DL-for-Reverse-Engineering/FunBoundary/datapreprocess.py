import os
import subprocess
import pickle
import numpy as np
import json
import random
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
# from utils import graph


path = "Data/x86_O1/"

class MultiThread(object):
    def __init__(self, args, path):
        self.args = args
        self.bb_size = 100
        self.path = path
        self.max_length = 100

    def batch_task(self, args):
        for arg in args:
            self.task(arg)
                
    def batch_task_bb(self, args):
        for arg in args:
            self.task_bb(arg)
        
    def task_bb(self, fname):
        print(fname)
        filepath = self.path+fname
        
        with open(filepath,'rb') as f:
            function_list = pickle.load(f)['function_list']
            out_file = open(self.path+'bbjsons/'+fname+'.json','w')
            for func in function_list:
                randomint = random.randint(5, 15)
                begin = func.get_first_n_ins(20-randomint)
                end = func.get_last_n_ins(randomint)
                if len(begin) + len(end) == 20:
                    sample = end+begin
                    label = [0 for _ in range(20)]
                    label[randomint] = 1
                    print(sample[0:randomint])
                    print(sample[randomint:])
                    print(randomint, sample[randomint])
                    # print([sample,label])
                    out_file.write(json.dumps([sample,label])+'\n')
        return 
        
    def process_bb(self, batch_size = 30):
        executor = ProcessPoolExecutor(max_workers=cpu_count())
        tasks = []
        for i in range(0, len(self.args), batch_size):
            batch = self.args[i : i+batch_size]
            tasks.append(executor.submit(self.batch_task_bb, batch)) 
    
        job_count = len(tasks)
        for future in as_completed(tasks):
            future.result() 
            job_count -= 1
            print("One Job Done, Remaining Job Count: %s " % (job_count))

import glob

def get_bb(path):
    filenames = glob.glob(path+'*.bb')
    fnames = [f.split('/')[-1] for f in filenames]
    excluded = []
    args = []
    for fn in fnames:
        if fn not in excluded:
            args.append(fn)
    # print(args)
    multithread = MultiThread(args, path)    
    multithread.process_bb()
    
get_bb(path)