

import glob
import angr
import networkx as nx
from networkx.readwrite import json_graph
from queue import Queue
from threading import Thread
import os
import json
from shutil import copyfile, rmtree
import config

class GNNGraph(object):
    def __init__(self, g, label, node_features=None):
        self.num_nodes = g.number_of_nodes() # 节点数量
        self.label = label                   # 图标签
        self.node_features = node_features   # 节点特征 numpy array (node_num * feature_dim)
        self.degrees = list(dict(g.degree()).values())   # 节点的度列表
        self.edges = list(g.edges)           # 网络边列表

def get_CFG(f):
    try:
        proj = angr.Project(f, main_opts={'backend':'blob','arch':'i386'}, load_options={'auto_load_libs':False})
        cfg = proj.analyses.CFGEmulated()
    except:
        return
    block = proj.factory.block(proj.entry)
    blocks = {}
    node_features = []
    G = nx.DiGraph()
    idx = 0
    for n in cfg.graph.nodes():
        blocks[hex(n.addr)] = idx
        G.add_node(idx)
        idx += 1
        uint8 = []
        if n.block != None:
            block_instructions = n.block.capstone.__str__()
            vector = n.block.bytes.hex()
            b = bytearray.fromhex(vector)
            for i in range(len(b)):
                uint8.append(b[i])
        uint8.extend([config.max_features]*(config.feature_dim - len(uint8)))
        node_features.append(uint8)
    for k, v in cfg.graph.edges():
        G.add_edge(blocks[hex(k.addr)], blocks[hex(v.addr)])
    return GNNGraph(G, config.malware_label, node_features)

def extract_CFG(f):
    try:
        proj = angr.Project(f, main_opts={'backend':'blob','arch':'i386'}, load_options={'auto_load_libs':False})
    
        #main = proj.loader.main_object.get_symbol("main")
        #start_state = proj.factory.blank_state(addr=main.rebased_addr)
        #cfg = proj.analyses.CFGEmulated(fail_fast=True, starts=[main.rebased_addr], initial_state=start_state)
        cfg = proj.analyses.CFGEmulated()
    except:
        return
    block = proj.factory.block(proj.entry)
    blocks = []
    vertor = []
    G = nx.DiGraph()
    for n in cfg.graph.nodes():
        block_id = hex(n.addr)
        block_name = n.name
        block_size = n.size
        if n.block != None:
            block_instructions = n.block.capstone.__str__()
            vector = n.block.bytes.hex()
            G.add_node(block_id, vector = vector)
        else:
            block_instructions = None
            vector = None
            G.add_node(block_id, vector=vector)
        blocks.append((block_id, block_name, block_size, vector, block_instructions))
    edges = []
    for k, v in cfg.graph.edges():
        edges.append((hex(k.addr), hex(v.addr)))
        G.add_edge(hex(k.addr), hex(v.addr))
    if (len(edges)==0 or len(blocks)==0):
        print('error.......')
        return
    print(f)
    save(blocks, edges, G, f)

def save(blocks, edges, G, f):
    os.makedirs(os.path.dirname('data2/CFG_features/'+f.lstrip('data/binary/')), exist_ok=True)
    with open('data2/CFG_features/'+f.lstrip('data/binary/'), 'w') as outfile:
        json.dump({'blocks': blocks, 'edges': edges}, outfile)
    
    os.makedirs(os.path.dirname('data2/CFG_hex/'+f.lstrip('data/binary/')), exist_ok=True)
    with open('data2/CFG_hex/'+f.lstrip('data/binary/'), 'w') as outfile:
        json.dump(json_graph.node_link_data(G), outfile)
    

def run(file_queue, tmp):
    while not file_queue.empty():
        filename = file_queue.get()
        print(file_queue.qsize(), filename)
        extract_CFG(filename)
        file_queue.task_done()


def Graphdata(thread_number=7):
    files = glob.glob('data/binary/Virus/*') + glob.glob('data/binary/Benign/*')
    file_queue = Queue()
    for f in files:
        os.makedirs(os.path.dirname('data/CFG_features/'+f.lstrip('data/binary/')), exist_ok=True)
        os.makedirs(os.path.dirname('data/CFG_hex/'+f.lstrip('data/binary/')), exist_ok=True)
        file_queue.put(f)
            
    print(file_queue.qsize())
    for index in range(thread_number):
        thread = Thread(target=run, args=(file_queue, 0))
        thread.daemon = True
        thread.start()
    file_queue.join()
   
if __name__ == '__main__':
    Graphdata()
    #extract_CFG('data/binary/Virus/VirusShare_0a1d3e2731c78a4fab49f96952259a74')
