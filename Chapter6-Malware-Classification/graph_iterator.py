
from tf_utils import GNNGraph
import config
import numpy as np
import tensorflow as tf
import math
import random
def gnn_batching(graph_batch):
    node_features = [g.node_features for g in graph_batch]
    node_features = np.concatenate(node_features, 0)
    g_num_nodes = [g.num_nodes for g in graph_batch]
    graph_indexes = [[sum(g_num_nodes[0:i-1]), sum(g_num_nodes[0:i])] for i in range(1, len(g_num_nodes)+1)]
    batch_label = [g.label for g in graph_batch]
    total_node_degree = []
    indices = []
    for i, g in enumerate(graph_batch):
        total_node_degree.extend(g.degrees)
        start_pos = graph_indexes[i][0]
        for e in g.edges:
            node_from = start_pos + e[0]
            node_to = start_pos + e[1]
            indices.append([node_from, node_to])
            indices.append([node_to, node_from])
    total_node_num = len(total_node_degree)
    values = np.ones(len(indices), dtype=np.float32)
    indices = np.array(indices, dtype=np.int32)
    shape = np.array([total_node_num, total_node_num], dtype=np.int32)
    #ajacent = tf.sparse.SparseTensor(indices, values, shape)
    index_degree = [([i, i], 1.0/degree if degree >0 else 0) for i, degree in enumerate(total_node_degree)]
    index_degree = list(zip(*index_degree))
    #degree_inv = tf.sparse.SparseTensor(index_degree[0], index_degree[1], shape)
    #return ajacent, node_features, batch_label, degree_inv, graph_indexes
    return indices, values, shape, node_features, batch_label, index_degree[0], index_degree[1], graph_indexes

def gnn_train(model, g_list):
    total_labels = []  
    total_predicts = []
    random.shuffle(g_list)
    for pos in range(math.ceil(len(g_list)/config.batch_size)):
        batch_graphs = g_list[pos * config.batch_size: (pos+1) * config.batch_size]
        indices, values, shape, features, batch_label, index_degree_0, index_degree_1, graph_indexes = gnn_batching(batch_graphs)
        total_labels.extend(batch_label)
        with tf.GradientTape() as tape:
            logits = model(indices, values, shape, features, index_degree_0, index_degree_1, graph_indexes)
            pos_score = tf.nn.softmax(logits)
            predicts = np.argmax(pos_score,-1)
            loss = model.loss(batch_label,logits)
        total_predicts.extend(predicts)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.opt.apply_gradients(zip(gradients, model.trainable_variables))
        #print(tf.keras.metrics.Accuracy()(total_labels, total_predicts))

def gnn_test(model, g_list):
    total_labels = []  
    total_predicts = []
    random.shuffle(g_list)
    print(len(g_list))
    for pos in range(math.ceil(len(g_list)/config.batch_size)):
        batch_graphs = g_list[pos * config.batch_size: (pos+1) * config.batch_size]
        indices, values, shape, features, batch_label, index_degree_0, index_degree_1, graph_indexes = gnn_batching(batch_graphs)
        total_labels.extend(batch_label)
        logits = model(indices, values, shape, features, index_degree_0, index_degree_1, graph_indexes, False)
        predicts = np.argmax(logits,-1)
        total_predicts.extend(predicts)
    return total_labels, total_predicts
