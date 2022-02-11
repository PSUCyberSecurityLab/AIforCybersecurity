import config
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class GNN_MAL(tf.keras.Model):
    def __init__(self, num_classes):
        """Initializes the GNN model
        :param num_classes: The number of classes in the dataset.
        """
        super(GNN_MAL, self).__init__(name="GNN_MAL")
        self.num_classes = num_classes
        self.gcnn_dims = config.gcnn_dims
        self.k = config.top_k

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt = tf.optimizers.Adam(learning_rate=config.learning_rate, clipnorm=config.max_gradient_norm)       
        def __graph__():
            #self.embedding_layer = layers.Embedding(config.max_features+1, config.hidden_units)
            self.normalization_layer = layers.experimental.preprocessing.Rescaling(1./257)
            self.weight_matrix = []
            in_dim = config.feature_dim
            for i, dim in enumerate(self.gcnn_dims):
                out_dim = dim
                self.weight_matrix.append(tf.Variable(tf.random.truncated_normal([in_dim, out_dim])))
                in_dim = out_dim
            self.first_conv_layer = layers.Conv1D(16, sum(self.gcnn_dims), strides=sum(self.gcnn_dims), padding='same', activation='relu')
            self.first_pooling_layer = layers.MaxPooling1D()
            self.second_conv_layer = layers.Conv1D(32, 4, padding='same', activation='relu')
            self.flatten_layer = tf.keras.layers.Flatten()
            self.dense_layer = layers.Dense(16, activation=tf.nn.relu)
            # Dropout, to avoid overfitting
            self.dropout_layer = tf.keras.layers.Dropout(config.dropout)
            # Readout layer
            self.output_layer = layers.Dense(num_classes)

        __graph__()
    
    def call(self, indices, values, shape, features, index_degree_0, index_degree_1, graph_indexes, training=True):
        self.ajacent = tf.sparse.SparseTensor(tf.cast(indices, tf.int64), values, shape)
        self.features = self.normalization_layer(features)
        self.dgree_inv = tf.sparse.SparseTensor(index_degree_0, index_degree_1, shape)
        self.graph_indexes = tf.convert_to_tensor(graph_indexes)

        gcnns_outputs = self.gcnn_layers(self.features)
        emmbed = self.sort_pooling_layer(gcnns_outputs)
        cnn_1d = self.cnn1d_layers(emmbed)
        output = self.fc_layer(cnn_1d)
        dropout_layer = self.dropout_layer(output, training = training)
        logits = self.output_layer(dropout_layer)
        return logits

    def gcnn_layer(self, input_Z, W):
        AZ = tf.sparse.sparse_dense_matmul(self.ajacent, input_Z)  # AZ
        AZ = tf.add(AZ, input_Z)                                   # AZ+Z = (A+I)Z
        AZW = tf.matmul(AZ, W)                                     # (A+I)ZW
        DAZW = tf.sparse.sparse_dense_matmul(self.dgree_inv, AZW)  # D^-1AZW
        return tf.nn.tanh(DAZW)  # tanh 激活

    def gcnn_layers(self, Z):
        Z1_h = []
        for i in range(len(self.gcnn_dims)):
            Z = self.gcnn_layer(Z, self.weight_matrix[i])
            Z1_h.append(Z)
        Z1_h = tf.concat(Z1_h, 1) 
        return Z1_h

    def sort_pooling_layer(self, gcnn_out):
        def sort_a_graph(index_span):
            indices = tf.range(index_span[0], index_span[1])  # 获取单个图的节点特征索引
            graph_feature = tf.gather(gcnn_out, indices)      # 获取单个图的全部节点特征

            graph_size = index_span[1] - index_span[0]
            k = tf.cond(self.k > graph_size, lambda: graph_size, lambda: self.k)  # k与图size比较
            # 根据最后一列排序，返回前k个节点的特征作为图的表征
            top_k = tf.gather(graph_feature, tf.nn.top_k(graph_feature[:, -1], k=k).indices)

            # 若图size小于k，则补0行
            zeros = tf.zeros([self.k - k, sum(self.gcnn_dims)], dtype=tf.float32)
            top_k = tf.concat([top_k, zeros], 0)
            return top_k

        sort_pooling = tf.map_fn(sort_a_graph, self.graph_indexes, dtype=tf.float32)
        return sort_pooling

    def cnn1d_layers(self, inputs):
        total_dim = sum(self.gcnn_dims)
        graph_embeddings = tf.reshape(inputs, [-1, self.k * total_dim, 1])  # (batch, width, channel)
        first_conv = self.first_conv_layer(graph_embeddings)
        first_conv_pool = self.first_pooling_layer(first_conv)

        second_conv = self.second_conv_layer(first_conv_pool)
        return second_conv

    def fc_layer(self, inputs):
        cnn1d_embed = self.flatten_layer(inputs)
        outputs = self.dense_layer(cnn1d_embed)
        return outputs