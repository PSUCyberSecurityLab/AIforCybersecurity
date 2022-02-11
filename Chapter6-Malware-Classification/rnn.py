import config
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, TimeDistributed, Bidirectional

class RNN_MAL(tf.keras.Model):
    def __init__(self, num_classes, rnn_type):
        """Initializes the RNN model
        :param num_classes: The number of classes in the dataset.
        :param rnn_type: basic rnn, rnn, gru
        """
        super(RNN_MAL, self).__init__(name="RNN_MAL")
        self.num_classes = num_classes
        
        def __graph__():
            self.embedding_layer = layers.Embedding(config.max_features, config.hidden_units)
            #self.normalization_layer = layers.experimental.preprocessing.Rescaling(1./config.max_features)
            if rnn_type == 'lstm':
                self.rnn = LSTM(config.hidden_units, return_state=False, return_sequences=True)
            elif rnn_type == 'rnn':
                self.rnn = SimpleRNN(config.hidden_units, return_state=False, return_sequences=True)
            elif rnn_type == 'gru':
                self.rnn = GRU(config.hidden_units, return_state=False, return_sequences=True)
            self.biRNN = Bidirectional(self.rnn)
            self.dropout_layer1 = layers.Dropout(config.dropout)
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.attention_context = tf.Variable(tf.random.truncated_normal([config.hidden_units * 2, 1]))
            self.relation_embedding = tf.Variable(tf.random.truncated_normal([num_classes, config.hidden_units * 2]))
            self.sen_d = tf.Variable(tf.random.truncated_normal([num_classes]))
            self.output_layer = layers.Dense(num_classes)
        __graph__()
    
    def call(self, inputs, training):
        #x_input = tf.reshape(x_input, (-1, config.img_width, config.img_height * config.channel))#int(config.max_size / config.sequence_length), config.sequence_length])
        #x_input = self.normalization_layer(x_input)
        x_input = inputs[0]
        self.shape = inputs[1]
        self.query = inputs[2]
        inputs_embedded = self.embedding_layer(x_input)
        outputs = self.biRNN(inputs_embedded)
        outputs = self.layernorm1(outputs)
        outputs = tf.reshape(outputs, [-1, config.sequence_length * config.hidden_units * 2])
        outputs = self.across_sum_layer(outputs, training = training)
        #outputs = self.sub_attention(outputs)
        #outputs = self.across_attention_layer(outputs, training = training)
        outputs = self.dropout_layer1(outputs, training = training)
        logits = self.output_layer(outputs)
        return logits 
    def across_sum_layer(self, attention_r, training):
        def sub_windows_sum(index_span):
            indices = tf.range(index_span[0], index_span[1])  
            bag_hidden_mat = tf.gather(attention_r, indices) 
            return tf.reduce_sum(bag_hidden_mat, 0)
        sort_pooling = tf.map_fn(sub_windows_sum, self.shape, dtype=tf.float32)
        return sort_pooling
    
    def across_attention_layer(self, attention_r, training):
        def __logit__(x):
            return tf.matmul(x, tf.transpose(self.relation_embedding)) + self.sen_d
        def sub_windows_train_attention(index_span):
            indices = tf.range(index_span[0], index_span[1])  
            bag_hidden_mat = tf.gather(attention_r, indices) 
            attention_score = tf.nn.softmax(tf.reshape(tf.gather(attention_logit, indices) ,[1, -1]))
            return tf.reshape(tf.matmul(attention_score, bag_hidden_mat),[config.hidden_units*2])
        def sub_windows_test_attention(index_span):
            indices = tf.range(index_span[0], index_span[1])  
            bag_hidden_mat = tf.gather(attention_r, indices) 
            attention_score = tf.nn.softmax(tf.transpose(tf.gather(attention_logit, indices)))
            return tf.linalg.diag_part(tf.nn.softmax(__logit__(tf.matmul(attention_score,bag_hidden_mat)), -1))
            
        if training:
            current_rel = tf.reshape(tf.nn.embedding_lookup(self.relation_embedding, self.query), [-1, config.hidden_units*2])
            attention_logit = tf.reduce_sum(current_rel * attention_r, -1)
            sort_pooling = tf.map_fn(sub_windows_train_attention, self.shape, dtype=tf.float32)
            sort_pooling = __logit__(sort_pooling)
        else:
            attention_logit = tf.matmul(attention_r, tf.transpose(self.relation_embedding))
            sort_pooling = tf.map_fn(sub_windows_test_attention, self.shape, dtype=tf.float32)
        return sort_pooling

    def sub_attention(self, inputs):
        tmp = tf.matmul(tf.reshape(tf.tanh(inputs),[-1, config.hidden_units * 2]),self.attention_context)
        tmp = tf.matmul(tf.expand_dims(tf.nn.softmax(tf.reshape(tmp,[-1, config.sequence_length])),1), inputs)
        return tf.reshape(tmp,[-1, config.hidden_units * 2])
