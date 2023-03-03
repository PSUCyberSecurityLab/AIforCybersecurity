benign_label = 0
malware_label = 1

num_classes = 2
learning_rate = 0.0300
max_gradient_norm = 1.0
dropout = 0.5
batch_size = 500
epochs = 50
train_size = 16545
test_size = 4137

# cnn mlp
img_height = 32
img_width = 32
image_type = 'L'
channel = 1

# gcnn 
gcnn_dims = [32, 32, 32]
feature_dim = 345
top_k = 1000

# rnn
hidden_units = 16
sequence_length = 50#500
max_features = 257#TODO
max_size = 1000000
overlap = 0

dataset = "data/dataset.pickle"
CFG_split_data = "data/CFG_split_data.pickle"
ALL_split_data = "data/ALL_split_data.pickle"
data_dir = 'data/'+image_type+'/'
model_save_path = "models/"
graph_data = 'data/graph.pickle'
seq_data = 'data/seq.pickle'
seed = 2612
