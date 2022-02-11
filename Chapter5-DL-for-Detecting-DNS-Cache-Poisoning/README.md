# Detecting DNS cache poisoning with deep learning

## Required packages:

numpy

pyshark

tqdm

tensorflow

tensorflow_addons

scikit-learn

iteration_utilities

## Contents
```
┌───────────────────────────────────┐
│Chapter5-DL-for-DNS-Cache-Poisoning│
└┬──────────────────────────────────┘
 │
 ├─► data ===► place raw, intermediate, and processed data
 │   │
 │   ├─► benign-dns_split[1,2,3,...,55].pcapng
 │   │
 │   └─► malicious-dns_split[1,2,3,4,5].pcapng
 │
 ├─► model ===► place trained models and training logs
 │   │
 │   └─► ...
 │
 ├─► chop-by-session.py
 │
 ├─► load-model.py
 │
 ├─► load-optimized-model.py
 │
 ├─► process-to-datasets.py
 │
 ├─► README.md
 │
 ├─► run-time-optimization.py
 │
 └─► train-classifiers.py
```

## Usage instructions

### Step 1: Run "chop-by-session.py":

   No arguments needed.

   Reads raw data (.pcapng) in the "data" directory, and outputs intermediate data (.npy) to the "data" directory.

### Step 2: Run "process-to-datasets.py":

   Optional argument:
   
   --cpu-num CPU_NUM --> Number of CPU cores for parallel execution. For machines with 16 GB or less RAM, the default value 1 is recommended.

   Reads intermediate data (.npy) in the "data" directory, and outputs processed data sets (.npy) to the "data" directory.

   Please note that this script can occupy lots of RAM, especially if CPU_NUM>1. At least 16 GB of RAM is recommended.

### Step 3: Run "train-classifiers.py":

   No arguments needed.

   Reads processed data sets (.npy) in the "data" directory, and trains a series of neural networks. At the same time, the training process is logged in a text file (.log), and the trained models (.h5) are also saved. Both the log and trained modes are saved to the "model" directory.

   Note that a total of 2250 combinations of tunable parameters are tried, and 4-fold cross-validation is applied. As a result, 9000 models will be trained by default, which may take days to finish.

### Step 4: Run "load-model.py":

   Required arguments:

   --n N --> ID of training.

   --k K --> ID of model. The models to load should be placed in the "model" directory, named "classifier-N-K-[0,1,2,3].h5"

   Loads specified models (.h5) in the "model" directory, reads needed data sets (.npy) in the "data" directory, and evaluates the specified model.

   Note that the evaluation results are based on the voting results of all 4 models trained on different folds.

   Four example trained models are uploaded to the "model" directory. To evaluate them, please run "python load-model.py --n 0 --k 450" after the step 2 above.

### Step 5: Run "run-time-optimization.py"

   Required arguments:

   --n N --> ID of training.

   --k K --> ID of model. The models to load should be placed in the "model" directory, named "classifier-N-K-[0,1,2,3].h5"

   Loads specified models (.h5) in the "model" directory, optimizes the models, and output optimized models (.tflite) in the "model" directory.

   To optimize the four example trained models in the "model" directory, please run "python run-time-optimization.py --n 0 --k 450".

### Step 6: Run "load-optimized-model.py"

   --n N --> ID of training.

   --k K --> ID of optimized model. The models to load should be placed in the "model" directory, named "tflite_quantilized_model-N-K-[0,1,2,3].tflite"

   Loads specified optimized models (.tflite) in the "model" directory, reads needed data sets (.npy) in the "data" directory, and evaluates the specified optimized model.

   Note that the evaluation results are based on the voting results of all 4 models from different folds.
