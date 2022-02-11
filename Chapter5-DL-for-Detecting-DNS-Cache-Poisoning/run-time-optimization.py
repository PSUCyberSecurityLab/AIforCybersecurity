import os
import argparse

import tensorflow as tf
from tensorflow_addons.metrics import F1Score

FLAGS=None

def perf_measure(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_pred)):
        if y_actual[i][1] == 1 and y_pred[i][1] >= 0.5:
            TP += 1
        if y_pred[i][1] >= 0.5 and y_actual[i][1] == 0:
            FP += 1
        if y_actual[i][1] == 0 and y_pred[i][1] < 0.5:
            TN += 1
        if y_pred[i][1] <= 0.5 and y_actual[i][1] == 1:
            FN += 1

    return(TP, FP, TN, FN)

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '--n',
        type=int,
        required=True,
        help='ID of training.'
    )
    parser.add_argument(
        '--k',
        type=int,
        required=True,
        help='ID of model. The models to load should be placed in the model directory, named \"classifier-N-K-[0,1,2,3].h5\"'
    )
    FLAGS, unparsed =parser.parse_known_args()

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        tf.config.threading.set_inter_op_parallelism_threads(8)
        tf.config.threading.set_intra_op_parallelism_threads(8)

    for i in range(4):
        model=tf.keras.models.load_model(os.path.join("model","classifier-"+str(FLAGS.n)+"-"+str(FLAGS.k)+"-"+str(i)+".h5"),custom_objects={"metric":F1Score(num_classes=2)})

        converter=tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations=[tf.lite.Optimize.DEFAULT]
        tflite_quantilized_model=converter.convert()
        fi=open(os.path.join("model","tflite_quantilized_model-"+str(FLAGS.n)+"-"+str(FLAGS.k)+"-"+str(i)+".tflite"),'wb')
        fi.write(tflite_quantilized_model)
        fi.close()
