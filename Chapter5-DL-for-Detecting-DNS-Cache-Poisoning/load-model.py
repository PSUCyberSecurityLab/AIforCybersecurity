import os
import argparse
import pickle

import numpy as np
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

    n_epoch = 10
    n_batch = 50

    ss=FLAGS.k//90

    X_train=np.load(os.path.join("data","X_train-%.3d.npy"%ss),allow_pickle=True)
    X_test=np.load(os.path.join("data","X_test-%.3d.npy"%ss),allow_pickle=True)
    Y_train=np.load(os.path.join("data","Y_train-%.3d.npy"%ss),allow_pickle=True)
    Y_test=np.load(os.path.join("data","Y_test-%.3d.npy"%ss),allow_pickle=True)

    predictions_train=[]
    predictions_test=[]
    for i in range(4):
        model=tf.keras.models.load_model(os.path.join("model","classifier-"+str(FLAGS.n)+"-"+str(FLAGS.k)+"-"+str(i)+".h5"),custom_objects={"metric":F1Score(num_classes=2)})
        predictions_train.append(model.predict(X_train))
        predictions_test.append(model.predict(X_test))

    model.summary()

    pred_train=np.average(predictions_train,axis=0)
    pred_test=np.average(predictions_test,axis=0)

    print("training set:")
    Mat=perf_measure(Y_train,pred_train)
    print("Model accuracy: %.2f %%" % ((Mat[0]+Mat[2])/sum(Mat)*100))
    print("Model f1 score: %.4f" % (Mat[0]/(Mat[0]+0.5*(Mat[1]+Mat[3]))))
    print("Model detection rate: %.2f %%" % (Mat[0]/(Mat[0]+Mat[3])*100))
    if Mat[1]+Mat[2]==0:
        print("FPR NaN!")
    else:
        print("Model FPR: %.2f %%\n" % (Mat[1]/(Mat[1]+Mat[2])*100))
    print("(TP, FP, TN, FN):"+str(Mat))
    print("test set:")
    Mat=perf_measure(Y_test,pred_test)
    print("Model accuracy: %.2f %%" % ((Mat[0]+Mat[2])/sum(Mat)*100))
    print("Model f1 score: %.4f" % (Mat[0]/(Mat[0]+0.5*(Mat[1]+Mat[3]))))
    print("Model detection rate: %.2f %%" % (Mat[0]/(Mat[0]+Mat[3])*100))
    if Mat[1]+Mat[2]==0:
        print("FPR NaN!")
    else:
        print("Model FPR: %.2f %%\n" % (Mat[1]/(Mat[1]+Mat[2])*100))
    print("(TP, FP, TN, FN):"+str(Mat))
