import os
import argparse

import numpy as np
import tensorflow as tf

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

if __name__=="__main__":
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
        help='ID of optimized model. The optimized models to load should be placed in the model directory, named \"tflite_quantilized_model-N-K-3.tflite\"'
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

    def run_tflite_model(data,indices):
        # Initialize the interpreter
        predictions = np.zeros((len(indices),2))
        for j in range(4):
            interpreter = tf.lite.Interpreter(model_path=os.path.join("model","tflite_quantilized_model-"+str(FLAGS.n)+"-"+str(FLAGS.k)+"-"+str(j)+".tflite"))
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()[0]

            for i, test_index in enumerate(indices):
                test_image = data[test_index]

            # Check if the input type is quantized, then rescale input data to uint8
                if input_details['dtype'] == np.uint8:
                    input_scale, input_zero_point = input_details["quantization"]
                    test_image = test_image / input_scale + input_zero_point

                test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
                interpreter.set_tensor(input_details["index"], test_image)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details["index"])[0]

                predictions[i]+=output

        return predictions/4

    print("training set:")
    train_indices=[x for x in range(len(X_train))]
    pred=run_tflite_model(X_train,train_indices)
    Mat=perf_measure(Y_train[train_indices],pred)
    print("Model accuracy: %.2f %%" % ((Mat[0]+Mat[2])/sum(Mat)*100))
    print("Model f1 score: %.4f" % (Mat[0]/(Mat[0]+0.5*(Mat[1]+Mat[3]))))
    print("Model detection rate: %.2f %%" % (Mat[0]/(Mat[0]+Mat[3])*100))
    if Mat[1]+Mat[2]==0:
        print("FPR NaN!")
    else:
        print("Model FPR: %.2f %%\n" % (Mat[1]/(Mat[1]+Mat[2])*100))
    print("(TP, FP, TN, FN):"+str(Mat))
    print("test set:")
    test_indices=[x for x in range(len(X_test))]
    pred=run_tflite_model(X_test,test_indices)
    Mat=perf_measure(Y_test[test_indices],pred)
    print("Model accuracy: %.2f %%" % ((Mat[0]+Mat[2])/sum(Mat)*100))
    print("Model f1 score: %.4f" % (Mat[0]/(Mat[0]+0.5*(Mat[1]+Mat[3]))))
    print("Model detection rate: %.2f %%" % (Mat[0]/(Mat[0]+Mat[3])*100))
    if Mat[1]+Mat[2]==0:
        print("FPR NaN!")
    else:
        print("Model FPR: %.2f %%\n" % (Mat[1]/(Mat[1]+Mat[2])*100))
    print("(TP, FP, TN, FN):"+str(Mat))
