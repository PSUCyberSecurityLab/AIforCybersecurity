import config
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CNN_MAL(tf.keras.Model):
    def __init__(self, num_classes):
        """Initializes the CNN model
        :param num_classes: The number of classes in the dataset.
        """
        super(CNN_MAL, self).__init__(name="CNN_MAL")
        self.num_classes = num_classes
        
        def __graph__():
            # First convolutional layer
            self.normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
            self.first_conv_layer = layers.Conv2D(16, 3, padding='same', activation='relu')
            self.first_pooling_layer = layers.MaxPooling2D()
            # Second convolutional layer
            self.second_conv_layer = layers.Conv2D(32, 3, padding='same', activation='relu')
            self.second_pooling_layer = layers.MaxPooling2D()
            # Fully-connected layer (Dense Layer)
            self.flatten_layer = layers.Flatten()
            self.dense_layer = layers.Dense(128, activation='relu')
            # Dropout, to avoid overfitting
            self.dropout_layer = tf.keras.layers.Dropout(config.dropout)
            # Readout layer
            self.output_layer = layers.Dense(num_classes)

        __graph__()
    
    def call(self, x_input, training=False):
        input_image = tf.reshape(x_input, [-1, config.img_width, config.img_height, config.channel])
        x_input = self.normalization_layer(x_input)
        
        first_conv = self.first_conv_layer(input_image)
        first_conv_pool = self.first_pooling_layer(first_conv)

        second_conv = self.second_conv_layer(first_conv_pool)
        first_conv_pool = self.first_pooling_layer(first_conv)
        second_conv_pool = self.second_pooling_layer(second_conv)

        second_conv_pool_flatten = self.flatten_layer(second_conv_pool)
        dense_layer = self.dense_layer(second_conv_pool_flatten)
        dropout_layer = self.dropout_layer(dense_layer, training = training)
        
        logits = self.output_layer(dropout_layer)
        return logits

