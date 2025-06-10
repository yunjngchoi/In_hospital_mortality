import tensorflow as tf
from tensorflow.keras import layers, regularizers, optimizers, losses
from tensorflow.keras.models import Model

def build_dnn(input_size):
    """
    Build a DNN model with two hidden layers.
    """
    # Input layer
    input_data = layers.Input(shape=(input_size, ))

    # Hidden layer 1: Dense + LeakyReLU + Dropout
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001), kernel_initializer=tf.random_normal_initializer(stddev=0.01))(input_data)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(0.5)(x)

    # Hidden layer 2: Dense + LeakyReLU + Dropout
    x = layers.Dense(32, kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(0.2)(x)

    # Output layer
    x = layers.Dense(1, activation='sigmoid')(x)

    # Model definition and compilation
    model = Model(inputs=input_data, outputs=x)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=losses.binary_crossentropy)
    
    return model
