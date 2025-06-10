from tensorflow.keras import layers, regularizers, optimizers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

def get_dnn_model(input_size):
    """
    Initialize a Deep Neural Network model using Keras Functional API
    - 2 hidden layers with LeakyReLU and Dropout
    - L2 regularization and Adam optimizer
    """
    input_data = layers.Input(shape=(input_size,))
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001), kernel_initializer='random_normal')(input_data)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(rate=0.5)(x)

    x = layers.Dense(32, kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(rate=0.2)(x)

    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_data, outputs=output)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=losses.binary_crossentropy)
    
    return model
