import tensorflow as tf
from tensorflow.keras import layers, regularizers, optimizers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

def build_model(input_size):
    """
    Build a DNN model with two hidden layers.
    """
    input_data = layers.Input(shape=(input_size,))
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001), kernel_initializer=tf.random_normal_initializer(stddev=0.01))(input_data)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(32, kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_data, outputs=x)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=losses.binary_crossentropy)
    
    return model

def train_model(train_x, train_y, input_size):
    """
    Train a DNN model with early stopping.
    """
    model = build_dnn(input_size)
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, min_delta=0.001)
    model.fit(train_x, train_y, epochs=50, batch_size=64, callbacks=[early_stopping], verbose=0)
    
    return model

def evaluate_model(model, data, label):
    """
    Evaluate a trained DNN model on given data.
    """
    prob = model.predict(data)
    pred = (prob > 0.5).astype(int)

    acc = accuracy_score(label, pred)
    cm = confusion_matrix(label, pred)
    tn, fp, fn, tp = cm.ravel()

    pre = tp / (tp + fp) if (tp + fp) != 0 else 0
    spe = tn / (tn + fp) if (tn + fp) != 0 else 0
    sen = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * pre * sen / (pre + sen) if (pre + sen) != 0 else 0
    balacc = (spe + sen) / 2
    roc = roc_auc_score(label, prob)

    print('Confusion Matrix:\n', cm)
    print(f'Accuracy: {acc:.3f}, Specificity: {spe:.3f}, Sensitivity: {sen:.3f}, Precision: {pre:.3f}, F1 Score: {f1:.3f}, Balanced Accuracy: {balacc:.3f}, ROC AUC Score: {roc:.3f}\n')
