import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

def evaluation_dnn(model, data, label):
    """
    Evaluate DNN model performance on given data and labels.
    """

    # Predict probabilities and convert to binary predictions
    probability = model.predict(data)
    pred = (probability > 0.5).astype(int)

    # Accuracy
    acc = accuracy_score(label, pred)
    
    # Confusion matrix
    cm = confusion_matrix(label, pred)
    print('Confusion Matrix:\n', cm)

    # Calculate precision, specificity, sensitivity, F1, balanced accuracy
    tn, fp, fn, tp = cm.ravel()
    pre = tp / (tp + fp) if (tp + fp) != 0 else 0
    spe = tn / (tn + fp) if (tn + fp) != 0 else 0
    sen = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * pre * sen / (pre + sen) if (pre + sen) != 0 else 0
    balacc = (spe + sen) / 2

    # ROC AUC score
    roc = roc_auc_score(label, probability)

    # Print evaluation results
    print(f'Accuracy: {acc:.3f}, Specificity: {spe:.3f}, Sensitivity: {sen:.3f}, Precision: {pre:.3f}, 'f'F1 Score: {f1:.3f}, Balanced Accuracy: {balacc:.3f}, ROC AUC Score: {roc:.3f}\n')
