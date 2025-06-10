import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_split(filepath):
    """
    Load dataset and split into train/test sets.
    """
    df = pd.read_csv(filepath)

    data = df.drop(columns=['death'])
    label = df['death']
    
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.2, stratify=label, random_state=1)

    return train_x, test_x, train_y, test_y, data.columns.tolist()

def scaling(train_x, test_x, cols):
    """
    Apply MinMax scaling to training and testing data.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_x)

    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    train_x = pd.DataFrame(train_x, columns=cols)
    test_x = pd.DataFrame(test_x, columns=cols)

    return train_x, test_x
