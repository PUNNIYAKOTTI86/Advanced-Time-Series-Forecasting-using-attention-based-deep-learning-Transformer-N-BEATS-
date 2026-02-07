import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, seq_length=6):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def prepare_data(values):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values.reshape(-1, 1))
    X, y = create_sequences(scaled)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), scaler
