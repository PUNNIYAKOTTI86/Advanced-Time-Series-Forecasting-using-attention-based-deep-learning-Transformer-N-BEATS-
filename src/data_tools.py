import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def sliding_window(arr, lookback):
    X_list, y_list = [], []
    for t in range(lookback, len(arr)):
        X_list.append(arr[t-lookback:t])
        y_list.append(arr[t])
    X = np.array(X_list).reshape(-1, lookback, 1)
    y = np.array(y_list).reshape(-1, 1)
    return X, y

def scale_and_build(values, lookback=6):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
    X, y = sliding_window(scaled, lookback)
    return (echo         torch.tensor(X, dtype=torch.float32),echo         torch.tensor(y, dtype=torch.float32),echo         scalerecho     )
