import numpy as np

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def wape(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = np.sum(np.abs(y_true)) + 1e-9
    return np.sum(np.abs(y_true - y_pred)) / denom

def mase(y_true, y_pred, training_series):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    train = np.asarray(training_series).reshape(-1)
    mae_model = np.mean(np.abs(y_true - y_pred))
    naive_errors = np.abs(train[1:] - train[:-1])
    mae_naive = np.mean(naive_errors) + 1e-9
    return mae_model / mae_naive
