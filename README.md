# Advanced Time Series Forecasting (Attention + SARIMAX)

## Overview
This project compares a classical forecasting method (SARIMAX) with an attention-based deep learning approach (Transformer Encoder).

## Models
- SARIMAX baseline (statsmodels)
- Transformer Encoder (PyTorch)

## Metrics
- RMSE
- WAPE
- MASE

## How to Run
### Install
```bash
conda create -n ts_attention python=3.10
conda activate ts_attention
pip install -r requirements.txt
