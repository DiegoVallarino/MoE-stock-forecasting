# Adaptive Market Intelligence: A Mixture of Experts Framework for Volatility-Sensitive Stock Forecasting

This repository contains the full source code, data preprocessing scripts, and evaluation routines for the paper:

**"Adaptive Market Intelligence: A Mixture of Experts Framework for Volatility-Sensitive Stock Forecasting"**  
by Diego Vallarino (2025)  
[arXiv Preprint - Coming Soon]

---

## Overview

This project implements a Mixture of Experts (MoE) architecture that dynamically combines a Recurrent Neural Network (RNN) and a Linear Regression model for stock price forecasting. The model is explicitly designed to adapt to heterogeneous volatility regimes observed across firms.

- **Volatile stocks** are modeled using RNNs (LSTM layers)
- **Stable stocks** are modeled using traditional linear regression
- A **static gating mechanism** assigns weights based on realized volatility classification

---

## Dataset

- **Source:** Yahoo Finance via the `yfinance` Python package
- **Coverage:** 30 U.S.-listed publicly traded companies  
- **Time range:** January 1, 2015 – December 31, 2024  
- **Frequency:** Daily adjusted closing prices  
- **Sectors:** Technology, Energy, Consumer Goods, Financials, Industrials, etc.

---

## Methodology

1. **Data Acquisition:** Downloaded using `yfinance` and adjusted for corporate actions.
2. **Volatility Classification:** Firms are labeled as *Volatile* or *Stable* based on their 21-day rolling log-return standard deviation.
3. **Sequence Construction:** Input features consist of 20-day windows of log-returns; the next day’s return is the target.
4. **Modeling:**
   - RNN for volatile firms
   - Linear regression for stable firms
5. **Mixture of Experts (MoE):**
   - Static weighting:  
     - Volatile: 70% RNN + 30% LM  
     - Stable: 70% LM + 30% RNN
6. **Validation:**  
   - Walk-forward validation  
   - Volatility-stratified error reporting  
   - Multi-horizon forecasting (5, 20, 60 days)
7. **Metrics:** RMSE, MAE, and optional MASE.

---

## Files Included

| File | Description |
|------|-------------|
| `MoE_stock_forecasting.py` | Main pipeline: data download, preprocessing, modeling, evaluation |
| `requirements.txt` | Python dependencies |
| `figures/` | Optional output plots and summary tables |
| `README.md` | This file |

---

## Citation

If you use this code or findings in academic work, please cite:

```bibtex
@unpublished{vallarino2025adaptive,
  author = {Vallarino, Diego},
  title = {Adaptive Market Intelligence: A Mixture of Experts Framework for Volatility-Sensitive Stock Forecasting},
  note = {SSRN Preprint},
  year = {2025},
  url = {https://github.com/DiegoVallarino/MoE-stock-forecasting}
}


