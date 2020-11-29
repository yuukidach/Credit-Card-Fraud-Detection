# Credit Card Fraud Detection

## Introduction

This is the group project of COMP7404B. Original dataset can be found at [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). 
It contains data about credit card transactions that occurred during a period of two days, with 492 frauds out of 284,807 transactions.
All variables in the dataset are numerical. The data has been transformed using PCA transformation(s) due to privacy reasons.

We use Autoencoder Neural Network + SVM for anomaly detection in credit card transaction data and evaluate its performences with ony using logistic regression and svm.

## File Structure

``` shell
.
├── autoencoder.ipynb
├── constrast.ipynb
├── dataset
│   └── creditcard.csv
├── docs
│   ├── autoencoder.html
│   ├── constrast.html
│   ├── Data analysis for credit card fraud detection.pptx
│   └── fraud_detection.html
├── fraud_detection.ipynb
├── model.h5
├── original.png
└── README.md
```

- `fraud_detection.ipynb`: is the main file with Autoencoder + SVM
- `autoencoder.ipynb`: try to use Antoencoder to do classification
- `constrast.ipynb`: logitsic regression and SVM

To view the result, just download `docs/` folder is enough.
