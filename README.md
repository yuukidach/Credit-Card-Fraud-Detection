# Credit Card Fraud Detection

## Introduction

This is the group project of COMP7404B. Original dataset can be found at [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). 
It contains data about credit card transactions that occurred during a period of two days, with 492 frauds out of 284, 807 transactions.
All variables in the dataset are numerical. The data has been transformed using PCA transformation(s) due to privacy reasons.

We use Autoencoder Neural Network + SVM for anomaly detection in credit card transaction data and evaluate its performences with only using logistic regression and svm.

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

* `fraud_detection.ipynb`: is the main file with Autoencoder + SVM
* `autoencoder.ipynb`: try to use Antoencoder to do classification
* `constrast.ipynb`: logitsic regression and SVM

To view the result, just download `docs/` folder is enough.

## Results

| Method | Precision | Recall | F-1 Score |
| :---: | :--- | :---: | :---: |
| AE + SVM | 0.84 | 0.75 | 0.79 |
| Logistic Regression | 0.85 | 0.61 | 0.71 |
| SVM | 0 | 0 | NA |

1. `SVM` does not work for this imbalanced dataset.

2. `AE + SVM` has nearly the same precision with `logistic regression`. But the recall of `AE + SVM` is higher, thus results in a better F-1 score.

3. `AE + SVM` performs best among 3 methods.

Autoencoder is able to effectively extract sample features and can be applied to dealing with imbalanced dataset.
