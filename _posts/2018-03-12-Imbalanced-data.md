---
layout: post
title: Machine Learning Project on Imbalanced Datasets
---




## Introduction ##
Imbalanced data typically refers to a problem with classification problems where the classes are not represented equally(e.g. 90% of the data belongs to one class). They are commonly seen in  fraud detection, cancer detection, manufacturing defects, and online ads conversion. 

Working on an imbalanced dataset tends to be extremely tricky as simple classification algorithms tend to struggle in such situations, usually resulting in overfitting on the majority class and completely ignoring the minority class.

The dataset in this project is taken from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/). However you will find the column names not included in the dataset- you can download the training data and the test data in the links below, courtesy of the Analytics Vidhya team:

[Download Training Data](https://www.analyticsvidhya.com/wp-content/uploads/2016/09/train.zip)

[Download Test data](https://www.analyticsvidhya.com/wp-content/uploads/2016/09/test.zip)

## 1. The Problem Statement ##
#### _"Given various features, the aim is to build a predictive model to determine the income level for people in US. The income levels are binned at below 50K and above 50K."_

From the problem statement, it’s evident that this is a binary classification problem, to find out if the income level is below or above 50k, based on a set of features stated below: 

1. Age
2. Marital Status
3. Income
4. Family Members
5. No. of Dependents
6. Tax Paid
7. Investment (Mutual Fund, Stock)
8. Return from Investments
9. Education
10. Spouse Education
11. Nationality
12. Occupation
13. Region in US
14. Race
15. Occupation category

_**Note:**_ I will be completing the data exploration / data manipulation using Python and the machine learning portion using R.

## 2. Data Exploration ##
Let's begin by importing the libraries required, and loading the training data and test data into Python.
`In````javascript
#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#To display our plots and graphs inline
%matplotlib inline
```
```javascript
#Loading the train and test data
train = pd.read_csv("C:/Users/ML/Documents/train.csv")
test = pd.read_csv("C:/Users/ML/Documents/test.csv")
```
```javascript
#how many columns our training dataset has
train.info()
```
`
```javascript
#how many rows our training dataset has
len(train)
```
