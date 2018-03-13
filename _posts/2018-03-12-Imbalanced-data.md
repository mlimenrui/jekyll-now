---
layout: post
title: Machine Learning Project on Imbalanced Datasets
---




## Introduction ##
Imbalanced data typically refers to a model with classification problems where the classes are not represented equally(e.g. 90% of the data belongs to one class). They are commonly seen in  fraud detection, cancer detection, manufacturing defects, and online ads conversion analytics.

Working on an imbalanced dataset tends to be extremely tricky as simple classification algorithms tend to struggle in such situations, usually resulting in overfitting on the majority class and completely ignoring the minority class. This post details a guide on how to conduct data analysis and machine learning using an imbalanced dataset to predict a classification outcome. 

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
```javascript
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
`<class 'pandas.core.frame.DataFrame'>`  
`RangeIndex: 199523 entries, 0 to 199522`  
`Data columns (total 41 columns):`  
`.`  
`.`  


```javascript
#how many rows our training dataset has
len(train)
```
`199523`

```javascript
#how many rows our training dataset has
len(test)
```
`99762`

We see that train data has 199523 rows & 41 columns, and test data has 99762 rows and 41 columns. Generally, test data comes with one less column than train (the variable we want to predict; income_level). It means that this data set has test prediction values also. This will help us in evaluating our model.

Let's verify the target variable we want to predict for both train and test datasets. 

```javascript
train['income_level'].value_counts()
```
`-50000    187141`   
`50000     12382`  
`Name: income_level, dtype: int64`  

```javascript
test['income_level'].value_counts()
```

`-50000        93576`  
`50000+.       6186`    
`Name: income_level, dtype: int64`  

We can already see some discrepancy in our target variable- the value 50000 is different across train and test data. This disparity will cause trouble in model evaluation. Being a binary classification problem, we can encode these variables as 0 and 1. I'm showing two different ways(for train and test data) we can do this below.

```javascript
#Changing train dataset income levels to binary
train['income_level'].replace(-50000, '0',inplace=True)
train['income_level'].replace(50000, '1',inplace=True)
```
```javascript
#Changing test dataset income levels to binary
a1 = test['income_level'].str.contains('-50000')
test['income_level'] = np.select([a1], [0], default=1)
```
Now let's see how it looks like.
```javascript 
train['income_level'].value_counts()
```
`0    187141`  
`1     12382`  
`Name: income_level, dtype: int64`  

Perfect! we've successfully changed this to binary values (0 and 1). Now let’s look at the severity of imbalanced classes in our data. Already we can see that 187141(out of 199523) of our values are 0; this is our majority class with a proportion of 94%. In other words, with a decent ML algorithm, our model would get 94% model accuracy. In absolute figures, it looks incredible. But our performance would depend on how good can we predict the **minority classes**.

Let’s separate the categorical variables & numerical variables. This will help us in conducting our distribution analysis.
```javascript
#state which column numbers categorical and numerical variables belong to
factcols = np.r_[1:5,7,8:16,20:29,31:38,40]
numcols = np.r_[0,5:7,16:20,29:31,38:40]
```
```javascript
#subset the categorical variables and numerical variables for train and test data
cat_train = train.iloc[:, factcols]
num_train = train.iloc[:, numcols]

cat_test = test.iloc[:, factcols]
num_test = test.iloc[:, numcols]
```

## 3. Distribution/Bivariate Analysis ##
Let’s conduct our distribution analysis for **numerical variables** now. The best way to understand these variables is by using a Histogram plot.
```javascript
num_train['age'].hist(bins=100)
```
![histogram on age]({{ site.baseurl }}/images/histogram.png "an image title")

As we can see, the data set consists of people aged from 0 to 90 with the frequency of people **declining** with age. Using some intuition, it is highly unlikely the population below age 20 could earn >50K under normal circumstances. Therefore, we can **bin this variable into age groups**_(covered in part 5. Data Manipulation)_.

```javascript
num_train['capital_losses'].hist(bins=20)
```
![histogram on capital losses]({{ site.baseurl }}/images/histogram2.png "an image title")

This is a nasty right skewed graph. In skewed distribution, normalizing is always an option. But we need to look into this variable deeper as this insight isn’t significant enough for decision making. One option could be to check for unique values. If they are less, we can tabulate the distribution (done in upcoming sections).

Furthermore, in classification problems, we should also plot numerical variables with dependent variable. This would help us determine the clusters (if exists) of classes 0 and 1. For this, we need to add the target variable in num_train data:

```javascript
plt.scatter(num_train['age'], num_train['wage_per_hour'], c= cat_train['income_level'], s=10)
```
![scatterplot of age vs wage per hour colored by income level]({{ site.baseurl }}/images/scatter.png "an image title")

_yellow scatter plots being those with income_level of binary value 1, and purple being those with income_level of binary value 0._

As we can see, most of the people having income_level 1, seem to fall in the age of 25-65 earning wage of $1000 to $4000 per hour. This plot further strengthens our assumption that age < 20 would have income_level 0, hence we will bin this variable.

Similarly, we can visualize our *categorical variables* as well. Let's do so for `class_of_worker` variable.


We can see that this variable looks imbalanced i.e. only two category levels seem to dominate. In such situation, a good practice is to **combine levels having less than 5% frequency** of the total category frequency _(covered in part 5. Data Manipulation)_. 
The response `Not in universe` category appears unintuitive. Let’s assume that this response is given by people who got frustrated (due to any reason) while filling their census data.

Let's visualize our `education` variable.

Evidently, all children have `income_level` 0. Also, we can infer than Bachelors degree holders have the largest proportion of people have income_level 1. Similarly, you can plot other categorical variables also.

## 4. Data Cleaning ##

Let’s check for missing values in numeric variables.
```javascript
#Check for null values in our numerical columns
num_train.apply(lambda x: sum(x.isnull()),axis=0)
```

We see that numeric variables has no missing values. Good for us! Now, let’s check for missing values in categorical data.
We find that some of the variables have ~50% missing values for cat_train; namely the `migration` columns. High proportion of missing value can be attributed to difficulty in data collection. For now, we’ll remove these category levels, for both train and test data.

```javascript
cat_train = cat_train.drop(['migration_msa','migration_reg', 'migration_within_reg', 'migration_sunbelt'], axis = 1)
cat_test = cat_test.drop(['migration_msa','migration_reg', 'migration_within_reg', 'migration_sunbelt'], axis = 1)
```
For the rest of missing values, a nicer approach would be to label them as ‘Unavailable’. Imputing missing values on large data sets can be painstaking.

```javascript
cat_train = cat_train.fillna("Unavailable")
```
Let's look at `country_father`, which previously had 6713 NA values, which should be labelled as `Unavailable` now.
cat_train['country_father'].value_counts()


