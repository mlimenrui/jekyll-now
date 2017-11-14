---
layout: post
title: Data Analysis Part 1
---




Quick Data Exploration (Variable identification)
First, identify Predictor (Input) and Target (output) variables. Next, identify the data type and category of the variables.

	df.head(10)
This helps us see rows and columns, variables


Quick Data Exploration (Univariate Analysis)

	df.describe()

This helps use see the amount of missing values, get a possible skew of the data by comparing the mean and median
	
	df[‘column_name’].value_counts() 

This helps us see the frequency distribution of categorical variables

	df['ApplicantIncome'].hist(bins=50)

This plots the histograms. This helps us observe extreme values.

