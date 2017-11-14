---
layout: post
title: Data Analysis: Part 1
---

Quick Data Exploration (Variable identification)
First, identify Predictor (Input) and Target (output) variables. Next, identify the data type and category of the variables.

	df.head(10) - see rows and columns, variables


Quick Data Exploration (Univariate Analysis)
	df.describe() – see the amount of missing values, get a possible skew of the data by comparing 				mean and median
	df[‘column_name’].value_counts() – see the frequency distribution of categorical variables
df['ApplicantIncome'].hist(bins=50)- Plotting histograms. This helps us observe extreme values.
