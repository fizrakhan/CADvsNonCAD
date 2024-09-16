# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 20:12:02 2024

@author: HP
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

file_name = "C:/Users/HP/Documents/Masters/Small Sample Data Analysis/Projectdata.xls"

#importing data
df = pd.read_excel(file_name)

disease = df.groupby("CAD")["MaxHR"].count()
print(disease)

#filtering data into two dataframes for disease with 
#CAD and Non-CAD

disease1 = df[df['CAD'] == 1]
disease0 = df[df['CAD'] == 0]

#some general characteristics for CAD patients
disease1.describe
disease1.count()

disease1[['MaxHR',
'ST-DEP',
'RWA',
'ST/HR',
'Age'
]].median()

disease1[['MaxHR',
'ST-DEP',
'RWA',
'ST/HR',
'Age'
]].std()

##to calculate confidence interval
from scipy.stats import t
conf_intervals_1 = {}
selected_columns = ['MaxHR',
'ST-DEP',
'RWA',
'ST/HR',
'Age'
]

for column in disease1:
    if column in selected_columns:
        mean = disease1[column].mean()
        std_dev = disease1[column].std()
        margin_of_error = t.ppf((1 + 0.95) / 2, len(disease1) - 1) * (std_dev / np.sqrt(len(disease1)))
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
        conf_intervals_1[column] = (lower_bound, upper_bound)

print("Confidence Intervals:")
for column, interval in conf_intervals_1.items():
    print(f"{column}: {interval}")
    
##general characteristics for non-CAD patients
disease0[['MaxHR',
'ST-DEP',
'RWA',
'ST/HR',
'Age'
]].median()

disease0[['MaxHR',
'ST-DEP',
'RWA',
'ST/HR',
'Age'
]].mean()

disease0[['MaxHR',
'ST-DEP',
'RWA',
'ST/HR',
'Age'
]].std()

##to calculate confidence interval
from scipy.stats import t
conf_intervals_0 = {}

for column in disease0:
    if column in selected_columns:
        mean = disease0[column].mean()
        std_dev = disease0[column].std()
        margin_of_error = t.ppf((1 + 0.95) / 2, len(disease0) - 1) * (std_dev / np.sqrt(len(disease0)))
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
        conf_intervals_0[column] = (lower_bound, upper_bound)

print("Confidence Intervals:")
for column, interval in conf_intervals_0.items():
    print(f"{column}: {interval}")
    
##Task 1.2
#distribution plotting for parameters in CAD using seaborn library
plt.figure(figsize=(45, 30))
plt.subplot(2, 2)
    
sns.distplot(disease1['MaxHR'])
sns.distplot(disease1['ST-DEP'])
sns.distplot(disease1['RWA'])
sns.distplot(disease1['ST/HR'])
sns.distplot(disease1['Age'])

#making another column having gender as 0 or 1 in CAD
disease1['Sex_binary'] = np.where(disease1['Sex'] == 'F', 0, 1)
#F = 0, M = 1

#distribution plot for new gender column
sns.distplot(disease1['Sex_binary'])


from scipy.stats import shapiro

#using shapiro-wilk test for normality testing in CAD
statsHR = shapiro(disease1["MaxHR"])
statsST_Dep = shapiro(disease1['ST-DEP'])
statsRWA = shapiro(disease1['RWA'])
statsST_HR = shapiro(disease1['ST/HR'])
statsAge = shapiro(disease1['Age'])

print(statsHR)
print(statsST_Dep)
print(statsRWA)
print(statsST_HR)
print(statsAge)

#distribution plotting for parameters in non-CAD using seaborn library
sns.distplot(disease0['MaxHR'])
sns.distplot(disease0['ST-DEP'])
sns.distplot(disease0['RWA'])
sns.distplot(disease0['ST/HR'])
sns.distplot(disease0['Age'])

#using shapiro-wilk test for normality testing in non-CAD
statsHR = shapiro(disease0["MaxHR"])
statsST_Dep = shapiro(disease0['ST-DEP'])
statsRWA = shapiro(disease0['RWA'])
statsST_HR = shapiro(disease0['ST/HR'])
statsAge = shapiro(disease0['Age'])

print(statsHR)
print(statsST_Dep)
print(statsRWA)
print(statsST_HR)
print(statsAge)

#making another column having gender as 0 or 1 in non-CAD
disease0['Sex_binary'] = np.where(disease0['Sex'] == 'F', 0, 1)
sns.distplot(disease0['Sex_binary'])


##Task 1.1

#making a profile for descriptive characteristics in CAD
import pandas_profiling
profile = pandas_profiling.ProfileReport(disease1)
profile.to_file(output_file='C:/Users/HP/Documents/Masters/Small Sample Data Analysis/CAD_Patients')
print(profile)

#making a profile for descriptive characteristics in non-CAD
import pandas_profiling
profile = pandas_profiling.ProfileReport(disease0)
profile.to_file(output_file='C:/Users/HP/Documents/Masters/Small Sample Data Analysis/nonCAD_Patients')
print(profile)


##Task 2.1
##statistical difference between patients with CAD and without CAD with respect to
##distributions of ST depression, ST/HR index, ΔRWA and maxHR.

#since the data is not normal, we will use Mann Whitney U test

from scipy.stats import mannwhitneyu
mannwhitneyu(disease0['MaxHR'], disease1['MaxHR'])
mannwhitneyu(disease0['ST-DEP'], disease1['ST-DEP'])
mannwhitneyu(disease0['ST/HR'], disease1['ST/HR'])
mannwhitneyu(disease0['RWA'], disease1['RWA'])

##Task 2.2
##statistical difference between patients with CAD and without CAD with respect to a) age
mannwhitneyu(disease0['Age'], disease1['Age'])

##and b) sex distributions
from scipy.stats import chi2_contingency
contigency_table = pd.crosstab(df['CAD'], df['Sex'])
chi2_contingency(contigency_table)

##Task 2.3
##a) Are the age distributions of women and men having CAD statistically different? 

#filter data based on gender in CAD
female_CAD1 = disease1[disease1['Sex'] == 'F']
male_CAD1 = disease1[disease1['Sex'] == 'M']

#normality of Age using distribution plots
sns.distplot(female_CAD1['Age'])
plt.xlabel("Age in female patients")
sns.distplot(male_CAD1['Age'])

#normality of Age using shapiro test
shapiro(female_CAD1['Age'])
shapiro(male_CAD1['Age'])

#data in not normal, so mann-whitney u test
mannwhitneyu(female_CAD1['Age'], male_CAD1['Age'])

##b) What about distributions of maxHR in women and men having CAD?

#normality of maxHR using distribution plots
sns.distplot(female_CAD1['MaxHR'])
sns.distplot(male_CAD1['MaxHR'])

#normality of maxHR using shapiro test
shapiro(female_CAD1['MaxHR'])
shapiro(male_CAD1['MaxHR'])

#data is normal, so independent t-test
from scipy.stats import ttest_ind
ttest_ind(female_CAD1['MaxHR'], male_CAD1['MaxHR'])

##Task 3.1 
##statistically significant linear correlation between ST depression
##ST/HR index and ΔRWA
##a) in patients with CAD


#correlation between ST-DEP and MaxHR
from scipy.stats import spearmanr
spearmanr(disease1['ST-DEP'], disease1['ST/HR'])

##correlation between ST-DEP and RWA
spearmanr(disease1['ST-DEP'], disease1['RWA'])

##correlation between RWA and ST/HR
spearmanr(disease1['RWA'], disease1['ST/HR'])

##b) in patients without CAD

##correlation between ST-DEP and MaxHR
spearmanr(disease0['ST-DEP'], disease0['ST/HR'])

##correlation between ST-DEP and MaxHR
spearmanr(disease0['ST-DEP'], disease0['RWA'])

##correlation between ST/HR and RWA
spearmanr(disease0['RWA'], disease0['ST/HR'])

##Task 3.2
##Give the equation and significance level for linear regression between maxHR and age 

##a) in patient without CAD
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(disease0['Age'], disease0['MaxHR'])

print(slope, intercept, r_value, p_value)

plt.scatter(disease0['Age'], disease0['MaxHR'], label='Original data')
plt.plot(disease0['MaxHR'], slope * disease0['Age'] + intercept, label='Regression line', color='red')
plt.xlabel('Age')
plt.ylabel('MaxHR')
plt.legend()
plt.show()

##b) in patients with CAD
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(disease1['Age'], disease1['MaxHR'])

print(slope,intercept, r_value, p_value)

plt.scatter(disease1['Age'], disease1['MaxHR'], label='Original data')
plt.plot(disease1['Age'], slope * disease1['Age'] + intercept, label='Regression line', color='red')
plt.xlabel('Age')
plt.ylabel('MaxHR')
plt.legend()
plt.show()

##Task3.3
##a) Study by regression analysis whether there is a relationship between maxHR and ST depression. 

from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(df['ST-DEP'], df['MaxHR'])

print(slope,intercept, r_value, p_value)

plt.scatter(df['ST-DEP'], df['MaxHR'], label='Original data')
plt.plot(df['ST-DEP'], slope * df['ST-DEP'] + intercept, label='Regression line', color='red')
plt.xlabel('ST-DEP')
plt.ylabel('MaxHR')
plt.legend()
plt.show()

##Implement the same analysis for maxHR and ST/HR index

from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(df['ST/HR'], df['MaxHR'])

print(slope,intercept, r_value, p_value)

plt.scatter(df['ST/HR'], df['MaxHR'], label='Original data')
plt.plot(df['ST/HR'], slope * df['ST/HR'] + intercept, label='Regression line', color='red')
plt.xlabel('ST/HR')
plt.ylabel('MaxHR')
plt.legend()
plt.show()

##Task 3.4
##Test the correlation between age and RWA in total population

#normality of age and RWA in total population using dist plot
sns.distplot(df['Age'])
sns.distplot(df['RWA'])

#normality of age and RWA in total pop using shapiro test
shapiro(df['Age'])
shapiro(df['RWA'])

spearmanr(df['Age'], df['RWA'])

##Task 4.1 
##Calculate the
##sensitivity, specificity, diagnostic accuracy, and diagnostic performance for each classifier a) in
##a)total population
##No of people who are test positive or negative
df['ST-DEP Test'] = np.where(df['ST-DEP'] >= 0.10, 1, 0)
df['ST-DEP Test'].value_counts()

df['ST/HR Test'] = np.where(df['ST/HR'] >= 1.60, 1, 0)
df['ST/HR Test'].value_counts()

df['RWA Test'] = np.where(df['RWA'] >= 0.0, 1, 0)
df['RWA Test'].value_counts()

#No. of people who are both disease and test positive
df['ST-DEP dis-test-pos'] = np.where((df['CAD'] == 1) & (df['ST-DEP Test'] == 1), 1, 0)
df['ST-DEP dis-test-pos'].value_counts()

df['ST/HR dis-test-pos'] = np.where((df['CAD'] == 1) & (df['ST/HR Test'] == 1), 1, 0)
df['ST/HR dis-test-pos'].value_counts()

df['RWA dis-test-pos'] = np.where((df['CAD'] == 1) & (df['RWA Test'] == 1), 1, 0)
df['RWA dis-test-pos'].value_counts()

#No. of people who are both disease and test positive
df['ST-DEP dis-test-neg'] = np.where((df['CAD'] == 0) & (df['ST-DEP Test'] == 0), 1, 0)
df['ST-DEP dis-test-neg'].value_counts()

df['ST/HR dis-test-neg'] = np.where((df['CAD'] == 0) & (df['ST/HR Test'] == 0), 1, 0)
df['ST/HR dis-test-neg'].value_counts()

df['RWA dis-test-neg'] = np.where((df['CAD'] == 0) & (df['RWA Test'] == 0), 1, 0)
df['RWA dis-test-neg'].value_counts()

#confusion matrix for total population
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(df['CAD'], df['ST-DEP Test'])

# Display the confusion matrix as a heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)

plt.xlabel('ST-DEP')
plt.ylabel('CAD')
plt.title('Contigency table')
plt.show()

conf_matrix = confusion_matrix(df['CAD'], df['ST/HR Test'])

# Display the confusion matrix as a heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)

plt.xlabel('ST/HR')
plt.ylabel('CAD')
plt.title('Contigency table')
plt.show()

conf_matrix = confusion_matrix(df['CAD'], df['RWA Test'])

# Display the confusion matrix as a heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)

plt.xlabel('RWA')
plt.ylabel('CAD')
plt.title('Contigency table')
plt.show()

##b)in female population 
femalepop = df[df['Sex'] =='F']
femalepop['Sex'].value_counts()
from sklearn.metrics import confusion_matrix
conf_matrix_female = confusion_matrix(femalepop['CAD'], femalepop['ST-DEP Test'])

# Display the confusion matrix as a heatmap
sns.heatmap(conf_matrix_female, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('ST-DEP')
plt.ylabel('CAD')
plt.title('Confusion Matrix')
plt.show()

conf_matrix_female = confusion_matrix(femalepop['CAD'], femalepop['ST/HR Test'])

# Display the confusion matrix as a heatmap
sns.heatmap(conf_matrix_female, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('ST/HR')
plt.ylabel('CAD')
plt.title('Confusion Matrix')
plt.show()

conf_matrix_female = confusion_matrix(femalepop['CAD'], femalepop['RWA Test'])
sns.heatmap(conf_matrix_female, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('ST/HR')
plt.ylabel('CAD')
plt.title('Confusion Matrix')
plt.show()

##Task 4.2
##a) Calculate the sensitivity and specificity for ST/HR index with partition values 0.8, 1.2, 1.6, 2.0
##and 2.4 µV/bpm.
df['ST/HR Test1'] = np.where(df['ST/HR'] >=0.8, 1, 0)
df['ST/HR Test1'].value_counts()

df['ST/HR Test2'] = np.where(df['ST/HR'] >=1.2, 1, 0)
df['ST/HR Test2'].value_counts()

df['ST/HR Test3'] = np.where(df['ST/HR'] >=1.6, 1, 0)
df['ST/HR Test3'].value_counts()

df['ST/HR Test4'] = np.where(df['ST/HR'] >=2.0, 1, 0)
df['ST/HR Test4'].value_counts()

df['ST/HR Test5'] = np.where(df['ST/HR'] >=2.4, 1, 0)
df['ST/HR Test5'].value_counts()

from sklearn.metrics import confusion_matrix
conf_matrix1 = confusion_matrix(df['CAD'], df['ST/HR Test1'])
conf_matrix2 = confusion_matrix(df['CAD'], df['ST/HR Test2'])
conf_matrix3 = confusion_matrix(df['CAD'], df['ST/HR Test3'])
conf_matrix4 = confusion_matrix(df['CAD'], df['ST/HR Test4'])
conf_matrix5 = confusion_matrix(df['CAD'], df['ST/HR Test5'])

sns.heatmap(conf_matrix1, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('ST/HR index at 0.8')
plt.ylabel('CAD')
plt.title('Contigency table')
plt.show()

sns.heatmap(conf_matrix2, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('ST/HR index at 1.2')
plt.ylabel('CAD')
plt.title('Contigency table')
plt.show()

sns.heatmap(conf_matrix3, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('ST/HR index at 1.6')
plt.ylabel('CAD')
plt.title('Contigency table')
plt.show()

sns.heatmap(conf_matrix4, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('ST/HR index at 2.0')
plt.ylabel('CAD')
plt.title('Contigency table')
plt.show()

sns.heatmap(conf_matrix5, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('ST/HR index at 2.4')
plt.ylabel('CAD')
plt.title('Contigency table')
plt.show()


true_negative1 = conf_matrix1[0,0]
false_positive1 = conf_matrix1[0,1]
false_negative1 = conf_matrix1[1,0]
true_positive1 = conf_matrix1[1,1]

sensitivity1 = true_positive1 / (true_positive1 + false_negative1)
specificity1 = true_negative1 / (true_negative1 + false_positive1)

true_negative2 = conf_matrix2[0,0]
false_positive2 = conf_matrix2[0,1]
false_negative2 = conf_matrix2[1,0]
true_positive2 = conf_matrix2[1,1]

sensitivity2 = true_positive2 / (true_positive2 + false_negative2)
specificity2 = true_negative2 / (true_negative2 + false_positive2)

true_negative3 = conf_matrix3[0,0]
false_positive3 = conf_matrix3[0,1]
false_negative3 = conf_matrix3[1,0]
true_positive3 = conf_matrix3[1,1]

sensitivity3 = true_positive3 / (true_positive3 + false_negative3)
specificity3 = true_negative3 / (true_negative3 + false_positive3)

true_negative4 = conf_matrix4[0,0]
false_positive4 = conf_matrix4[0,1]
false_negative4 = conf_matrix4[1,0]
true_positive4 = conf_matrix4[1,1]

sensitivity4 = true_positive4 / (true_positive4 + false_negative4)
specificity4 = true_negative4 / (true_negative4 + false_positive4)

true_negative5 = conf_matrix5[0,0]
false_positive5 = conf_matrix5[0,1]
false_negative5 = conf_matrix5[1,0]
true_positive5 = conf_matrix5[1,1]

sensitivity5 = true_positive5 / (true_positive5 + false_negative5)
specificity5 = true_negative5 / (true_negative5 + false_positive5)

##c) Illustrate the ROC-curves also for ST depression and ΔRWA. Put the ROC-curves of all
##diagnostic classifiersinto the same figure. The area under ROC-curve facilitatesthe comparison
##of diagnostic classifiers without selecting the partition values: the larger area the better method. 

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Sample data (replace with your actual data)

# Compute ROC curve
fpr1, tpr1, thresholds1 = roc_curve(df['CAD'], df['ST-DEP'])
fpr2, tpr2, thresholds2 = roc_curve(df['CAD'], df['ST/HR'])
fpr3, tpr3, thresholds3 = roc_curve(df['CAD'], df['RWA'])

# Calculate Area Under the Curve (AUC)
roc_auc1 = auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)

# Plot ROC curve
plt.figure(figsize=(5, 5))
plt.plot(fpr1, tpr1, color='darkorange', lw=2, label=f'AUC of ST-DEP = {roc_auc1:.2f}')
plt.plot(fpr2, tpr2, color='blue', lw=2, label=f'AUC of ST/HR = {roc_auc2:.2f}')
plt.plot(fpr3, tpr3, color='red', lw=2, label=f'AUC of RWA = {roc_auc3:.2f}')
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

roc1_coord = pd.DataFrame()
roc1_coord['Sensitivty'] = tpr1
roc1_coord['1-Specificity'] = fpr1

roc2_coord = pd.DataFrame()
roc2_coord['Sensitivty'] = tpr2
roc2_coord['1-Specificity'] = fpr2

roc3_coord = pd.DataFrame()
roc3_coord['Sensitivty'] = tpr3
roc3_coord['1-Specificity'] = fpr3

##Task 4.3b
##Test whether there is a significance difference in sensitivity values at 85% specificity between
##ST depression and ST/HR index

from statsmodels.stats.contingency_tables import mcnemar
table_for_mcnemar = [[49, 33], [67, 51]]
results = mcnemar(table_for_mcnemar)
print(results)
