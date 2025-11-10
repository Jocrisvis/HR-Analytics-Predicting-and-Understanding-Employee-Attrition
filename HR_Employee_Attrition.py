# HR-Analytics-Predicting-and-Understanding-Employee-Attrition
## 1 Phase - Data Management
import logreg
#%%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

att = pd.read_csv("C:/Users/Windows/PycharmProjects/DSprojects/HR Project/Emp_Attrition.csv")
demo = pd.read_csv("C:/Users/Windows/PycharmProjects/DSprojects/HR Project/Emp_Demo.csv")
inc = pd.read_csv("C:/Users/Windows/PycharmProjects/DSprojects/HR Project/Emp_Income.csv")
det = pd.read_csv("C:/Users/Windows/PycharmProjects/DSprojects/HR Project/Emp_Job Details.csv")
#%%
### 1. We will fix issues and errors we have found, like datatype, blank cells and NaN.
### 2. We start checking empty value and if possible identifying which values.
### Counting how many empty or blanks cells we have in our master data.

empty = inc.isna().sum().sum()
print(empty)

b = inc[inc["EmpId"].isin([100015, 100038])]
print(b)
#%%
# Drop NaN, 0 values.
inc = inc.drop(index=14)
inc = inc.drop(index=37)
inc

#%%
### To find duplicates values
det.duplicated().sum()
duplicates = det[det.duplicated()]
duplicates
#%%
# Drop duplicated value and view
det = det.drop(index=148)
rows = det.iloc[[146,147,148,149]]
rows

#%%
new1 = pd.merge(att,demo, on = 'EmpId')
new2 = pd.merge(inc,det, on = 'EmpId')
master_data = pd.merge(new1,new2, on = 'EmpId')
# pd.options.display.max_columns = 100 If we want to display all the columns
master_data.head(6)

#%%

master_data["MonthlyIncome"] = master_data["MonthlyIncome"].astype("int64")
master_data["HourlyRate"] = master_data["HourlyRate"].astype("int64")
master_data.info()

#%%
# Department_x and Department_y are equals. We will drop one of them.
master_data = master_data.drop('Department_y', axis=1)

# StandardHours_x and StandardHours_y are equals. No valuable information.
master_data = master_data.drop('StandardHours_x', axis=1)
master_data = master_data.drop('StandardHours_y', axis=1)

#%%
## 0 Phase - Quick Visualization

master_data_1 = master_data.copy() # For Quick Visualization

master_data_1 = master_data_1.drop('EmpId', axis=1)

#Binary¨: Attrition, Gender, OverTime
master_data_1["Attrition"] = master_data_1["Attrition"].apply(lambda x: 1 if x == 'Yes' else 0)
master_data_1["Gender"] = master_data_1["Gender"].apply(lambda x: 1 if x == 'Male' else 0)
master_data_1["OverTime"] = master_data_1["OverTime"].apply(lambda x: 1 if x == 'Yes' else 0)
#%%
master_data_1.Education.value_counts()
master_data_1.JobLevel.value_counts()
master_data_1.JobRole.value_counts()
master_data_1.StockOptionLevel.value_counts()
master_data_1.MaritalStatus.value_counts()
master_data_1.BusinessTravel.value_counts()
master_data_1.EducationField.value_counts()
master_data_1.Education.value_counts()

#%%
# One hot encoding
master_data_1 = master_data_1.join(pd.get_dummies(master_data_1["BusinessTravel"])).drop('BusinessTravel', axis=1)
master_data_1 = master_data_1.join(pd.get_dummies(master_data_1["Department_x"], prefix='Department')).drop('Department_x', axis=1)
master_data_1 = master_data_1.join(pd.get_dummies(master_data_1["EducationField"], prefix='EducationField')).drop('EducationField', axis=1)
master_data_1 = master_data_1.join(pd.get_dummies(master_data_1["JobRole"], prefix='JobR')).drop('JobRole', axis=1)
master_data_1 = master_data_1.join(pd.get_dummies(master_data_1["MaritalStatus"], prefix='MaritalStat')).drop('MaritalStatus', axis=1)

# Boolean being managed
master_data_1 = master_data_1.map(lambda x: 1 if x is True else 0 if x is False else x)
master_data_1.sample(3)

# Histogram
import matplotlib.pyplot as plt

master_data_1.hist(figsize=(24,15))
plt.show()

#%%

# 6. Finally, we detect our attrition rate using the formula and obtaining a %

attrition_rate = (master_data['Attrition'].value_counts()[1] / len(master_data)) * 100

print(f"Overall Attrition Rate: {attrition_rate:.2f}%")


#%%

# HR-Analytics-Predicting-and-Understanding-Employee-Attrition
## 2 Phase - Data Analysis
import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))
plt.boxplot(
    [master_data['Age'], master_data['DistanceFromHome'],master_data['TotalWorkingYears'],
        master_data['NumCompaniesWorked'],master_data['YearsAtCompany'], master_data['YearsInCurrentRole'],
        master_data['YearsSinceLastPromotion'],master_data['YearsWithCurrManager'],master_data['TrainingTimesLastYear']],

        tick_labels=['Age', 'Home Distance','Total Working years', '# Previous Companies',
             'Years Company', 'Years Role','Years Last Promotion','Years w. Manager','Training Times Last Year'],

        orientation = 'horizontal')

plt.title('Whisker Box Plot')
plt.show()

#%%
plt.figure(figsize=(8,5))
plt.boxplot(master_data['MonthlyIncome'], tick_labels=['Monthly Income'], orientation='horizontal')
plt.title('Boxplot')
plt.show()







#%%
# 2. Box Whisker plot grouped by attrition:
import seaborn as sns

# We will identify a number of numeric_cols
numeric_cols = ['Age','DistanceFromHome','HourlyRate','TotalWorkingYears', 'NumCompaniesWorked', 'YearsAtCompany',
                'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'TrainingTimesLastYear']

# Melt dataframe for seaborn (change the dataframe from wide to long)
df_melted = master_data.melt(id_vars='Attrition', value_vars=numeric_cols, var_name='Variable', value_name='Value')

# Create boxplot
plt.figure(figsize=(10,6))
sns.boxplot(x='Value', y='Variable', hue='Attrition', data=df_melted, orientation='horizontal')
plt.title('Box-Whisker Plot by Attrition')
plt.xticks(rotation=45)  # rotate labels if many variables
plt.show()

#%%
# We will identify a number of numeric_cols
numeric_cols2 = ['MonthlyIncome']

# Melt dataframe for seaborn
df_melted2 = master_data.melt(id_vars='Attrition', value_vars=numeric_cols2, var_name='Variable', value_name='Value')

# Create boxplot
plt.figure(figsize=(10,4))
sns.boxplot(x='Value', y='Variable', hue='Attrition', data=df_melted2, orientation='horizontal')
plt.show()






#%%
# Mean of numeric variables.


# Group by Attrition (0/1) and take the mean of numeric columns
# We create group of columns in order to have a better visualization.
sel_cols = ['Age', 'DistanceFromHome', 'Education', 'HourlyRate', 'PercentSalaryHike',
            'StockOptionLevel', 'JobLevel', 'JobSatisfaction','NumCompaniesWorked',
            'PerformanceRating', 'TotalWorkingYears','TrainingTimesLastYear', 'WorkLifeBalance',
            'YearsAtCompany', 'YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']

the_mean = master_data.groupby("Attrition")[sel_cols].mean()

# Plot as bar chart
the_mean.T.plot(kind="bar", figsize=(18,14))
plt.title("Mean of Numeric Variables by Attrition (0=No, 1=Yes)")
plt.ylabel("Mean Value")
plt.xlabel("Numeric Variables")
plt.xticks(rotation=55)
plt.legend(fontsize= 30, title="Attrition")
plt.grid(axis='y', linestyle='--')
plt.show()

#%%
income = ['MonthlyIncome']
the_mean2 = master_data.groupby("Attrition")[income].mean()

# Plot as bar chart
the_mean2.T.plot(kind="bar", figsize=(7,6))
plt.title("Mean of Numeric Variables by Attrition (0=No, 1=Yes)")
plt.ylabel("Mean Value")
plt.xticks(rotation=1800)
plt.legend(title="Attrition")
plt.grid(axis='y', linestyle='--')
plt.show()

#%%
# New df in order to create a heatmap, naives bays, decision tree and random forest.
df1 = master_data_1.copy() # For Heatmap I
df2 = master_data_1.copy() # For Heatmap II
df3 = master_data_1.copy() # For Binary Logistic Regression
df4 = master_data_1.copy() # Naïve Bayes Method
df5 = master_data_1.copy() # Decision Tree
df6 = master_data_1.copy() # Random Forest Method

#%%
pivot_table = df1.pivot_table(
    index='JobSatisfaction',
    columns='OverTime',
    values='Attrition',
    aggfunc='mean'
)

# Plot heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="Reds")
plt.title("Attrition Rate: JobSatisfaction vs OverTime")
plt.ylabel("Job Satisfaction")
plt.xlabel("OverTime (Yes/No)")
plt.show()

#%%

# Create bins for YearsSinceLastPromotion (e.g. 0-2, 2-4, etc.)
df2['PromotionBin'] = pd.cut(df2['YearsSinceLastPromotion'], bins=5)

# Create pivot table for attrition rate
pivot_table2 = df2.pivot_table(
    index='WorkLifeBalance',
    columns='PromotionBin',
    values='Attrition',
    aggfunc='mean'
)

# Plot heatmap
plt.figure(figsize=(10,6))
sns.heatmap(pivot_table2, annot=True, fmt=".2f", cmap="Reds")
plt.title("Attrition Rate: WorkLifeBalance vs YearsSinceLastPromotion")
plt.ylabel("Work-Life Balance")
plt.xlabel("Years Since Last Promotion (binned)")
plt.show()

#%%

# HR-Analytics-Predicting-and-Understanding-Employee-Attrition
## 3 Phase - Attrition Model using Binary Logistic Regression



'''
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = master_data_1.drop('Attrition', axis=1), master_data_1['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_jobs=-1)
model.fit(X_train, y_train)
#%% md
# THIS IS A RANDOMFORESTCLASSIFIER IN A SIMPLER VERSION
#
#%%
model.score(X_test, y_test)
#%%
model.feature_importances_
#%%
sorted_importances = dict(sorted(zip(model.feature_names_in_, model.feature_importances_), key=lambda x: x[1], reverse = True))
#%%
plt.figure(figsize=(20,8))
plt.bar(sorted_importances.keys(), sorted_importances.values())
plt.xticks(rotation=45, ha = 'right')
plt.show()

'''
#%%
''' DIVISION PARA CREAR BINARY LOGISTIC REGRESSION '''
#%%

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


#%%
x_variables =['Age', 'DistanceFromHome', 'Education', 'Gender',
        'HourlyRate', 'MonthlyIncome', 'PercentSalaryHike', 'StockOptionLevel',
        'JobLevel', 'JobSatisfaction', 'NumCompaniesWorked',
        'PerformanceRating', 'TotalWorkingYears', 'TrainingTimesLastYear',
        'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
        'YearsSinceLastPromotion', 'YearsWithCurrManager', 'OverTime',
        'Non-Travel', 'Travel_Frequently',
        'Department_Human Resources', 'Department_Research & Development',
        'EducationField_Human Resources',
        'EducationField_Life Sciences', 'EducationField_Marketing',
        'EducationField_Medical', 'EducationField_Other',
        'JobR_Healthcare Representative',
        'JobR_Human Resources', 'JobR_Laboratory Technician', 'JobR_Manager',
        'JobR_Manufacturing Director', 'JobR_Research Director',
        'JobR_Research Scientist', 'JobR_Sales Executive',
        'MaritalStat_Divorced', 'MaritalStat_Married']
X = df3.loc[:,x_variables]
y = df3.loc[:,'Attrition']

#%%

# # When we create dummies, we do not use all the new columns created. One should be eliminated.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state = 0)
#%%
y_train.shape # shape of the array
#%%
df3.shape # shape of the array
#%%
X_train_const = sm.add_constant(X_train)  # add intercept
Logit_model = sm.Logit(y_train, X_train_const).fit()
print(Logit_model.summary())
#%%
'''
# Calculate VIF for each explanatory variable
vif_data = pd.DataFrame()
vif_data["feature"] = X_train_const.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_train_const.values, i)
    for i in range(X_train_const.shape[1])
]

print(vif_data)
'''
#%%
'''
HourlyRate could be causing a high level of multicollinearity for JobLevel and MonthlyIncome. 
we will proceed to eliminate that variable and  when we create dummies, we do not use all the new columns created. 
One should be eliminated and being taken as reference.

After a 2nd check, We still have some multicollinearity some variables as Department of Research and Sales.
'''
#%%
'''
x_variables =['Age', 'DistanceFromHome', 'Education', 'Gender',
       'HourlyRate', 'PercentSalaryHike', 'StockOptionLevel',
       'JobSatisfaction', 'NumCompaniesWorked',
       'PerformanceRating', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager', 'OverTime','Travel_Frequently', 'Travel_Rarely',
       'EducationField_Human Resources',
       'EducationField_Life Sciences', 'EducationField_Marketing','EducationField_Medical',
       'EducationField_Technical Degree', 'JobR_Healthcare Representative',
       'JobR_Human Resources', 'JobR_Laboratory Technician', 'JobR_Manager',
       'JobR_Manufacturing Director', 'JobR_Research Director',
       'JobR_Research Scientist', 'JobR_Sales Representative', 'MaritalStat_Divorced',
       'MaritalStat_Married']

# X and y will be the variables tha we will use for Log. regression. *maybe later we add numbers X2,x3,x3
X = df3.loc[:,x_variables]
y = df3.loc[:,'Attrition']
# When we create dummies, we do not use all the new columns created. One should be eliminated.
#%%
# Deparment_sales y Hourlyrate eliminados por la multicolinearidad
#%%
#Splitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state = 0)
#%%
y_train.shape
#%%
df3.shape
#%%
X_train_const = sm.add_constant(X_train)  # add intercept
Logit_model = sm.Logit(y_train, X_train_const).fit()
print(Logit_model.summary())
#%%

# Calculate VIF for each explanatory variable
vif_data = pd.DataFrame()
vif_data["feature"] = X_train_const.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_train_const.values, i)
    for i in range(X_train_const.shape[1])
]

print(vif_data)
'''
#%%
'''
We will select only significant variables and check multicollinearity

We check multicolinearity following the next point as references:

VIF = 1 → There is no multicolinearity
VIF entre 1 y 5 → Moderate multicolinearity, generally accepted.
VIF > 5 → High multicolinearity, to check variable.
'''
#%%
x_variables_significant =['DistanceFromHome', 'HourlyRate',
       'MonthlyIncome', 'JobSatisfaction', 'NumCompaniesWorked',
       'TotalWorkingYears', 'TrainingTimesLastYear',
       'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole','YearsSinceLastPromotion',
       'YearsWithCurrManager', 'EducationField_Life Sciences', 'EducationField_Medical',
       'OverTime','Non-Travel', 'Travel_Frequently', 'MaritalStat_Divorced',
       'MaritalStat_Married']
X_sigf = df3.loc[:,x_variables_significant]
y_sigf = df3.loc[:,'Attrition']
#%%
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_sigf, y_sigf, test_size=0.20,random_state = 0)
#%%

X_train_const_s = sm.add_constant(X_train_s)  # agregar intercepto
Logit_model_s = sm.Logit(y_train_s, X_train_const_s).fit()
print(Logit_model_s.summary())

#%%
# Calculate VIF for each explanatory variable
vif_data = pd.DataFrame()
vif_data["feature"] = X_train_const_s.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_train_const_s.values, i)
    for i in range(X_train_const_s.shape[1])
]

print(vif_data)

#%%
'''
No multicollinearity detected after selecting just the signficant variables.

'''
#%%
# TRAIN DATA - We will obtain AUC and ROC

y_pred_probtrain_s = Logit_model_s.predict(X_train_const_s)
fpr, tpr, thresholds =roc_curve(y_train_s,y_pred_probtrain_s)

roc_auc = auc(fpr, tpr)

#%%
plt.figure();
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc);
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--');
plt.xlim([0.0, 1.0]);plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate');
plt.title('Receiver operating characteristic');plt.legend(loc="lower right");
plt.show()

#%%

# 6.	Obtain classification table and accuracy (%)

predicted_values1= Logit_model_s.predict(X_train_const_s)
threshold=0.5
predicted_class1=np.zeros(predicted_values1.shape)
predicted_class1[predicted_values1>threshold]=1

from sklearn.metrics import classification_report
print(classification_report(y_train_s,predicted_class1))

#%% md
# 7.	Obtain threshold to balance sensitivity and specificity(note that default threshold is 0.5)
#%%
fpr, tpr, thresholds =roc_curve(y_train_s,y_pred_probtrain_s)
# roc_auc = auc(fpr, tpr)

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = roc.iloc[optimal_idx][4]
optimal_threshold

# TRAIN DATA - Optimal threshold















#%%
predicted_values1=logreg.predict_proba(X_train)[::,1]
threshold=optimal_threshold
predicted_class1=np.zeros(predicted_values1.shape)
predicted_class1[predicted_values1>threshold]=1

from sklearn.metrics import classification_report
print(classification_report(y_train,predicted_class1))
#%% md
# TEST DATA - Optimal Threshold (comparison between TRAIN DATA with optimal threshold)
#%%
y_pred_probtrain = logreg.predict_proba(X_test)[::,1]
fpr, tpr, thresholds =roc_curve(y_test,y_pred_probtrain)
roc_auc = auc(fpr, tpr)

ruc_auc = auc(fpr,tpr)

plt.figure();
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % ruc_auc);
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--');
plt.xlim([0.0, 1.0]);plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate');
plt.title('Receiver operating characteristic');plt.legend(loc="lower right");
plt.show()
#%%
predicted_values1=logreg.predict_proba(X_test)[::,1]
threshold=optimal_threshold
predicted_class1=np.zeros(predicted_values1.shape)
predicted_class1[predicted_values1>threshold]=1

print(classification_report(y_test,predicted_class1))
#%% md
# The Area Under ROC Curve (AUC) for train and test data indicate that the model is stable and performed well on test data. The sensitivity value using optimum threshold is 64% for test data
#%% md
# Note: Recall or Sensitivity , Precision and Accuracy (To check concepts or add it to the note)
#%% md
# variables train data, test data y luego variable coded significantes (creo que solo el train de esas)
#%% md
# esto es el tercero con las variables significates. pero no me sale no se porque
#%%

#%%
















