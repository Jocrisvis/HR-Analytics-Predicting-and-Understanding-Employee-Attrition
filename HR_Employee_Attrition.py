# HR-Analytics-Predicting-and-Understanding-Employee-Attrition
## 1 Phase - Data Management

#%%
import pandas as pd
import numpy as np

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
'''
# One hot encoding
master_data_1 = master_data_1.join(pd.get_dummies(master_data_1["BusinessTravel"])).drop('BusinessTravel', axis=1)
master_data_1 = master_data_1.join(pd.get_dummies(master_data_1["Department_x"], prefix='Department')).drop('Department_x', axis=1)
master_data_1 = master_data_1.join(pd.get_dummies(master_data_1["EducationField"], prefix='EducationField')).drop('EducationField', axis=1)
master_data_1 = master_data_1.join(pd.get_dummies(master_data_1["JobRole"], prefix='JobR')).drop('JobRole', axis=1)
master_data_1 = master_data_1.join(pd.get_dummies(master_data_1["MaritalStatus"], prefix='MaritalStat')).drop('MaritalStatus', axis=1)

# Boolean being managed
master_data_1 = master_data_1.map(lambda x: 1 if x is True else 0 if x is False else x)
master_data_1.sample(3)
'''
#%%

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
df1 = master_data_1.copy() # For Heatmap, I
df2 = master_data_1.copy() # For Heatmap, II
df3 = master_data_1.copy() # For Binary Logistic Regression
'''
df4 = master_data_1.copy() # Naïve Bayes Method
df5 = master_data_1.copy() # Decision Tree
df6 = master_data_1.copy() # Random Forest Method
'''
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

import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


#%%

# THIS IS THE RIGHT VERIOSN WITHOUT ENCODING THE DATASET

X_train, X_test = train_test_split(df3, test_size=0.20,random_state = 0)

#%%
X_train.shape # shape of the array
#%%
X_test.shape # shape of the array

#%%

att_model = smf.logit(
    formula="""Attrition ~ Age + DistanceFromHome + Education + Gender + HourlyRate + MonthlyIncome + 
    PercentSalaryHike + StockOptionLevel + JobLevel + JobSatisfaction + NumCompaniesWorked + 
    TotalWorkingYears + TrainingTimesLastYear + WorkLifeBalance + YearsAtCompany + YearsInCurrentRole + 
    YearsSinceLastPromotion + YearsWithCurrManager + Department_x + EducationField + MaritalStatus + BusinessTravel + JobRole + OverTime""",
    data=X_train
).fit()
#%%

print(att_model.summary())

#%%

# To check multicolinearity
# Get design matrix (X1) from your fitted model
X0 = att_model.model.exog
feature_names = att_model.model.exog_names

# Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data["feature"] = feature_names
vif_data["VIF"] = [variance_inflation_factor(X0, i) for i in range(X0.shape[1])]

print(vif_data)

#%%

# 1. Get predicted probabilities for the positive class (Attrition=1)
# y_true = df3['Attrition']  # actual values
y_pred_prob_att = att_model.predict(X_train)  # predicted probabilities

# 2. ROC and AUC curve
Lfpr, Ltpr, thresholds = roc_curve(X_train['Attrition'], y_pred_prob_att)
auc_score = roc_auc_score(X_train['Attrition'], y_pred_prob_att)
print(f"AUC: {auc_score:.4f}")
#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.plot(Lfpr, Ltpr, color='darkorange', label=f'ROC curve (AUC = {auc_score:.4f})')
plt.plot([0,1], [0,1], color='blue', linestyle='--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Training Data')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

#%%
# TEST DATA -

# 1. Get predicted probabilities for the positive class (Attrition=1)
# y_true = df3['Attrition']  # actual values
y_pred_prob_att1 = att_model.predict(X_test)  # predicted probabilities

# 2. ROC and AUC curve
Lfpr, Ltpr, thresholds = roc_curve(X_test['Attrition'], y_pred_prob_att1)
auc_score = roc_auc_score(X_test['Attrition'], y_pred_prob_att1)
print(f"AUC: {auc_score:.4f}")
#%%

plt.figure(figsize=(8,6))
plt.plot(Lfpr, Ltpr, color='darkorange', label=f'ROC curve (AUC = {auc_score:.4f})')
plt.plot([0,1], [0,1], color='blue', linestyle='--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Test Data')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()





#%%
# SIGNIFICATN VARIABLES

reduced_model = smf.logit(
    formula="""
        Attrition ~ 
        MaritalStatus + BusinessTravel + JobRole + Gender + OverTime +
        Age + DistanceFromHome + JobSatisfaction + TotalWorkingYears + NumCompaniesWorked +
        TrainingTimesLastYear + WorkLifeBalance + YearsAtCompany +
        YearsInCurrentRole + YearsSinceLastPromotion + YearsWithCurrManager
    """,
    data=X_train
).fit()
print(reduced_model.summary())


#%%

# To check multicolinearity
# Get design matrix (X1) from your fitted model
X1 = reduced_model.model.exog
feature_names = reduced_model.model.exog_names

# Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data["feature"] = feature_names
vif_data["VIF"] = [variance_inflation_factor(X1, i) for i in range(X1.shape[1])]

print(vif_data)
#%%

# 1. Get predicted probabilities for the positive class (Attrition=1)
# y_true = df3['Attrition']  # actual values
y_pred_prob = reduced_model.predict(X_train)  # predicted probabilities

# 2. ROC and AUC curve
lfpr, ltpr, thresholds = roc_curve(X_train['Attrition'], y_pred_prob)
auc_score = roc_auc_score(X_train['Attrition'], y_pred_prob)
print(f"AUC: {auc_score:.4f}")
#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.plot(lfpr, ltpr, color='darkorange', label=f'ROC curve (AUC = {auc_score:.4f})')
plt.plot([0,1], [0,1], color='blue', linestyle='--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Training Data')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

#%%
# 6.	Obtain classification table and accuracy (%)

threshold=0.5
predicted_class1=np.zeros(y_pred_prob.shape)
predicted_class1[y_pred_prob>threshold]=1

print(classification_report(X_train['Attrition'],predicted_class1))

#%%
# TRAIN DATA - Optimal threshold

# 3. Compute Youden's J statistic (tpr - fpr)
youden_j = ltpr - lfpr

# 4. Find index of maximum J
optimal_idx = np.argmax(youden_j)

# 5. Obtain the optimal threshold
optimal_threshold = thresholds[optimal_idx]
print("Optimal threshold: ", optimal_threshold)


#%%
# TEST DATA -

# 1. Get predicted probabilities for the positive class (Attrition=1)
# y_true = df3['Attrition']  # actual values
y_pred_prob_t = reduced_model.predict(X_test)  # predicted probabilities

# 2. ROC and AUC curve
lfpr, ltpr, thresholds = roc_curve(X_test['Attrition'], y_pred_prob_t)
auc_score = roc_auc_score(X_test['Attrition'], y_pred_prob_t)
print(f"AUC: {auc_score:.4f}")
#%%

plt.figure(figsize=(8,6))
plt.plot(lfpr, ltpr, color='darkorange', label=f'ROC curve (AUC = {auc_score:.4f})')
plt.plot([0,1], [0,1], color='blue', linestyle='--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Test Data')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

#%%

predicted_class2=np.zeros(y_pred_prob_t.shape)
predicted_class2[y_pred_prob_t>threshold]=1

print(classification_report(X_test['Attrition'],predicted_class2))

#%%

# Confusion matrix: rows = actual, columns = predicted
cm = confusion_matrix(X_test['Attrition'], predicted_class2)
print("Confusion Matrix:\n", cm)

#%%

# Extract values
TN, FP, FN, TP = cm.ravel() # flattens the 2×2 matrix into a 1D array (e.g.: cm[0,0])

# Sensitivity and Specificity formulas
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print('Sensitivity:', sensitivity)
print('Specificity:', specificity)


#%%
# MIXED VARIABLES
mixed_model = smf.logit(
    formula="""Attrition ~ Age + DistanceFromHome + Education + Gender + MonthlyIncome + 
    PercentSalaryHike + StockOptionLevel + JobSatisfaction + NumCompaniesWorked + 
    TotalWorkingYears + TrainingTimesLastYear + WorkLifeBalance + YearsAtCompany + YearsInCurrentRole + 
    YearsSinceLastPromotion + YearsWithCurrManager + MaritalStatus + BusinessTravel + JobRole + OverTime""",
    data=X_train
).fit()


#%%

# 1. Get predicted probabilities for the positive class (Attrition=1)
# y_true = df3['Attrition']  # actual values
y_pred_prob_mix = mixed_model.predict(X_train)  # predicted probabilities

# 2. ROC and AUC curve
fpr, tpr, thresholds = roc_curve(X_train['Attrition'], y_pred_prob_mix)
auc_score = roc_auc_score(X_train['Attrition'], y_pred_prob_mix)
print(f"AUC: {auc_score:.4f}")
#%%

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {auc_score:.4f})')
plt.plot([0,1], [0,1], color='blue', linestyle='--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Training Data')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

#%%
# TEST DATA -

# 1. Get predicted probabilities for the positive class (Attrition=1)
# y_true = df3['Attrition']  # actual values
y_pred_prob_mix1 = mixed_model.predict(X_test)  # predicted probabilities

# 2. ROC and AUC curve
fpr, tpr, thresholds = roc_curve(X_test['Attrition'], y_pred_prob_mix1)
auc_score = roc_auc_score(X_test['Attrition'], y_pred_prob_mix1)
print(f"AUC: {auc_score:.4f}")
#%%

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {auc_score:.4f})')
plt.plot([0,1], [0,1], color='blue', linestyle='--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Test Data')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


#%%

# HR-Analytics-Predicting-and-Understanding-Employee-Attrition
## 4 Phase - ML Methods

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

df4 = master_data_1.copy() # Naïve Bayes Method
# df5 = master_data_1.copy() # Decision Tree
# df6 = master_data_1.copy() # Random Forest Method

#%%

#Naive Bayes Method

from sklearn.naive_bayes import GaussianNB
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
X = df4.loc[:,x_variables]
y = df4.loc[:,'Attrition']


# # When we create dummies, we do not use all the new columns created. One should be eliminated.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state = 0)

#%%

# Create and fit the Gaussian Naive Bayes modelx
NBmodel = GaussianNB()

NBmodel.fit(X_train, y_train)


#%%
# Prediction for train and test data

y_pred_train = NBmodel.predict_proba(X_train)[:,1]
y_pred_test = NBmodel.predict_proba(X_test)[:,1]

# Train Data
auc_train = roc_auc_score(y_train, y_pred_train)
fpr_gnb_train, tpr_gnb_train, _ = roc_curve(y_train, y_pred_train)

# Test Data
auc_test = roc_auc_score(y_test, y_pred_test)
fpr_gnb_test, tpr_gnb_test, _ = roc_curve(y_test, y_pred_test)

# Plot
plt.figure(figsize=(8,8))
lw = 2

# Train curve
plt.plot(fpr_gnb_train, tpr_gnb_train, color='darkorange', lw=lw,
         label='Train ROC (AUC = %0.3f)' % auc_train)

# Test curve
plt.plot(fpr_gnb_test, tpr_gnb_test, color='green', lw=lw,
         label='Test ROC (AUC = %0.3f)' % auc_test)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Gaussian Naive Bayes')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

#%%

# Confusion Matrix for TRAIN DATA

y_pred_naives = NBmodel.predict(X_train)

cm_n_train = confusion_matrix(y_train,y_pred_naives)
print(cm_n_train)
#%%

# Confusion Matrix for TEST DATA

y_pred_naives = NBmodel.predict(X_test)

cm_n_test = confusion_matrix(y_test,y_pred_naives)
print(cm_n_test)

#%%

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

dtcl = DecisionTreeClassifier(criterion='entropy', min_samples_split= int(len(X_train)*.10))
dtcl.fit(X_train, y_train)

# Prediction for train and test data

y_pred_train_prob = dtcl.predict_proba(X_train)[:,1]
y_pred_test_prob = dtcl.predict_proba(X_test)[:,1]

# Train
auc_train = roc_auc_score(y_train, y_pred_train_prob)
fpr_dt_train, tpr_dt_train, _ = roc_curve(y_train, y_pred_train_prob)

# Test
auc_test = roc_auc_score(y_test, y_pred_test_prob)
fpr_dt_test, tpr_dt_test, _ = roc_curve(y_test, y_pred_test_prob)



#%%

plt.figure(figsize=(6,6))
lw = 2
plt.plot(fpr_dt_train, tpr_dt_train, color='darkorange', lw=lw,
         label='Train ROC (AUC = %0.3f)' % auc_train)

plt.plot(fpr_dt_test, tpr_dt_test, color='green', lw=lw,
         label='Test ROC (AUC = %0.3f)' % auc_test)

plt.plot([0,1],[0,1], color='navy', lw=lw, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


#%%

# Train
y_pred_dt = dtcl.predict(X_train)
cm_tr_dt = confusion_matrix(y_train, y_pred_dt)
TN, FP, FN, TP = cm_tr_dt.ravel()
print(cm_tr_dt)
print(f"Accuracy: {(TN+TP)/(TN+FP+FN+TP):.3f}")
print(f"Sensitivity: {TP/(TP+FN):.3f}")
print(f"Specificity: {TN/(TN+FP):.3f}\n")
#%%

# Test
y_pred_test_class = dtcl.predict(X_test)
cm_test_dt = confusion_matrix(y_test, y_pred_test_class)
TN, FP, FN, TP = cm_test_dt.ravel()
print(cm_test_dt)
print(f"Accuracy: {(TN+TP)/(TN+FP+FN+TP):.3f}")
print(f"Sensitivity: {TP/(TP+FN):.3f}")
print(f"Specificity: {TN/(TN+FP):.3f}")

#%%

# Random Forest Method

#Building Model
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=0, class_weight='balanced', n_estimators=100, min_samples_split=0.1,
oob_score=True, max_features='sqrt')
rf.fit(X_train, y_train)


#%%
# Prediction for train and test data
y_pred_train = rf.predict_proba(X_train)[:, 1]   # Probabilidad clase 1
y_pred_test = rf.predict_proba(X_test)[:, 1]


# Train Data
auc_train = roc_auc_score(y_train, y_pred_train)
fpr_rf_train, tpr_rf_train, _ = roc_curve(y_train, y_pred_train)

# Test Data
auc_test = roc_auc_score(y_test, y_pred_test)
fpr_rf_test, tpr_rf_test, _ = roc_curve(y_test, y_pred_test)

# --- PLOT BOTH ---
plt.figure(figsize=(6,6))
lw = 2

# Train curve
plt.plot(fpr_rf_train, tpr_rf_train, color='darkorange', lw=lw,
         label='Train ROC (AUC = %0.3f)' % auc_train)

# Test curve
plt.plot(fpr_rf_test, tpr_rf_test, color='green', lw=lw,
         label='Test ROC (AUC = %0.3f)' % auc_test)

# Diagonal line (random guess)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

#%%

# Confusion Matrix for TRAIN DATA

y_pred_rf = rf.predict(X_train)
cm_tr_rf = confusion_matrix(y_train, y_pred_rf)

print(cm_tr_rf)

#%%

# Confusion Matrix for TEST DATA

y_pred_rf_test = rf.predict(X_test)
cm_test_rf = confusion_matrix(y_test, y_pred_rf_test)

print(cm_test_rf)


#%%
hrtext = [line.rstrip() for line in open("C:/Users/Windows/Documents/GitHub/HR-Analytics-Predicting-and-Understanding-Employee-Attrition/Dataset/comments.txt", "r", encoding= "utf-8")]
hrtext[0:5]
#%%
import contractions

# Join corpus into one string (1 dimension)
text = " ".join(hrtext).lower()
# Expand contractions
text = contractions.fix(text)


#%%

import re
# No special characters or spaces
text = re.sub(r'[^a-zA-Z\s]', '', text)

#%%

words = text.split()

#%%
# no tira no se porque
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Load once
#nltk.download('averaged_perceptron_tagger_eng')

lemmatizer = WordNetLemmatizer()

# Map NLTK POS → WordNet POS
def map_pos(tag):
    return {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN
    }.get(tag[0], None)

# Efficient lemmatization
def fast_lemmatize(words1):  # Chequear si esto funciona o no
    tagged = pos_tag(words)
    return [
        lemmatizer.lemmatize(word, map_pos(tag))
        for word, tag in tagged
        if map_pos(tag) is not None
    ]

# Apply
words = fast_lemmatize(words)


#%%
# Extended Stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Add custom words that appear in performance reviews and HR texts
extra = {
    "also", "always", "boss", "company", "could",
    "day", "days",
    "employee", "employees", "even",
    "feel", "felt",
    "get", "give", "given", "got",    "however",
    "job",    "need", "needs", "never",
    "make", "management", "manager", "many", "might", "much",
    "one",    "people", "put",    "really", "role",
    "should", "sometimes", "something", "someone", "still",
    "take", "team", "thing", "things", "think",
    "want", "work", "working", "would",
    "year", "years"
}


stop_words |= extra

# Apply stopwords
words = [w for w in words if w not in stop_words]


#%%
# Removing very short words

words = [w for w in words if 3 <= len(w) <= 15]


#%%
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
words = [word for word in words if word not in stop_words]

#custom_stopwords = ['ejemplo', 'texto']  # palabras que no quieres en tu WordCloud
#words = [word for word in words if word not in custom_stopwords]

#%%
# Merge again
clean_text = " ".join(words)
clean_text

#%%

from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(width=800, height=400, background_color='white',
                      max_words=200, colormap='viridis').generate(clean_text)

plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


#%%


# PRUEBAA SOLOP PRUEBA

from textblob import TextBlob

blob = TextBlob(clean_text)
sentiment = blob.sentiment

print("Polarity:", sentiment.polarity)   # -1 (negativo) a 1 (positivo)
print("Subjectivity:", sentiment.subjectivity)  # 0 (objetivo) a 1 (subjetivo)

#%%

from nltk.sentiment import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
scores = sia.polarity_scores(clean_text)

print(scores)
#%%

import matplotlib.pyplot as plt

# Etiquetas y valores
labels = ['Positive', 'Neutral', 'Negative']
values = [scores['pos'], scores['neu'], scores['neg']]

# Crear gráfico de barras
plt.figure(figsize=(6,4))
plt.bar(labels, values, color=['green', 'grey', 'red'])
plt.title('Sentiment Distribution (Overall)')
plt.ylabel('Proportion')
plt.ylim(0,1)  # Escala 0 a 1, porque VADER devuelve proporciones
plt.show()







#%%
# Perform sentiment analysis

# Only first time
# nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

#%%
comments = hrtext  # original text lines
#%%
sentiments = []

for comment in comments:
    score = sia.polarity_scores(comment)
    sentiments.append(score)

# Example of first 5
for i, s in enumerate(sentiments[:5]):
    print(f"Comment {i+1}: {s}")

#%%
def label_sentiment(compound):
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

sentiment_labels = [label_sentiment(s['compound']) for s in sentiments]

# Example: first 10
for comment, label in zip(comments[:10], sentiment_labels[:10]):
    print(f"{label}: {comment}")

#%%
import matplotlib.pyplot as plt

from collections import Counter
counter = Counter(sentiment_labels)

plt.bar(counter.keys(), counter.values(), color=['green','blue','red'])
plt.title("Sentiment Distribution of Employee Comments")
plt.show()
