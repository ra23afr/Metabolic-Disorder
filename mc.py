import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE
import shap
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/Users/rajagrawal/Desktop/UNIVERSITY/MSC PROJECT/content/Metabolic_Syndrome.csv')

df

df.info()

df.isnull().sum()

df['Marital'] = df['Marital'].fillna(df['Marital'].mode()[0])
df['Income'] = df['Income'].fillna(df['Income'].mean())
df['WaistCirc'] = df['WaistCirc'].fillna(df['WaistCirc'].mean())
df['BMI'] = df['BMI'].fillna(df['BMI'].mean())

df.describe()

# EDA

# Distribution of Age
plt.figure(figsize=(12, 6))
sns.histplot(df['Age'], kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Countplot of Gender
plt.figure(figsize=(12, 6))
sns.countplot(x='Sex', data=df)
plt.title('Count of Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# Boxplot of BMI by Metabolic Syndrome status
plt.figure(figsize=(12, 6))
sns.boxplot(x='MetabolicSyndrome', y='BMI', data=df)
plt.title('BMI Distribution by Metabolic Syndrome Status')
plt.xlabel('Metabolic Syndrome')
plt.ylabel('BMI')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

# Distribution of Income
plt.figure(figsize=(12, 6))
sns.histplot(df['Income'], kde=True)
plt.title('Distribution of Income')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.show()

# Countplot of Marital Status
plt.figure(figsize=(12, 6))
sns.countplot(x='Marital', data=df)
plt.title('Count of Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.show()

# Pairplot of numerical features
numerical_cols = ['Age', 'WaistCirc', 'BMI', 'HDL', 'Triglycerides', 'Income']
sns.pairplot(df[numerical_cols + ['MetabolicSyndrome']], hue='MetabolicSyndrome', diag_kind='kde')
plt.suptitle('Pairwise Relationships of Numerical Features colored by Metabolic Syndrome', y=1.02)
plt.show()
