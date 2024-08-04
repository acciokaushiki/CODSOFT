import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.offline import iplot
from plotly.subplots import make_subplots
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

# Load the Titanic dataset
df = pd.read_csv('Titanic-Dataset.csv')

# Explore the data
df.head()
df.shape
df.duplicated().sum()
df.isnull().sum()

# Handle missing values
df.groupby('Sex')['Age'].mean().reset_index()
df['Age'].fillna(df['Age'].mean(), inplace=True)
df.drop('Cabin', axis=1,inplace=True)
df.dropna(inplace=True)

# Data analysis and visualizations
df.info()
df.head()

# Visualize the gender distribution
gender = df['Sex'].value_counts()
plt.figure(figsize=(10,6))
plt.pie(gender, labels=['Male', 'Female'] ,autopct='%.1f%%', colors=['lightblue','pink'])
plt.legend()
plt.title('Male & Female')
plt.show()

# Visualize the age distribution
sns.histplot(data=df, x='Age', bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Analyze survival rates based on gender
sv_sex = df[['Survived','Sex']].value_counts().reset_index()
sv_sex
plt.figure(figsize=(5,6))
sns.barplot(data=sv_sex , x=sv_sex['Survived'], y=sv_sex['count'], hue=sv_sex['Sex'])
plt.title('Survived Frequency')
plt.xlabel('Survived')
plt.ylabel('Frequency')
plt.show()

# Analyze survival rates based on embarkation port
Em_sex = df[['Embarked', 'Sex']].value_counts().reset_index()
Em_sex
plt.figure(figsize=(7,6))
sns.barplot(data=Em_sex , x=Em_sex['Embarked'], y=Em_sex['count'], hue=Em_sex['Sex'])
plt.title('Embarked & Sex Frequency')
plt.xlabel('Embarked')
plt.ylabel('Frequency')
plt.show()

sv_em = df[['Survived', 'Embarked']].value_counts().reset_index()
sv_em
plt.figure(figsize=(8,6))
sns.barplot(data=sv_em , x=sv_em['Survived'], y=sv_em['count'], hue=sv_em['Embarked'])
plt.title('Survived & Embarked Frequency')
plt.xlabel('Survived')
plt.ylabel('Frequency')
plt.show()

# Analyze survival rates based on passenger class
sv_class = df[['Survived', 'Pclass']].value_counts().reset_index()
sv_class
plt.figure(figsize=(8,6))
sns.barplot(data=sv_class , x=sv_class['Survived'], y=sv_class['count'], hue=sv_class['Pclass'])
plt.title('Survived & Pclass Frequency')
plt.xlabel('Survived')
plt.ylabel('Frequency')
plt.show()

# Analyze the distribution of siblings/spouses
sibling = df['SibSp'].value_counts().reset_index()
sibling
plt.figure(figsize=(8,6))
sns.barplot(x=sibling['SibSp'], y=sibling['count'])
plt.title('Siblings or Spouses Frequancy')
plt.show()

# Analyze survival rates based on siblings/spouses
sv_sibling = df[['Survived', 'SibSp']].value_counts().reset_index()
sv_sibling
plt.figure(figsize=(8,6))
sns.barplot(data=sv_sibling , x=sv_sibling['Survived'], y=sv_sibling['count'], hue=sv_sibling['SibSp'])
plt.title('Survived & SibSp Frequency')
plt.legend(loc='upper right')
plt.xlabel('Survived')
plt.ylabel('Frequency')
plt.show()

# Prepare the data for modeling
test = df.drop(['PassengerId','Name','Ticket'], axis=1)
test

# Encode categorical features
label_encoder = LabelEncoder()
test['Sex'] = label_encoder.fit_transform(test['Sex'])
test['Embarked'] = label_encoder.fit_transform(test['Embarked'])
test

# Split the data into features and target variable
x= test.drop('Survived', axis=1)
y = test['Survived']

# Split the data into training and testing sets
X_train, x_test, Y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=11)

# Train a Logistic Regression model
model_logistic = LogisticRegression() 
model_logistic.fit(X_train,Y_train)
print(f"Logistic Regression Training Accuracy: {model_logistic.score(X_train,Y_train)}")
print(f"Logistic Regression Testing Accuracy: {model_logistic.score(x_test,y_test)}")

# Train a Random Forest model
model_random = RandomForestClassifier()
model_random.fit(X_train,Y_train)
print(f"Random Forest Training Accuracy: {model_random.score(X_train,Y_train)}")
print(f"Random Forest Testing Accuracy: {model_random.score(x_test,y_test)}")
