# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load the Iris dataset
df = pd.read_csv('IRIS.csv')

# Explore the dataset
print(df)
print(df.head())
print(df.info())
print(df.describe())
print(df.describe(include='object'))
print(df.isna().sum())

# Analyze the distribution of species
count_iris = df['species'].value_counts()
print(count_iris)

# Visualize the distribution of species
plt.figure(figsize=(20, 10))
explode = (0, 0, 0.09)
count_iris.plot(kind='pie', fontsize=12, explode=explode, autopct='%.1f%%')
plt.title('species')
plt.xlabel('species', weight="bold", color="#000000", fontsize=14, labelpad=20)
plt.ylabel('counts', weight="bold", color="#000000", fontsize=14, labelpad=20)
plt.legend(labels=count_iris.index, loc="best")
plt.show()

# Visualize the distribution of each feature
df['sepal_length'].hist(figsize=(12, 8))
df['sepal_width'].hist(figsize=(12, 8))
df['petal_length'].hist(figsize=(12, 8))
df['petal_width'].hist(figsize=(12, 8))

# Analyze the relationship between species and sepal length
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=df['species'], y=df['sepal_length'], palette='viridis')
ax.bar_label(ax.containers[0], fontsize=10)
plt.title('Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length')
plt.show()

# Analyze the relationship between species and sepal width
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=df['species'], y=df['sepal_width'], palette='viridis')
ax.bar_label(ax.containers[0], fontsize=10)
plt.title('Sepal Width by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Width')
plt.show()

# Analyze the relationship between species and petal length
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=df['species'], y=df['petal_length'], palette='viridis')
ax.bar_label(ax.containers[0], fontsize=10)
plt.title('Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length')
plt.show()

# Analyze the relationship between species and petal width
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=df['species'], y=df['petal_width'], palette='viridis')
ax.bar_label(ax.containers[0], fontsize=10)
plt.title('Petal Width by Species')
plt.xlabel('Species')
plt.ylabel('Petal Width')
plt.show()

# Encode categorical features
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])
print(df)

# Analyze the correlation between features
plt.figure(figsize=(22, 18))
sns.heatmap(df.corr(), annot=True, fmt=".2f", linewidths=0.5, cbar=True)
plt.show()

# Prepare data for model training
x = df.drop(columns=['species'])
y = df['species']
y = y.values.reshape(-1, 1)
print(y)

# Scale features
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)

# Train a Decision Tree Classifier model
Dt = DecisionTreeClassifier()
Dt.fit(X_train, y_train)

# Evaluate model performance on training data
print(Dt.score(X_train, y_train))

# Make predictions on test data
y_pred = Dt.predict(X_test)

# Evaluate model performance on test data
print(accuracy_score(y_test, y_pred))

# Visualize the confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))

# Print classification report
print(classification_report(y_test, y_pred))
