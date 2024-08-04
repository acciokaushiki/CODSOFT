import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

import plotly.express as px
from plotly.offline import init_notebook_mode, iplot, plot

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Read the CSV file
df = pd.read_csv("IMDb Movies India.csv", encoding='latin1')

# Display the first few rows of the DataFrame
df.head()

# Get the shape of the DataFrame (number of rows and columns)
df.shape

# Get the names of the columns
df.columns

# Get information about the DataFrame (data types, missing values, etc.)
df.info()

# Define a function to calculate the percentage of missing values in a DataFrame
def missing_values_percent(dataframe):
  # Calculate the number of missing values for each column
  missing_values = dataframe.isna().sum()
  
  # Calculate the percentage of missing values for each column
  percentage_missing = (missing_values / len(dataframe) * 100).round(2)

  # Create a DataFrame to store the results
  result_movie = pd.DataFrame({'Missing Values': missing_values, 'Percentage': percentage_missing})
  
  # Format the percentage column as strings with '%' sign
  result_movie['Percentage'] = result_movie['Percentage'].astype(str) + '%'

  # Return the DataFrame with missing values and their percentages
  return result_movie

# Calculate the percentage of missing values in the original DataFrame
result = missing_values_percent(df)
result

# Drop the 'Actor 2' and 'Actor 3' columns
df.drop(['Actor 2' , 'Actor 3'], axis=1, inplace=True)

# Drop rows with missing values in the 'Duration' column
df.dropna(subset=['Duration'], inplace = True)

# Keep rows with 5 or less missing values
df = df[df.isnull().sum(axis=1).sort_values(ascending=False) <=5]

# Check missing values after dropping rows with many missing values
missing_values_percent(df)

# Drop rows with missing values in the 'Rating' and 'Votes' columns
df.dropna(subset=['Rating', 'Votes'], inplace=True)

# Get descriptive statistics for the 'Director' column
director_description = df['Director'].describe()

# Count the occurrences of each director
director_counts = df['Director'].value_counts().sort_values(ascending=False)

# Fill missing values in the 'Director' column with 'rajmouli'
df['Director'].fillna('rajmouli', inplace=True)

# Count the occurrences of each genre
genre_counts = df['Genre'].value_counts().sort_values(ascending=False)

# Fill missing values in the 'Genre' column with 'Action'
df['Genre'].fillna('Action', inplace=True)

# Get descriptive statistics for the 'Actor 1' column
actor1_description = df['Actor 1'].describe()

# Fill missing values in the 'Actor 1' column with 'mahesh babu'
df['Actor 1'].fillna('mahesh babu', inplace=True)

# Create a DataFrame to store the missing values and their percentages
missing_values_df = pd.DataFrame({
    'Missing Values': df.isnull().sum(),
    'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})

# Display the last few rows of the DataFrame
df.tail()

# Check missing values after filling missing values
missing_values_percent(df)

# Remove parentheses from the 'Year' column
df['Year'] = df['Year'].str.replace(r'[()]', '', regex=True)

# Remove ' min' from the 'Duration' column
df['Duration'] = df['Duration'].str.replace(r' min', '', regex=True)

# Get information about the DataFrame after cleaning
df.info()

# Define a list of columns that should be integers
int_columns = ['Year', 'Duration']

# Convert the specified columns to integers
df[int_columns] = df[int_columns].astype(int)

# Remove commas from the 'Votes' column and convert it to integers
df['Votes'] = df['Votes'].str.replace(',', '').astype(int)

# Get information about the DataFrame after data type conversion
df.info()

# Create a figure for the plot
plt.figure(figsize=(20, 10))

# Count the number of movies released in each year
year_counts = df['Year'].value_counts().sort_index()

# Get the years from the index of the year counts
years = year_counts.index

# Plot the number of movies per year
plt.plot(years, year_counts, marker='o' )

# Set the title of the plot
plt.title('Number of Movies Per Year')

# Set the label for the x-axis
plt.xlabel('Year')

# Set the label for the y-axis
plt.ylabel('Number of Movies')

# Display the plot
plt.show()

# Get the labels (genres) and sizes (counts) for the pie chart
label = df["Genre"].value_counts().index
sizes = df["Genre"].value_counts()

# Create a figure for the pie chart
plt.figure(figsize = (10,10))

# Create the pie chart
plt.pie(sizes, labels= label, startangle = 0 , shadow = False , autopct='%1.1f%%')

# Display the pie chart
plt.show()

# Create a scatter plot of Rating versus Votes
px.scatter(df,x='Rating',y='Votes',color='Rating',color_continuous_scale=px.colors.sequential.Plasma,title='<b>Rating Versus Votes')

# Create a scatter plot of Rating versus Duration
px.scatter(df,x='Rating',y='Duration',color='Rating',color_continuous_scale=px.colors.sequential.Plasma,title='<b>Rating Versus Duration')

# Calculate the mean rating for each genre and create a new column
genre_mean_rating = df.groupby('Genre')['Rating'].transform('mean')
df['Genre_mean_rating'] = genre_mean_rating

# Calculate the mean rating for each director and create a new column
df['Director_encoded'] = df.groupby('Director')['Rating'].transform('mean')

# Calculate the mean rating for each actor and create a new column
df['Actor_encoded'] = df.groupby('Actor 1')['Rating'].transform('mean')

# Define the features and target variable
features = ['Year', 'Votes', 'Duration', 'Genre_mean_rating', 'Director_encoded', 'Actor_encoded']
X = df[features]
y = df['Rating']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")
