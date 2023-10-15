import numpy as np
import pandas as pd
import seaborn as sb
import plotly.express as px
import matplotlib.pyplot as mpl
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split

# Load the dataset
movie_file = pd.read_csv("C:\\Users\\shiva\\OneDrive\\Desktop\\CodSoft\\IMDb Movies India.csv", encoding='latin1')

# Display the first 11 rows of the dataset
a = movie_file.head(11)
print(a)

# Display summary statistics
b = movie_file.describe()
print(b)

# Display data types of columns
c = movie_file.dtypes
print(c)

# Check for missing values
d = movie_file.isnull().sum()
print(d)

# Calculate the total number of missing values in the dataset
e = movie_file.isnull().sum().sum()
print(e)

# Get the shape of the dataset
f = movie_file.shape
print(f)

# Drop rows with missing values
g = movie_file.dropna(inplace=True)
print(g)

# Display the first 11 rows after dropping missing values
h = movie_file.head(11)
print(h)

# Clean the 'Year' column by removing non-numeric characters
movie_file['Year'] = movie_file['Year'].str.extract('(\d+)').astype(int)

# Display the 'Year' column
movie_file["Year"].head()

# Extract and display the 'Genre' column
genre = movie_file['Genre']
genre.head(11)

# Split the 'Genre' column into multiple genres and display the first 11 rows
genres = movie_file['Genre'].str.split(', ', expand=True)
genres.head(11)

# Count the occurrences of each genre
genre_counts = {}
for genre in genres.values.flatten():
    if genre is not None:
        if genre in genre_counts:
            genre_counts[genre] += 1
        else:
            genre_counts[genre] = 1

# Sort the genre counts
genereCounts = {genre: count for genre, count in sorted(genre_counts.items())}

# Display genre counts
for genre, count in genereCounts.items():
    print(f"{genre}: {count}")

# Count the number of movies in each genre
genresPie = movie_file['Genre'].value_counts()
genresPie.head(11)

# Create a DataFrame for genre counts
genrePie = pd.DataFrame(list(genresPie.items()))
genrePie = genrePie.rename(columns={0: 'Genre', 1: 'Count'})
genrePie.head(11)

# Clean the 'Votes' column by removing commas and converting it to an integer
movie_file['Votes'] = movie_file['Votes'].str.replace(',', '').astype(int)

# Display the 'Votes' column
movie_file['Votes'].head(11)

# Get the number of unique directors
movie_file["Director"].nunique()

# Count the number of movies directed by each director
directors = movie_file["Director"].value_counts()
directors.head(11)

# Concatenate and count the occurrences of actors from 'Actor 1', 'Actor 2', and 'Actor 3'
actors = pd.concat([movie_file['Actor 1'], movie_file['Actor 2'], movie_file['Actor 3']]).dropna().value_counts()
actors.head(11)

# Set the style and font for seaborn
sb.set(style="darkgrid", font="Calibri")

# Create a box plot for the 'Year' column
ax = sb.boxplot(data=movie_file, y='Year')
ax.set_ylabel('Year')
ax.set_title('Box Plot of Year')
mpl.show()

# Calculate average movie duration over the years and create a line plot
ax = sb.lineplot(data=movie_file.groupby('Year')['Duration'].mean().reset_index(), x='Year', y='Duration')
darkgrid_positions = range(min(movie_file['Year']), max(movie_file['Year']) + 1, 5)
ax.set_title("Average Movie Duration Trends Over the Years")
ax.set_xticks(darkgrid_positions)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_xlabel("Years")
ax.set_ylabel('Average Duration (in minutes)')
mpl.show()

# Create a box plot for the 'Duration' column
ax = sb.boxplot(data=movie_file, y='Duration')
ax.set_title("Box Plot of Average Movie Durations")
ax.set_ylabel('Average Duration (in minutes)')
mpl.show()

# Calculate quartiles and the IQR for the 'Duration' column
Q1 = movie_file['Duration'].quantile(0.25)
Q3 = movie_file['Duration'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the dataset to remove outliers in 'Duration'
df = movie_file[(movie_file['Duration'] >= lower_bound) & (movie_file['Duration'] <= upper_bound)]
df.head(11)

# Count the occurrences of each genre in the stacked genre data
genre_counts = movie_file['Genre'].str.split(', ', expand=True).stack().value_counts()

# Generate a word cloud based on genre counts
wordcloud = WordCloud(width=950, height=550, background_color='white').generate_from_frequencies(genre_counts)

# Display the genre word cloud
mpl.figure(figsize=(16, 6))
mpl.imshow(wordcloud, interpolation='bilinear')
mpl.axis('off')
mpl.title('Genre Word Cloud')
mpl.show()

# Group genres with counts less than 50 into 'Other'
genrePie.loc[genrePie['Count'] < 50, 'Genre'] = 'Other'

# Create a pie chart for genre distribution
ax = px.pie(genrePie, values='Count', names='Genre', title='More than one Genre of movies in Indian Cinema')
ax.show()

# Create a histogram for the 'Rating' column
ax = sb.histplot(data=movie_file, x="Rating", bins=20, kde=True)
ax.set_xlabel('Rating')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Movie Ratings')
mpl.show()

# Create a box plot for the 'Rating' column
ax = sb.boxplot(data=movie_file, y='Rating')
ax.set_ylabel('Rating')
ax.set_title('Box Plot of Movie Ratings')
mpl.show()

# Calculate quartiles and the IQR for the 'Rating' column
Q1 = movie_file['Rating'].quantile(0.25)
Q3 = movie_file['Rating'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the dataset to remove outliers in 'Rating'
movie_file = movie_file[(movie_file['Rating'] >= lower_bound) & (movie_file['Rating'] <= upper_bound)]
movie_file.head(16)

# Group and sum the 'Votes' column by 'Rating'
rating_votes = movie_file.groupby('Rating')['Votes'].sum().reset_index()

# Create a line plot for total votes per rating
mpl.figure(figsize=(10, 6))
ax_line_seaborn = sb.lineplot(data=rating_votes, x='Rating', y='Votes', marker='o')
ax_line_seaborn.set_xlabel('Rating')
ax_line_seaborn.set_ylabel('Total Votes')
ax_line_seaborn.set_title('Total Votes per Rating')
mpl.show()

# Create a bar plot for the top 20 directors by frequency of movies
mpl.figure(figsize=(10, 6))
ax = sb.barplot(x=directors.head(20).index, y=directors.head(20).values, palette='viridis')
ax.set_xlabel('Directors')
ax.set_ylabel('Frequency of Movies')
ax.set_title('Top 20 Directors by Frequency of Movies')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
mpl.show()

# Create a bar plot for the top 20 actors with the total number of movies
mpl.figure(figsize=(10, 6))
ax = sb.barplot(x=actors.head(20).index, y=actors.head(20).values, palette='viridis')
ax.set_xlabel('Actors')
ax.set_ylabel('Total Number of Movies')
ax.set_title('Top 20 Actors with Total Number of Movies')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
mpl.show()

# Create a new 'Actor' column by concatenating 'Actor 1', 'Actor 2', and 'Actor 3'
movie_file["Actor"] = movie_file['Actor 1'] + ', ' + movie_file['Actor 2'] + ', ' + movie_file['Actor 3']

# Convert categorical features to numeric codes
movie_file["Directors"] = movie_file['Director'].astype('category').cat.codes
movie_file["Genres"] = movie_file['Genre'].astype('category').cat.codes
movie_file["Actors"] = movie_file['Actor'].astype('category').cat.codes

# Create a box plot for the 'Genres' column
ax = sb.boxplot(data=movie_file, y='Genres')
ax.set_ylabel('Genres')
ax.set_title('Box Plot of Genres')
mpl.show()

# Calculate quartiles and the IQR for the 'Genres' column
Q1 = movie_file['Genres'].quantile(0.25)
Q3 = movie_file['Genres'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the dataset to remove outliers in 'Genres'
movie_file = movie_file[(movie_file['Genres'] >= lower_bound) & (movie_file['Genres'] <= upper_bound)]

# Create a box plot for the 'Directors' column
ax = sb.boxplot(data=movie_file, y='Directors')
ax.set_ylabel('Directors')
ax.set_title('Box Plot of Directors')
mpl.show()

# Calculate quartiles and the IQR for the 'Directors' column
Q1 = movie_file['Directors'].quantile(0.25)
Q3 = movie_file['Directors'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the dataset to remove outliers in 'Directors'
movie_file = movie_file[(movie_file['Directors'] >= lower_bound) & (movie_file['Directors'] <= upper_bound)]

# Create a box plot for the 'Actors' column
ax = sb.boxplot(data=movie_file, y='Actors')
ax.set_ylabel('Actors')
ax.set_title('Box Plot of Actors')
mpl.show()

# Calculate quartiles and the IQR for the 'Actors' column
Q1 = movie_file['Actors'].quantile(0.25)
Q3 = movie_file['Actors'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the dataset to remove outliers in 'Actors'
movie_file = movie_file[(movie_file['Actors'] >= lower_bound) & (movie_file['Actors'] <= upper_bound)]

# Drop irrelevant columns
Input = movie_file.drop(['Name', 'Genre', 'Rating', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Actor'], axis=1)

# Define the target variable
Output = movie_file['Rating']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(Input, Output, test_size=0.2, random_state=1)

# Import necessary regression models and evaluation metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score as score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

# Instantiate and fit regression models
LR = LinearRegression()
LR.fit(x_train, y_train)
lr_preds = LR.predict(x_test)

RFR = RandomForestRegressor(n_estimators=100, random_state=1)
RFR.fit(x_train, y_train)
rf_preds = RFR.predict(x_test)

DTR = DecisionTreeRegressor(random_state=1)
DTR.fit(x_train, y_train)
dt_preds = DTR.predict(x_test)

XGBR = XGBRegressor(n_estimators=100, random_state=1)
XGBR.fit(x_train, y_train)
xgb_preds = XGBR.predict(x_test)

GBR = GradientBoostingRegressor(n_estimators=100, random_state=60)
GBR.fit(x_train, y_train)
gb_preds = GBR.predict(x_test)

LGBMR = LGBMRegressor(n_estimators=100, random_state=60)
LGBMR.fit(x_train, y_train)
lgbm_preds = LGBMR.predict(x_test)

CBR = CatBoostRegressor(n_estimators=100, random_state=1, verbose=False)
CBR.fit(x_train, y_train)
catboost_preds = CBR.predict(x_test)

KNR = KNeighborsRegressor(n_neighbors=5)
KNR.fit(x_train, y_train)
knn_preds = KNR.predict(x_test)

# Evaluate and compare the performance of the models
def evaluate_model(y_true, y_pred, model_name):
    print("Model: ", model_name)
    print("Accuracy = {:0.2f}%".format(score(y_true, y_pred) * 100))
    print("Root Mean Squared Error = {:0.2f}\n".format(mean_squared_error(y_true, y_pred, squared=False)))
    return round(score(y_true, y_pred) * 100, 2)

LRScore = evaluate_model(y_test, lr_preds, "Linear Regression")
RFScore = evaluate_model(y_test, rf_preds, "Random Forest")
DTScore = evaluate_model(y_test, dt_preds, "Decision Tree")
XGBScore = evaluate_model(y_test, xgb_preds, "XGBoost")
GBScore = evaluate_model(y_test, gb_preds, "Gradient Boosting")
LGBScore = evaluate_model(y_test, lgbm_preds, "LightGBM")
CBRScore = evaluate_model(y_test, catboost_preds, "CatBoost")
KNNScore = evaluate_model(y_test, knn_preds, "K Nearest Neighbors")

# Create a DataFrame to compare the model scores
models = pd.DataFrame(
    {
        "MODELS": ["Linear Regression", "Random Forest", "Decision Tree", "XGBoost", "Gradient Boosting", "LightGBM", "CatBoost", "K Nearest Neighbors"],
        "SCORES": [LRScore, RFScore, DTScore, XGBScore, GBScore, LGBScore, CBRScore, KNNScore]
    }
)

# Sort models by score in descending order
models.sort_values(by='SCORES', ascending=False)
