# Import necessary libraries
import os  # For interacting with the operating system (e.g., file paths)
import numpy as np  # For numerical operations and array manipulations
import pandas as pd  # For data manipulation and analysis

import seaborn as sns  # For statistical data visualization
import plotly.express as px  # For interactive visualizations
import matplotlib.pyplot as plt  # For plotting static graphs

from sklearn.cluster import KMeans  # For K-Means clustering
from sklearn.preprocessing import StandardScaler  # For standardizing features
from sklearn.pipeline import Pipeline  # For creating machine learning pipelines
from sklearn.manifold import TSNE  # For dimensionality reduction using t-SNE
from sklearn.decomposition import PCA  # For dimensionality reduction using PCA
from sklearn.metrics import euclidean_distances  # For calculating Euclidean distances
from scipy.spatial.distance import cdist  # For calculating distance between sets of points

import warnings  # For handling warnings
warnings.filterwarnings("ignore")  # Ignores warnings to keep the output clean
# Load data from CSV files into DataFrames
data = pd.read_csv("data.csv")  # Load the main dataset, assuming it contains general music data
genre_data = pd.read_csv('data_by_genres.csv')  # Load dataset containing music data categorized by genre
year_data = pd.read_csv('data_by_year.csv')  # Load dataset containing music data categorized by year
artist_data = pd.read_csv('data_by_artist.csv')  # Load dataset containing music data categorized by artist
print(data.info())
print(genre_data.info())
print(year_data.info())
print(artist_data.info())

import numpy as np
import matplotlib.pyplot as plt
from yellowbrick.target import FeatureCorrelation

# List of feature names to include in the analysis
feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                  'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
                  'duration_ms', 'explicit', 'key', 'mode', 'year']

# Prepare the feature matrix (X) and target vector (y) from the dataset
X, y = data[feature_names], data['popularity']

# Convert the list of feature names to a numpy array
features = np.array(feature_names)

# Instantiate the FeatureCorrelation visualizer from Yellowbrick
visualizer = FeatureCorrelation(labels=features)

# Set the size of the figure for better visibility
plt.rcParams['figure.figsize'] = (20, 20)

# Fit the visualizer to the feature matrix (X) and target vector (y)
visualizer.fit(X, y)

# Display the visualized correlation between features and the target variable
visualizer.show()

# Define the sound features to plot
sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']

# Create a line plot using Plotly Express
# 'year_data' should be a DataFrame with columns 'year' and the sound features defined above
fig = px.line(year_data, x='year', y=sound_features,
              labels={'year': 'Year', 'value': 'Sound Feature Value', 'variable': 'Sound Feature'},
              title='Sound Features Over Time')

# Display the plot
fig.show()

top10_genres = genre_data.nlargest(10, 'popularity')

fig = px.bar(top10_genres, x='genres', y=['valence', 'energy', 'danceability', 'acousticness'], barmode='group')
fig.show()

import matplotlib.pyplot as plt  # Import matplotlib for plotting
import seaborn as sns            # Import seaborn for advanced plotting

# Create a figure with specific size
plt.figure(figsize=(10, 6))
# Create a histogram plot of song popularity with KDE (Kernel Density Estimate)
sns.histplot(data=artist_data, x='popularity', bins=30, kde=True, color='blue')

# Set the title of the plot
plt.title('Distribution of Song Popularity')
# Set the label for the x-axis
plt.xlabel('Popularity')
# Set the label for the y-axis
plt.ylabel('Frequency')
# Add grid lines for better readability
plt.grid(True)

# Display the plot
plt.show()

# Select numerical columns for correlation analysis
# Define a list of numerical features that you want to include in the correlation analysis.
numerical_features = ['acousticness', 'danceability', 'duration_ms', 'energy',
                       'instrumentalness', 'liveness', 'loudness', 'speechiness',
                       'tempo', 'valence', 'popularity']

# Compute the correlation matrix
# Calculate the Pearson correlation coefficients between the numerical features.
correlation_matrix = artist_data[numerical_features].corr()

# Plot the heatmap
# Set the size of the figure for better visualization.
plt.figure(figsize=(12, 8))

# Create a heatmap of the correlation matrix.
# `annot=True` adds the correlation coefficient values to the heatmap.
# `cmap='coolwarm'` defines the color map, which ranges from blue (negative) to red (positive).
# `fmt='.2f'` specifies the format for the annotation text (2 decimal places).
# `vmin=-1` and `vmax=1` set the range of the color scale.
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)

# Add a title to the heatmap for context.
plt.title('Correlation Matrix of Audio Features')

# Display the plot.
plt.show()

\from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

# Initialize the pipeline for clustering
cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Scale features to have mean=0 and variance=1
    ('kmeans', KMeans(n_clusters=10, n_init=10))  # Step 2: Apply KMeans clustering with 10 clusters
    # n_init: Number of time the KMeans algorithm will be run with different centroid seeds
    # n_jobs is deprecated, use n_init instead
])

# Select only numerical columns from genre_data for clustering
X = genre_data.select_dtypes(np.number)

# Fit the pipeline on the data
cluster_pipeline.fit(X)  # This scales the data and then performs KMeans clustering

# Predict cluster labels for the data and add them to the DataFrame
genre_data['cluster'] = cluster_pipeline.predict(X)  # Assign the cluster labels to each row in genre_data

# Import necessary libraries
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.express as px

# Create a pipeline for t-SNE with data scaling
# This pipeline will first standardize the data and then apply t-SNE for dimensionality reduction
tsne_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Standardize the features
    ('tsne', TSNE(n_components=2, verbose=1))  # Step 2: Apply t-SNE to reduce to 2D
])

# Transform the feature data using the pipeline
# This will fit the StandardScaler, then fit and transform the data using t-SNE
genre_embedding = tsne_pipeline.fit_transform(X)

# Create a DataFrame to hold the 2D t-SNE projection
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)  # Columns for t-SNE output
projection['genres'] = genre_data['genres']  # Add genre information
projection['cluster'] = genre_data['cluster']  # Add cluster information

# Create a scatter plot using Plotly Express
# The plot will show the t-SNE projection with clusters differentiated by color
fig = px.scatter(
    projection,  # Data to plot
    x='x',  # X-axis values (from t-SNE)
    y='y',  # Y-axis values (from t-SNE)
    color='cluster',  # Color points by cluster assignment
    hover_data=['x', 'y', 'genres']  # Data to display when hovering over points
)
# Show the interactive plot
fig.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Define the target variable and features
threshold = 50  # Popularity threshold
artist_data['popularity_class'] = (artist_data['popularity'] > threshold).astype(int)

# Features and target
X = artist_data[['acousticness', 'danceability', 'duration_ms', 'energy',
              'instrumentalness', 'liveness', 'loudness', 'speechiness',
              'tempo', 'valence']]
y = artist_data['popularity_class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

# Initialize and train the model
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)

# Make predictions and evaluate the model
y_pred_log_reg = log_reg.predict(X_test_scaled)
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print(classification_report(y_test, y_pred_log_reg))

from sklearn.ensemble import RandomForestClassifier

# Initialize and train the model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred_rf = rf_clf.predict(X_test)
print("Random Forest:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

from sklearn.svm import SVC

# Initialize and train the model
svm_clf = SVC()
svm_clf.fit(X_train_scaled, y_train)

# Make predictions and evaluate the model
y_pred_svm = svm_clf.predict(X_test_scaled)
print("Support Vector Machine:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
results = {
    'Model': [],
    'Accuracy': []
}

gb_clf = GradientBoostingClassifier(random_state=42)
gb_clf.fit(X_train_scaled, y_train)
y_pred_gb = gb_clf.predict(X_test_scaled)
gb_accuracy = accuracy_score(y_test, y_pred_gb)
results['Model'].append('Gradient Boosting')
results['Accuracy'].append(gb_accuracy)
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_scaled, y_train)
y_pred_knn = knn_clf.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
results['Model'].append('K-Nearest Neighbors')
results['Accuracy'].append(knn_accuracy)

import xgboost as xgb

xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_clf.predict(X_test_scaled)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
results['Model'].append('XGBoost')
results['Accuracy'].append(xgb_accuracy)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Initialize dictionaries to store results
results = {
    'Model': [],
    'Accuracy': []
}

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
results['Model'].append('Logistic Regression')
results['Accuracy'].append(log_reg_accuracy)

# Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
results['Model'].append('Random Forest')
results['Accuracy'].append(rf_accuracy)

# Support Vector Machine (SVM)
svm_clf = SVC()
svm_clf.fit(X_train_scaled, y_train)
y_pred_svm = svm_clf.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
results['Model'].append('Support Vector Machine')
results['Accuracy'].append(svm_accuracy)

# K-Nearest Neighbors (KNN)
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_scaled, y_train)
y_pred_knn = knn_clf.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
results['Model'].append('K-Nearest Neighbors')
results['Accuracy'].append(knn_accuracy)

# Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(random_state=42)
gb_clf.fit(X_train_scaled, y_train)
y_pred_gb = gb_clf.predict(X_test_scaled)
gb_accuracy = accuracy_score(y_test, y_pred_gb)
results['Model'].append('Gradient Boosting')
results['Accuracy'].append(gb_accuracy)

# XGBoost Classifier
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_clf.predict(X_test_scaled)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
results['Model'].append('XGBoost')
results['Accuracy'].append(xgb_accuracy)
# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Plot the model comparison
plt.figure(figsize=(12, 6))
bars = plt.bar(results_df['Model'], results_df['Accuracy'], color=['blue', 'green', 'red', 'purple', 'orange', 'brown'])

# Add value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 2), ha='center', va='bottom')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Define the clustering pipeline
song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Standardize the features to have mean=0 and variance=1
    ('kmeans', KMeans(n_clusters=20,  # Step 2: Apply KMeans clustering with 20 clusters
                       verbose=False,  # Turn off verbose output to avoid clutter
                       n_init=4))  # Number of initializations of KMeans to ensure better convergence
], verbose=False)  # Turn off verbose output for the entire pipeline

# Select numerical columns from the dataset for clustering
X = data.select_dtypes(np.number)  # Select columns with numerical data types

# List of numerical feature column names
number_cols = list(X.columns)  # Get the list of numerical feature names

# Fit the clustering pipeline to the numerical data
song_cluster_pipeline.fit(X)  # Standardize the data and then apply KMeans clustering

# Predict cluster labels for each sample in the dataset
song_cluster_labels = song_cluster_pipeline.predict(X)  # Assign each sample to one of the 20 clusters

# Add the cluster labels to the original dataset
data['cluster_label'] = song_cluster_labels  # Create a new column in the dataset to store the cluster labels


# Import necessary libraries for PCA and visualization
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create a pipeline with StandardScaler and PCA
pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features to mean 0 and variance 1
    ('PCA', PCA(n_components=2))    # Reduce dimensionality to 2 components for visualization
])

# Fit and transform the feature matrix X to get 2D projections
song_embedding = pca_pipeline.fit_transform(X)

# Create a DataFrame for the PCA projection
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']  # Add song names to the DataFrame
projection['cluster'] = data['cluster_label']  # Add cluster labels to the DataFrame

# Create an interactive scatter plot using Plotly Express
fig = px.scatter(
    projection,              # DataFrame containing PCA projections and additional data
    x='x',                   # X-axis represents the first principal component
    y='y',                   # Y-axis represents the second principal component
    color='cluster',         # Points are colored based on their cluster label
    hover_data=['x', 'y', 'title']  # Display x, y coordinates and song title on hover
)

# Show the interactive scatter plot
fig.show()

import pandas as pd
import ipywidgets as widgets
from IPython.display import display

# Sample data of songs
songs_data = [
    {"name": "Blinding Lights", "artist": "The Weeknd", "year": 2019, "mood": "happy"},
    {"name": "Shape of You", "artist": "Ed Sheeran", "year": 2017, "mood": "happy"},
    {"name": "Someone Like You", "artist": "Adele", "year": 2011, "mood": "sad"},
    {"name": "Rolling in the Deep", "artist": "Adele", "year": 2010, "mood": "sad"},
    {"name": "Uptown Funk", "artist": "Mark Ronson ft. Bruno Mars", "year": 2014, "mood": "happy"},
    {"name": "Levitating", "artist": "Dua Lipa", "year": 2020, "mood": "happy"},
    {"name": "Someone You Loved", "artist": "Lewis Capaldi", "year": 2018, "mood": "sad"},
    {"name": "Bad Guy", "artist": "Billie Eilish", "year": 2019, "mood": "happy"},
    {"name": "Happier", "artist": "Marshmello ft. Bastille", "year": 2018, "mood": "happy"},
    {"name": "The Night We Met", "artist": "Lord Huron", "year": 2015, "mood": "sad"},
    {"name": "Watermelon Sugar", "artist": "Harry Styles", "year": 2019, "mood": "happy"},
    {"name": "Perfect", "artist": "Ed Sheeran", "year": 2017, "mood": "happy"},
    {"name": "When I Was Your Man", "artist": "Bruno Mars", "year": 2012, "mood": "sad"},
    {"name": "Dance Monkey", "artist": "Tones and I", "year": 2019, "mood": "happy"},
    {"name": "Hello", "artist": "Adele", "year": 2015, "mood": "sad"},
    {"name": "Shallow", "artist": "Lady Gaga & Bradley Cooper", "year": 2018, "mood": "sad"},
    {"name": "Starboy", "artist": "The Weeknd", "year": 2016, "mood": "happy"},
    {"name": "Good 4 U", "artist": "Olivia Rodrigo", "year": 2021, "mood": "happy"},
    {"name": "Stay", "artist": "The Kid LAROI & Justin Bieber", "year": 2021, "mood": "happy"},
    {"name": "All I Want", "artist": "Olivia Rodrigo", "year": 2021, "mood": "sad"}
]

 Convert the list of dictionaries to a DataFrame
songs_df = pd.DataFrame(songs_data)

# Function to recommend songs based on search criteria
def recommend_songs(criteria, value):
    # Search for songs based on criteria
    if criteria == 'year':
        recommendations = songs_df[songs_df['year'] == value]
    elif criteria == 'artist':
        recommendations = songs_df[songs_df['artist'].str.contains(value, case=False, na=False)]
    elif criteria == 'name':
        recommendations = songs_df[songs_df['name'].str.contains(value, case=False, na=False)]
    elif criteria == 'mood':
        recommendations = songs_df[songs_df['mood'].str.contains(value, case=False, na=False)]
    else:
        return "Invalid search criteria. Please use 'year', 'artist', 'name', or 'mood'."

    # Return the recommendations
    return recommendations if not recommendations.empty else "No songs found."

# Create widgets for user inputs
criteria_dropdown = widgets.Dropdown(
    options=['year', 'artist', 'name', 'mood'],
    description='Criteria:',
)

value_textbox = widgets.Text(
    description='Value:',
)

submit_button = widgets.Button(description="Submit")

output = widgets.Output()

# Define the function to be called on button click
def on_submit_button_clicked(b):
    with output:
        output.clear_output()
        criteria = criteria_dropdown.value
        value = value_textbox.value
        # Handle numerical input for year
        if criteria == 'year' and value.isdigit():
            value = int(value)
        # Display recommendations
        recommendations = recommend_songs(criteria, value)
        print(recommendations)

# Attach the button click event to the function
submit_button.on_click(on_submit_button_clicked)

# Display widgets
display(criteria_dropdown, value_textbox, submit_button, output)

