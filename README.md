🎵 Music Recommendation System using Machine Learning

This project is an MSc Artificial Intelligence and Robotics dissertation project at the University of Hertfordshire. The goal is to design and implement a machine learning–based music recommendation system that generates personalized playlists by analyzing user preferences and song attributes.

The system explores multiple machine learning models and evaluates their effectiveness in predicting song popularity and recommending tracks based on audio features, genre, and historical listening behaviour.

🚀 Features

Pre-processing and feature extraction from large-scale music datasets (Kaggle).

Exploratory Data Analysis (EDA) to identify patterns and correlations in musical attributes.

Clustering of music genres using K-Means and dimensionality reduction with t-SNE and PCA.

Implementation and comparison of ML algorithms:

Logistic Regression (76.7% accuracy)

Random Forest (82.7% accuracy – best performer)

Support Vector Machines (81.0% accuracy)

KNN (79.6% accuracy)

Gradient Boosting (80.4% accuracy)

XGBoost (82.0% accuracy)

Basic rule-based recommendation (by artist, genre, mood, or year).

Identification of limitations such as class imbalance and cold-start problems, with discussion of solutions like SMOTE and class weighting.

📊 Dataset

The project uses Kaggle music datasets, including:

data.csv → General track-level features (danceability, energy, valence, tempo, loudness, etc.).

data_by_genres.csv → Aggregated features by genre.

data_by_year.csv → Features grouped by release year.

data_by_artist.csv → Features grouped by artist.

Attributes include both audio features (e.g., tempo, acousticness, energy, instrumentalness) and metadata (year, explicit lyrics, popularity score).

🛠️ Tech Stack

Python

Libraries: numpy, pandas, matplotlib, seaborn, plotly, scikit-learn, yellowbrick

ML Models: Logistic Regression, Random Forest, SVM, KNN, Gradient Boosting, XGBoost

Dimensionality Reduction: PCA, t-SNE

Clustering: K-Means

📈 Results

Random Forest achieved the highest classification accuracy (82.7%).

XGBoost and SVM also performed well, with accuracies above 81%.

Logistic Regression was the weakest performer due to class imbalance issues.

Clustering revealed meaningful groupings of genres, showing potential for recommendation enhancement.
