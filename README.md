# CodTech-Task2
# Movie Sentiment Analysis

This project performs sentiment analysis on movie reviews using a Naive Bayes classifier. It uses the IMDB dataset to train a model that can classify movie reviews as either positive or negative.

## Project Overview

This project involves:

- Loading and preprocessing the IMDB dataset.
  
- Vectorizing text data using `CountVectorizer`.

- Training a Naive Bayes classifier (`MultinomialNB`).

- Evaluating the model's performance.

- Using the trained model to classify new movie reviews.

## Requirements

- Python 3.x
- Pandas
- Scikit-learn

You can install the required Python packages using pip:

pip install pandas scikit-learn


#Dataset
The dataset used in this project is the IMDB movie review dataset, which can be downloaded from Kaggle or other similar sources. Make sure you have a CSV file named IMDB Dataset.csv in the same directory as the script.



Code Description
Import Libraries: Necessary libraries are imported, including pandas, scikit-learn modules, and random for reproducibility.

Load Dataset: The dataset is loaded and preprocessed (sentiment labels are converted to binary).

Split Data: The data is split into training and testing sets.

Vectorize Text: CountVectorizer is used to convert text data into numerical feature vectors.

Train Model: A Naive Bayes classifier is trained on the vectorized training data.

Evaluate Model: The classifier's accuracy and performance are evaluated on the test set.

Classify Reviews: A function is provided to classify new movie reviews.
