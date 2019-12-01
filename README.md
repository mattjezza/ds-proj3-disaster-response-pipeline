# Disaster Response Pipeline Project

## Description
This is a project for Udacity's Data Science Nanodegree course (https://www.udacity.com/course/data-scientist-nanodegree--nd025), specifically the Data Engineering section. The aim of the project is to build a system to classify messages sent during disaster events so that they can be automatically directed to the right relief organisation. These messages could be concerned with essential requirements for survival in an emergency, such as food and water, so directing them to the right place quickly is vitally important.

## Data
The data set is supplied by Figure Eight (https://www.figure-eight.com/) and consists of messages sent during a disaster and their pre-labelled categories. The data is supplied as two csv files: messages.csv and categories.csv.

## Code
The code consists of three main parts.

### data/process_data.py
This is a python script that reads messages.csv and categories.csv, cleans the data and saves it as an SQL data base called DisasterResponse.db.

### models/train_classifier.py
This is a python script that reads the data in from DisasterReponse.db file, prepares it, then creates a machine learning pipeline using XGBoost and trains this using GridSearchCV to find the optimum parameters. It then tests this model with unseen data and outputs data about the performance (precision, recall, f1 score, accuracy) for each category. Finally it saves the trained model as a pickle file (classifier.pkl).

### app/run.py
This code is supplied entirely by Udacity with only minimal changes. It runs a web app which displays some visualisations about the data and uses the trained model to classify messages input manually via the user interface.

## Dependencies
In addition to the common data science packages used by Udacity, this code uses the XGBoost classifier. This can be installed in conda using:
conda install -c conda-forge xgboost

scikit-learn version 0.20 or above is also required. To install:
conda install scikit-learn=0.20

## Running Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## User Interface and Usage

### User Interface to Classifier
The web app provides a user interface for classifying messages. The user can manually enter a message and the trained model will classify it automatically.
The interface is shown below:

![User interface for classifier](msg_clf.png?raw=true "User interface for classifier")

### Data Visualisations
The web app also outputs three data visualisations:

1. Distribution of messages by genre (i.e., social, news or direct). This is the example provided in the Udacity course materials.

2. Distribution of messages by category.

3. The recall value of the positive case for each category obtained from the test data when the classifier was trained. This is probably the most important metric for evaluating the performance of the model.

The three visualisations are shown below.

![Data visualisations](visuals.png?raw=true "Data visualisations")

## Discussion of Data and Model Performance
The data set is unbalanced, i.e. for some categories there are very few examples where the value is 1 (indicating it is related to the category). The 'child\_alone' category has no examples of value 1. This makes training the data set difficult and affects the model performance.

Specifically, the model performs quite well at identifying the negative (i.e., category value=0) cases for all categories, with high precision, recall and f1 scores. However, for the category value=1 cases, especially those where there are few examples in the training set, precision, recall and f1 are much worse. Recall is the parameter of most interest in this application, becasue we do not want to miss any related examples, even if that means lower precision. In the offer category, for example, offer=1 has a recall of 0. This is because there are few examples of offer=1 in the training set, so the model has learned to always categorise messages as offer=0.

One possible way to help would be to remove the child\_alone category altogether, since the classifier cannot possibly train to classify this category with no data points, and use data stratification (using the stratify parameter in sklearn.model\_selection.train\_test_split()) to create a more evenly balanced set of training and testing data.

Since many of the relevant messages would probably include a location, it may be useful to modify the pipeline to include a custom transformer identifying messages that contain a location. This could be passed to the model as an additional parameter to train on using scikit-learn's FeatureUnion(). 

Another, very different, approach to this problem would be to use a deep learning model. This could use word2vec and feed this to a neural network to classify the messages.

