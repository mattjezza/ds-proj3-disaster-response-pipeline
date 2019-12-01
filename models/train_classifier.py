import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pickle as pkl


def load_data(database_filepath):
    """
    Read data from database.
    :param database_filepath: Relative path to database containing cleaned
    message data.
    :return:
    X: Messages.
    y: Category values for messages.
    labels: All available categories.
    """
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * FROM categorised_messages', engine)
    x = df['message'].values
    labels = df.drop(columns=['message', 'original', 'id', 'genre']).columns
    y = df.drop(columns=['message', 'original', 'id', 'genre']).values
    return x, y, labels


def tokenize(text):
    """
    Tokenization function (normalise, clean, lemmatize, tokenize).
    :param text: Message text.
    :return:
    tokens: Normalised, cleaned, lemmatized tokens.
    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # Normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word)
              for word in tokens if word not in stop_words]
    return tokens


def build_model():
    """
    Create a pipeline using the XGBoost classifier.
    :return: pipeline: Pipeline object.
    """
    clf = xgb.XGBClassifier(random_state=0)
    pipeline = Pipeline([
        ('tfidfvect', TfidfVectorizer(tokenizer=tokenize)),
        ('multioutclf', MultiOutputClassifier(clf))
    ])
    return pipeline


def train(x, y, model):
    """
    Create a GridSearchCV model using the input pipeline.
    Fit the GridSearchCV model to find optimum parameters.
    :param x: Message data.
    :param y: Category values for each message.
    :param model: The pipeline to use in GridSearchCV.
    :return: cv: Model fitted using GridSearchCV.
    """
    parameters = {
        'tfidfvect__max_df': [0.5, 0.75],
        'multioutclf__estimator__n_estimators': [100, 200],
        'multioutclf__estimator__learning_rate': [0.25, 0.5, 0.75]
    }
    cv = GridSearchCV(model, param_grid=parameters)
    cv.fit(x, y)
    cv.best_params = cv.best_params_
    return cv


def evaluate_model(model, x_test, y_test, category_names):
    """
    Use a fitted model to make predictions on test data.
    Compare to test labels to evaluate performance of the model.
    Save the evaluation metrics.
    :param model: Fitted model.
    :param x_test: Test message data.
    :param y_test: Test category values for each message.
    :param category_names: All available category labels.
    :return: None
    """
    y_pred = model.predict(x_test)
    preds_df = pd.DataFrame(y_pred, columns=category_names)
    true_df = pd.DataFrame(y_test, columns=category_names)
    print("Optimum parameters for model: {}".format(model.best_params_))
    clf_rep = {}
    for col in preds_df.columns:
        clf_rep[col] = classification_report(true_df[col].values,
                                             preds_df[col].values,
                                             output_dict=True)
        print("Category: {}".format(col))
        print(classification_report(true_df[col].values,
                                    preds_df[col].values))
    # Save classification reports for display in the app.
    model.clf_rep = clf_rep


def save_model(model, model_filepath):
    """
    Export model as a pickle file.
    :param model: Fitted model.
    :param model_filepath: Relative filepath to save file.
    :return: None.
    """
    pklfile = open(model_filepath, 'wb')
    pkl.dump(model, pklfile)
    pklfile.close()


def main():
    """
    Main, calling functions to build, train, evaluate and save model.
    :return:
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        x, y, category_names = load_data(database_filepath)
        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=0)

        print('Building model...')
        model = build_model()

        print('Training model...')
        trained_model = train(x_train, y_train, model)

        print('Evaluating model...')
        evaluate_model(trained_model, x_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(trained_model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
