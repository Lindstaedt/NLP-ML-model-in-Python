import sys
import pandas as pd
import numpy as np
import nltk
import re
import pickle

from sqlalchemy import create_engine

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    """
    Loads data from a SQLite database file
    Data is read from table "Messages" and split into two data frames based on column names
    :param database_filepath: full path to database file
    :return:
        X: "Message" column from database table
        Y: all other columns from database table except for 2 other named columns
        category_names: column labels for the Y dataframe
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df.message
    Y = df.drop(['message', 'original', 'genre'], axis=1)
    category_names = list(Y.columns)
    return X, Y, category_names


def tokenize(text):
    """
    Text processing pipeline:
        remove punctuation and cast to lower case
        tokenize text
        remove stopwords based on several common languages
        lemmatize
    :param text: a piece of text (multiple words)
    :return: tokens: tokens generated from the text
    """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # build multi-language list of stop words
    stop_words = [*stopwords.words("english"),
                  *stopwords.words("french"),
                  *stopwords.words("spanish")]

    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    Build a model object that consists of:
        sklearn Pipeline object with vectorizer, TFIDF and a MultiOutputClassifier
        GridSearch object with several parameter for the pipeline
    :return: model: the GridSearchCV object of the pipeline
    """
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('multiOutClf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC()))),
    ])

    parameters = {
        'tfidf__use_idf': (True, False),
        #'multiOutClf__estimator__estimator__class_weight': ['balanced', None],
        #'multiOutClf__estimator__estimator__C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        #'multiOutClf__estimator__estimator__fit_intercept': [True, False],
        #'multiOutClf__estimator__estimator__max_iter': [10, 100, 500],
    }

    model = GridSearchCV(pipeline, param_grid=parameters, cv=3, scoring='accuracy', n_jobs=1, verbose=6)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Predict Y values for X_test with the model and then calculate and print the classification report that compares
    the model results to the Y_test values
    :param model:
    :param X_test:
    :param Y_test:
    :param category_names:
    :return:
    """
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()