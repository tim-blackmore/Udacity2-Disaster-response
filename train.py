# Created by timot at 02/03/2021
import nltk
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import logging

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

logging.basicConfig(level=logging.DEBUG)


def load_data():
    """
    Load in data from database and split into X and Y vars
    :return: X and Y variables
    """
    engine = create_engine('sqlite:///disaster_db.db')
    df = pd.read_sql_table('disaster_db', engine)
    X_data = df['message']
    Y_data = df.iloc[:, 4:].values

    logging.debug('function:load_data:data loaded and separated into X and Y variables')

    return X_data, Y_data


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


if __name__ == "__main__":
    X, Y = load_data()

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, Y,  test_size=0.33, random_state=42)

    # train classifier
    pipeline.fit(X_train, y_train)

    # predict on test data
    y_pred = pipeline.predict(X_test)

    # display results
    accuracy = (y_pred == y_test).mean()

    print("Accuracy:", accuracy)

    # print(classification_report(y_test, y_pred))

    pipeline.get_params()

    parameters = {'tfidf__norm': ['l1', 'l2'],
                  'clf__estimator__criterion': ["gini", "entropy"]

                  }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    cv.fit(X_train, y_train)

    y_pred = cv.predict(X_test)

    # print(classification_report(y_test, y_pred))

    accuracy = (y_pred == y_test).mean()

    print("Accuracy:", accuracy)
