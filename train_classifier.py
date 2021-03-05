# Created by timot at 02/03/2021
import nltk
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import logging
from joblib import dump
from custom_transformer import StartingVerbExtractor, TextLengthExtractor, SentimentExtractor, WordCountExtractor

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

logging.basicConfig(level=logging.DEBUG)


def load_data():
    """
    Load in data from database and split into X and Y vars
    :return: X and Y variables
    """
    engine = create_engine('sqlite:///data/disaster_db.db')
    df = pd.read_sql_table('disaster_db', engine)
    X_data = df['message']
    Y_data = df.iloc[:, 4:].values

    logging.debug('function:load_data: data loaded and separated into X and Y variables')

    return X_data, Y_data


def tokenize(text):
    """
    tokenize text data. Replace urls, make lower case, strip whitespace, and lemmatize
    :param text:
    :return: the cleaned lemmatized version of the text.
    """
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

    # logging.debug('function:tokenize: text has been cleaned and lemmatized')

    return clean_tokens


def model_pipeline():
    """
    Set up model pipeline. Include custom transformers and optimise parameters using gridsearchCV
    :return: an instance of the model
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor()),
            ('text_length', TextLengthExtractor()),
            ('word_count', WordCountExtractor()),
            ('sentiment', SentimentExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # parameters = {'features__text_pipeline__tfidf__norm': ['l1', 'l2'],
    #               'clf__estimator__criterion': ["gini", "entropy"],
    #               'clf__estimator__max_features': ['auto', 'sqrt', 'log2'],
    #               'clf__estimator__class_weight': ['balanced']}  # used to account for class imbalance

    # Best CV params
    parameters = {'features__text_pipeline__tfidf__norm': ['l2'],
                  'clf__estimator__criterion': ["gini"],
                  'clf__estimator__max_features': ['sqrt'],
                  'clf__estimator__class_weight': ['balanced']}  # used to account for class imbalance

    # Focus on the f1 score due to the unbalanced classes
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=-1)

    logging.debug('function:model_pipeline: model pipeline instantiated')

    return cv


def display_results(cv, y_test, y_pred):
    """
    display the model results after gridsearchCV has finshed.
    :param cv: an instance of the model
    :param y_test: the test data
    :param y_pred: the predicted data
    """
    f_score_list = []
    for i in range(0, 35):
        print(classification_report(y_test[:, i], y_pred[:, i]))
        score = f1_score(y_test[:, i], y_pred[:, i], average='macro')
        f_score_list.append(score)

    logging.debug('function:display_results: results have been calculated')

    print('Average f score:', (round(sum(f_score_list) / len(f_score_list), 2)))
    print('Best parameters:', cv.best_params_)


if __name__ == "__main__":
    X, Y = load_data()  # load data

    X_train, X_test, y_train, y_test = train_test_split(X, Y,  test_size=0.3, random_state=45)  # train test split
    logging.debug('general: train/test split occurred successfully')

    model = model_pipeline()  # instantiate pipeline with gridsearchCV included

    model.fit(X_train, y_train)  # train classifier
    logging.debug('general: classifier trained successfully')

    y_pred = model.predict(X_test)  # predict on test data
    logging.debug('general: predictions on test data are completed')

    display_results(model, y_test, y_pred)  # display results

    dump(model, 'models/model.joblib')  # export model


