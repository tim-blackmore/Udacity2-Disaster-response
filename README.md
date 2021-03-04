# Udacity2-Disaster_response_pipeline
A machine learning pipeline for classifying messages from natural disasters and allocation them to the correct services.

## Quickstart

Navigate to www.website.com to see the project in action.
 
## Installation

The libraries used in this project are outlined in requirements.txt. They can be installed using the following command.

1. Install pipenv.
``` pip3 install pipenv```

2. Install packages in virtual environment
```pipenv install -r requirements.txt```

## How to run?
1. Open a command line in the project folder.
2. Run the following commands (the second takes a long time to run).
This command will output a database that the next command will use.
``` pipenv run process_data.py -m messages.csv -c categories.csv -d disaster_db```<br/>
This will train the classifier.
``` pipenv run train_classifier```

## Motivation

This project fulfils one of the requirements for the completion of the Data Science nanodegree with Udacity. 

## Project files

- requirements.txt - Contains the required packages to set up the development envrironment.
- process_data.py - A python script which imports, merges and clean the dataset and stores it in a SQlite database.
- train_classifir.py - A python script which creates a machine learning pipeline and trains a random forest classifier.
- custom_transformer.py - A python cript which contains transform classes for text data.
- models/model.joblib - The trained model.

### Data files
- categories.csv - the classes for each of the categories
- messages.csv - the mesages from which the classes relate too
- disaster.db - the database which is an output from prcoess data.py.



