# Natural Language Processing with Machine Learning and web app in Python

## Installation
This project uses the Anaconda Python 3.7 distribution and the following libraries: pandas, sqlalchemy, sklearn, flask, 
plotly, nltk, json, and pickle. 

## Project Motivation
This project is part of my completion of the Udacity Data Science Nanodegree.
The goal is to build a system consisting of two main components:

####1) ETL pipeline and ML model 
Building an ETL pipeline, storing data in a database, 
training an NLP ML model and then using a grid search to find the best model.

####2) Web frontend built that uses the trained ML model to categorize text
The front end is built with Flask. The web front end runs the ML model to 
categorize user input text. 

The tool uses a .

## File Descriptions
- data folder:
  - disaster_categories.csv and disaster_messages.csv: training data set provided by Figure Eight Inc as part
of the Udacity Data Science Nanodegree. These two files are the input for 
the ETL program
  - DisasterResponse.db: output file of the ETL program, contains cleaned up 
  data from the tron input files
- model folder:
  - best_model.pkl: saved off copy of the best model coming from the ML model building file
- process_data.py: ETL program that cleans up input data and produces SQLite database file
- train_classifier.py: ML program that performs grid search to identify optimized ML model
based on Natural Language Processing to categorize text
- app folder:
   - run.py: program to run web front end
   - templates subfolder: content for the web app

## How to use:

- To run ETL process: run process_data.py - doc string explains parameters
- To build NLP ML model: run train_classifier.py - doc string explains parameters.
Notice that the saved code has minimal parameters enabled in the pipeline to minimize initial runtime.
To find a much better model, expand the GridSearchCV parameters.
- To run the web app: execute run.py in the app folder, then open http://localhost:3001 in a browser on the same computer