# Required imports
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import sys
import pandas as pd
import numpy as np
import re
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    """  (1) Load the sqlite database DisasterResponse.db
         (2) Read UniqueMessages table into a pandas dataframe df
         (3) Create the Feature variable X, multiple targets variable Y, list of category names in Y
         (4) Return X, Y and category_names

    Parameters
    ----------
    database_filepath : sqlite database file path

    Returns
    -------
    X : Vector of Feature variable (column 1) e.g.'message'
    Y : Matrix of multiple outputs/target columns (columns 4-39) e.g. 'related', 'request', 'offer'...
    category_names : Names of Y columns e.g. 'related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products'...

   """
    # Create a sqlite engine with the given database_filepath
    engine = create_engine('sqlite:///' + database_filepath)

    # Read the UniqueMessages table into the dataframe df
    df = pd.read_sql_table('UniqueMessages', engine)

    # Create feature vector messages in X
    X = df.iloc[:, 1]

    # Create multiple output target matrix of response categories (including all columns from 4 through the end of the dataframe)
    Y = df.iloc[:, 4:]

    # List of column names in Y
    levels = list(Y.columns)

    # return X, Y and label values
    return X, Y, levels


def tokenize(text):
    """ (1) Convert a text message into set of tokens
        (2) Normalize the text (make all lower case)
        (3) Lemmatize the text (i.e. convert words like raining, rained, rains as "rain")
        (4) Remove stop words such as “the”, “a”, “an”, “in” from the text

    Parameters
    ----------
    text : The text to tokenize

    Returns
    -------
    tokens : Tokens or Symbols

    """
    # Get all english language stopwords
    stop_words = stopwords.words("english")

    # instantiate the WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    # Use regular expression text matching to retain only alphanumeric characters and also normalize the text by converting to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize the text
    tokens = word_tokenize(text)

    # Lemmatize the text (i.e. convert words like raining, rained, rains as "rain")
    # Remove stop words such as “the”, “a”, “an”, “in”
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Return the tokens
    return tokens


def build_model():
    """ (1) Build model by creating a pipeline of Transformers and Estimators
        (2) Add additional experimented model parameters
        (3) Use GridSearchCSV pipeline to run the model over the provided parameter space to obtain optimal the model performance
        (4) Return the built model

        Parameters
        ----------
        None

        Returns
        -------
        model : Return the model built using the pipeline and specified parameters

        """
    # RandomForest multi output classifier pipeline with a CountVectorizer and TfidfTransformer
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Tune the RandomForest classifier's hyperparameters
    parameters = {
        # Flag to use or not use the tfidf bag of words transformation
        'tfidf__use_idf': (True, False),
        # The minimum number of samples required to split an internal node trying out default 2 and another value 4
        'clf__estimator__min_samples_split': [2, 4]
    }

    # Build out the Grid search cross validation model using the provided pipeline and hyperparamters using all available processors
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

    # Return the built model for usage
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """  (1) Evaluate the given model using the test vector
         (2) Compare predicitons with the given true values of Y
         (3) Generate model evaulation summary report with F1 score (accuracy, precision and recall)

    Parameters
    ----------
    model : The model to evaluate
    X_test : The test vector of features
    Y_test : The true values of Y variable(s) to compare with scored predictions
    category_names : Names of Y variable columns (categories)

    Returns
    -------
    None

    """

    # Score the model on test cases
    Y_predicted = model.predict(X_test)

    # Generate the classification report
    for colId, colName in enumerate(Y_test):
        print(colName)
        print(classification_report(Y_test[colName], Y_predicted[:, colId]))


def save_model(model, model_filepath):
    """ Save the model as python pickle file to the provided filepath

    Parameters
    ----------
    model : The model to save
    model_filepath : The file path to save the picke file

    Returns
    -------
    None

    """
    # Save the model to Python picle file (a binary model object)
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ The main execution function that takes in the user provided terminal arguments and runs the built functions to create, evaluate and save the model:
    (1) load_data(...) - Load the datatable into a dataframe
    (2) train_test_split(...) - Create the train-test split (80-20)
    (3) build_model() - Build the Random Forest Classifier
    (4) model.Fit(...) - Fit the model
    (5) evaluate_model(...) - Evaluate the model
    (6) save_model(...) - Save the model into a Python picke file
    (7) Provides messages when each steps is executed
    (8) Handle command line errors

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
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

# Invoke the main function
if __name__ == '__main__':
    main()
