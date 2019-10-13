# Import required libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    (1) Load messages csv file into Pandas dataframe called messages
    (2) Load categories csv file into Pandas dataframe called categories

     Parameters
     ----------
     messages_filepath : disaster_messages.csv filepath
     categories_filepath : disaster_categories.csv filepath

     Returns
     -------
     messages : The messages dataframe that will be further transformed in the cleaning step
     categories : The categories dataframe that will be further transformed in the cleaning step

     """
    # Read the message csv dataset into pandas dataframe messages
    messages = pd.read_csv(messages_filepath)

    # Read the message csv dataset into pandas dataframe categories
    categories = pd.read_csv(categories_filepath)

    # Return the dataframes for further cleaning
    return messages, categories


def clean_data(messages, categories):
    """
     (1) Split the categories column into 36 sepearate disaster message numeric columns with 1's or 0's
     (2) Concatenate the original dataframe with the new `categories` dataframe with 36 columns

     Parameters
     ----------
     messages : The messages dataframe we want to transform and clean further to get into a shape we can analyze it.
     categories : The categories dataframe we want to transform and clean further to get into a shape we can analyze it.

     Returns
     -------
     df : The dataframe that is clean and ready to for modeling.

     """
    # Create a categories dataframe of the 36 individual category columns
    categories = categories["categories"].str.split(";", expand=True)

    # Select the first row of the categories dataframe
    firsRow = categories.iloc[0]

    # Use this row to extract a list of new column names for categories.
    # Apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = firsRow.apply(lambda x : x[:-2])

    # Rename the columns of `categories`
    categories.columns = category_colnames

    # Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). For example, related-0 becomes 0
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x[-1])
        # Convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([messages, categories], axis=1, sort=False)

    # Remove duplicates from the final dataframe
    df.drop_duplicates()

    # Return the dataframe for modeling and analysis
    return df


def save_data(df, database_filename):
    """
    (1) Create sqlite database engine with the specified database_filename
    (2) Save the dataframe to a Table ("UniqueMessages")

    Parameters
    ----------
    df : The datframe to save
    database_filename : The sqlite database filename

    Returns
    -------
    None

    """
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('UniqueMessages', engine, index=False)


def main():
    """
    The main execution function that takes in the user provided terminal arguments
    and runs the built functions to perform Extract, Tranform and Load (ETL) tasks:

    (1) load_data(...) - Load the csv files into their respective dataframes
    (2) clean_data(...) - Perform ETL steps and clean data
    (3) save_data(...) - Save the datframe into a sqlite database table
    (4) Provide messages when each steps is executed
    (5) Handle command line errors

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

# Invoke the main function
if __name__ == '__main__':
    main()
