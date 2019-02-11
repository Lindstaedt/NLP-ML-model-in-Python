import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_file_path, categories_file_path):
    """
    This function loads data from two files and merges them via left join based on ID column
    Input:
        message_file_path: full file path to CSV file containing messages
        categories_file_pat: full file path to CSV file containing categories
    Output:
        df: data frame containing joined content from the two inputs
    """

    messages = pd.read_csv(messages_file_path, index_col='id')
    categories = pd.read_csv(categories_file_path, index_col='id')
    df = pd.DataFrame(messages).join(pd.DataFrame(categories), how='left')
    return df


def clean_data(df):
    """
    This function cleans the data frame by splitting columns, cleaning text, and removing duplicates
    Input:
        df: data frame to be cleaned
    Output:
        df: cleaned data frame
    """

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.extract('(\d)').astype(int)
        categories[column] = categories[column].replace(2, 1)
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """"
    Save a data frame into an SQLite database
    Input:
        df: data to be saved
        database_filename: full path to database
    Output:
        none
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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


if __name__ == '__main__':
    main()
