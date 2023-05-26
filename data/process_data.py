import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.

    Args:
        messages_filepath (str): Filepath of the messages dataset.
        categories_filepath (str): Filepath of the categories dataset.

    Returns:
        pandas.DataFrame: Merged dataframe containing messages and categories.

    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, how='left', on=['id'])

    return df


def clean_data(df):
    """
    Clean and preprocess the merged dataframe.

    Args:
        df (pandas.DataFrame): Merged dataframe containing messages and categories.

    Returns:
        pandas.DataFrame: Cleaned and preprocessed dataframe.

    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2]).tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the category 'child alone' all data = 0
    categories.drop('child_alone', axis=1, inplace=True)

    # replace value 2 with 1 as we have only two classes 1 and 0
    categories.replace(2, 1, inplace=True)

    # replace `categories` column in `df` with new category columns.
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save the dataframe into an SQLite database.

    Args:
        df (pandas.DataFrame): Dataframe to be saved.
        database_filename (str): Filename of the SQLite database.

    Returns:
        None

    """

    # Save the df dataset into an sqlite database.
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('InsertTableName', engine, index=False, if_exists='replace')



def main():
    """
    Main function to process and save data.

    Args:
        None

    Returns:
        None
    """

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('I am starting!!!')
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
