import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads message and category data, then merges them into a dataframe.
    :param messages_filepath: relative path to messages data file (CSV).
    :param categories_filepath: relative path to categories data file (CSV).
    :return: data frame of merged messages and categories files.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='outer', on='id')
    return df


def clean_data(df):
    """
    Cleans the data frame as follows:
    - Create columns corresponding to message categories
    - Convert category values to 0 (unrelated) or 1 (related)
    - Drop duplicate message rows
    - Drop rows where 'related' category is 2
    :param df: The dataframe to clean
    :return: df: The cleaned dataframe
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Replace categories column in df with new category column
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    # Drop duplicates
    df.drop_duplicates(subset='message', inplace=True)
    # Drop rows where the related column is 2
    df.drop(index=df[df['related'] == 2].index, inplace=True)
    return df


def save_data(df, database_filename):
    """
    Save the dataframe as an SQL database.
    :param df: Cleaned data frame.
    :param database_filename: Relative path to SQL database to create.
    :return: None.
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('categorised_messages',
              engine,
              if_exists='replace',
              index=False)


def main():
    """
    Main function. Calls functions to load, clean and save data.
    :return: None.
    """
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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
