import sys
import getopt
import logging
import pandas as pd
from sqlalchemy import create_engine

logging.basicConfig(level=logging.DEBUG)


def parse_command_line_args(argv):
    """
    Parses filenames from command line arguments
    :param argv: a list of command line options and arguments
    :return: 2 file names to be imported, and the name of the database.
    """
    try:
        opts, args = getopt.getopt(argv, "m:c:d:", ["messages=", "categories=", "database="])
    except getopt.GetoptError:
        print('test.py -m <messages.csv file name> -c <categories.csv file name> -d <database_name>')
        sys.exit(2)

    messages_file_name = opts[0][1]
    categories_file_name = opts[1][1]
    database_name = opts[2][1]

    logging.debug('function:parse_command_line_args: Filenames parsed: {}, {}, {}.'.format(messages_file_name,
                                                                                           categories_file_name,
                                                                                           database_name))
    return messages_file_name, categories_file_name, database_name


def import_messages_and_categories(messages_file_name, categories_file_name):
    """
    Import messages and categories CSV data
    :param messages_file_name:
    :param categories_file_name:
    :return: 2 csv files
    """
    messages = pd.read_csv('data/' + messages_file_name)
    categories = pd.read_csv('data/' + categories_file_name)
    logging.debug('function:import_messages_and_categories:messages and category data imported.')
    return messages, categories


def merge_and_clean(messages_df, categories_df):
    """
    merge the two dfs and clean
    :param messages_df: a dataframe containing the messages info
    :param categories_df: a dataframe containing the categories info
    :return: a clean df
    """
    # merge datasets
    df = messages_df.merge(categories_df, on='id')
    logging.debug('function:merge_and_clean:data merge successful.')

    # split up category columns and get new names
    categories = df.categories.str.split(';', expand=True)
    row = categories.iloc[:1]
    category_colnames = row.apply(lambda x: x.str[:-2]).values.tolist()[0]
    categories.columns = category_colnames
    logging.debug('function:merge_and_clean:category split and rename successfully.')

    # extract label data from column data
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    logging.debug('function:merge_and_clean:category labels extracted successful.')

    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    logging.debug('function:merge_and_clean:clean and merge completed.')

    return df


def create_database(df, database_name):
    """
    creates a sqlite database from the dataframe
    :param df: a clean dataframe
    :param database_name: the name of the database
    """
    engine = create_engine('sqlite:///' + database_name + '.db')
    df.to_sql(database_name, engine, index=False)
    logging.debug('function:create_database:the database named {} was created successfully.'.format(database_name))


if __name__ == "__main__":
    mesg_file_name, categ_file_name, db_name = parse_command_line_args(sys.argv[1:])
    mesg_df, categ_df = import_messages_and_categories(mesg_file_name, categ_file_name)
    clean_df = merge_and_clean(mesg_df, categ_df)
    create_database(clean_df, db_name)

