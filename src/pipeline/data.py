"""
This script is responsible for the function to fetch the data from local directory

Author: Felipe Lana Machado
Date: 01/08/2022
"""
import os
import pandas as pd
import config


def get_clean_data(path, save: bool = True):
    """
    Loads and cleans the data from a given path
    Args:
        path (str): The path to the csv file

    Returns:
        X_df (pandas dataframe): Dataframe of the features from the loaded csv file
        y_df (pandas dataframe): Dataframe of the labels from the loaded csv file
    """
    data_df = pd.read_csv(path)

    # chaning column names to use _ instead of -
    columns = data_df.columns
    columns = [col.replace('-', '_') for col in columns]
    data_df.columns = columns
    data_df.columns = data_df.columns.str.strip()

    # remove duplicates
    data_df = data_df[~data_df.duplicated()]

    # make all characters to be lowercase in string columns
    data_df = data_df.applymap(
        lambda s: s.lower().strip() if isinstance(s, str) else s)

    # map label salary to numbers
    data_df['salary'] = data_df['salary'].map({'>50k': 1, '<=50k': 0})

    if save:
        data_df.to_csv(os.path.join(config.__MAIN_DIR, 'data', 'processed', 'census_cleaned.csv'), index=False)

    y_df = data_df.pop('salary')
    x_df = data_df

    return x_df, y_df
