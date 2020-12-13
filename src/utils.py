import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def describe_df(df):
    ''' Check dtypes and null in all columns of the dataframe.
    
    Parameters:
    df (dataframe) - Input dataframe 
    
    Returns:
    merge_df (dataframe): Dataframe with name, type, count nulls and ratio nulls
    for each column
    '''

    types = df.dtypes.reset_index()
    types.columns = ['col', 'type']

    nulls = df.isnull().sum().reset_index()
    nulls.columns = ['col', 'count_nulls']
    nulls['ratio_nulls'] = round(nulls['count_nulls']/df.shape[0], 2)

    merge_df = types.merge(nulls, on='col')
    
    return merge_df

def calc_delta_day(df, col_name, date_ref, drop_col=False):
    ''' Calculate the difference between a
    columns data in a dataframe and a date reference
    and return in a new column named [delta_ + col_name].

    Parameters:
    df (dataframe): input dataframe
    col_name (string): columns name with the dates to be calculated
    date_ref (string): date formatted as 'yyyy-mm-dd' to be the reference
    drop_col (bool): Flag to control if original date column will be dropped
    
    Returns:
    data (dataframe): Dataframe with new column with delta calculated values
    '''

    data = df.copy()
    date_aux = pd.to_datetime(data[col_name])
    date_ref = pd.to_datetime(date_ref)
    calc_delta = date_ref - date_aux
    calc_delta = calc_delta.apply(lambda x: x.days)
    data['delta_' + col_name] = calc_delta
    
    if drop_col:
        data.drop(col_name, axis=1, inplace=True)
    
    return data

def cast_to_float(df, col_name, remove_chars):
    ''' Cast a column in a dataframe from strings to floats,
    removing chars in remove_chars.

    Parameters:
    df (dataframe): input dataframe
    col_name (string): column name with numerical value in string
    remove_chars (list): list with chars to be removed, enter a extra '\' previews special chars.
        eg.: ['\$']
    
    Returns:
    data (dataframe): dataframe with col_name casted to float 
    '''
    data = df.copy()

    # remove chars from remove_char list
    patr = '|'.join(remove_chars)
    aux = data[col_name].apply(lambda string: re.sub(patr, '', string) if string is not np.nan else string)

    aux = aux.str.strip()
    aux = aux.astype('float')
    data[col_name] = aux
    
    return data    

def cast_bool(df, col_name):
    ''' Cast a column in a dataframe from strings
    't' or 'f' to int 1 or 0 respectively.

    Parameters:
    df (dataframe): input dataframe
    col_name (string): columns name with bool values as chars
    
    Returns:
    data (dataframe): dataframe with col_name casted to int 
    '''
    
    data = df.copy()
    convert_bool = {'t': 1, 'f': 0}
    data[col_name] = data[col_name].map(convert_bool)
    
    return data

def create_flag_entire_home(df, col_name, drop_col=False):
    ''' Create a flag in dataframe with room type is rented entirely.

    Parameters:
    df (dataframe): dataframe with features
    col_name (string): columns name with room type
    drop_col (bool): Flag to control if original date column will be dropped
    
    Returns:
    data (dataframe): Dataframe with new column with flag
    if the place is rented entirely.
    '''

    data = df.copy()
    data.loc[data[col_name] == 'Entire home/apt', 'flag_entire_home'] = 1
    data.flag_entire_home.fillna(0, inplace=True)
    
    if drop_col:
        data.drop(col_name, axis=1, inplace=True)
    
    return data

def train_lmodel(X, y, test_size=0.3, random_state=42):
    ''' Split a dataset, train a linear regression model 
    and calculate Mean Absolute error from train and test
    set.

    Parameters:
    X (dataframe): Dataframe with features
    y (series): target values
    test_size (float): percentage of test size
    random_state (int): random seed
    
    Returns:
    mae_train (float): Mean Absolute Error of train set
    mae_test (float): Mean Absolute Error of test set
    lmodel (sklearn.linear_model): Fitted model
    '''
    
    # Split train test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    
    # Fit regression linear model
    lmodel = LinearRegression(normalize=True)
    lmodel.fit(X_train, y_train)
    
    # Predict values for train and test set
    y_pred = lmodel.predict(X_test)
    y_train_pred = lmodel.predict(X_train)
    
    # Calculate MAE for train and test set
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_pred)

    return (mae_train, mae_test, lmodel)

def processes_mlb(df, col_name, drop_column=False):
    ''' Create columns in multiBinarizer encoding from a column in dataframe.

    Parameters:
    df (dataframe): input dataframe
    col_name (string): column name to be transformed in multiBinarizer encoding
    drop_column (bool): Flag to control if original date column will be dropped
    
    Returns:
    data (dataframe): Dataframe with col_name transformed in multiBinarizer encoding
    mlb (sklearn.preprocessing): MultiLabelBinarizer fitted
    '''

    data = df.reset_index(drop=True).copy()
    mlb = MultiLabelBinarizer()
    
    aux = data[col_name].apply(lambda x: eval(x))
    aux = mlb.fit_transform(aux)
    aux = pd.DataFrame(aux, columns=col_name + '_' + mlb.classes_)
    data = data.join(aux)
    
    if drop_column:
        data.drop(col_name, axis=1, inplace=True)
    
    return (data, mlb)

def create_dummies(df, col_name):
    ''' Transform a column in dataframe to other columns in one-hot encoding

    Parameters:
    df (dataframe): input dataframe
    col_name (string): column to be transformed in one-hot encoding
    
    Returns:
    data (dataframe): Dataframe with col_name transformed in one-hot encoding
    '''

    data = df.copy()
    data = pd.get_dummies(data, prefix=col_name, columns=[col_name])
    
    return data


def select_features(X, y, cuts):
    ''' Train linear regression models varying percentage minimun of
    each feature of 1 value. Return the best model fitted (lowest MAE test error),
    MAE train, MAE test, number of features of each configuration and dataframe with
    features used to train best model. 

    Parameters:
    X (dataframe): Dataframe with features
    y (series): Target values
    cuts (list): List varying minimun percentage of class 1 in features
    
    Returns:
    mae_trains (list): List of MAE in train set for each cut configuration
    mae_tests (list): List of MAE in test set for each cut configuration
    num_feats (list): Number of features for each cut configuration
    best_model (sklearn.linear_model): Best model fitted optimazing MAE test
    reduce_X (dataframe): Data set with features used to fit best model
    
    '''

    X_ = X.copy()
    y_ = y.copy()
    
    features_pct = X_.mean().reset_index()
    features_pct.columns = ['feature', 'pct']
    
    mae_trains, mae_tests, num_feats = [], dict(), []
    
    # Train different configurations of Linear regression models
    for cut in cuts:
        list_features = list(features_pct[features_pct.pct >= cut].feature)
        reduce_X = X_[list_features].copy()
    
        mae_train, mae_test, model = train_lmodel(reduce_X, y_)
        mae_trains.append(mae_train)
        num_feats.append(len(list_features))
        mae_tests[str(cut)] = mae_test
        
    # Best Model
    best_cutoff = min(mae_tests, key=mae_tests.get)
    list_features = list(features_pct[features_pct.pct >= float(best_cutoff)].feature)
    reduce_X = X_[list_features].copy()
    mae_train, mae_test, best_model = train_lmodel(reduce_X, y_)
        
    return (mae_trains, mae_tests, num_feats, best_model, reduce_X)


def create_month(df, col_date):
    ''' Extract month from a column in a dataframe with date to a new column.

    Parameters:
    df (dataframe): Input dataFrame
    col_date (string): String with column name with date formatted  as 'yyyy-mm-dd'
    
    Returns:
    data (dataframe): DataFrame with an adicional colummn of month
    '''

    data = df.copy()
    
    data['month'] = data[col_date].apply(lambda x: x.split('-')[1])
    
    return data

def count_column(df, col_name, label_name, label_value):
    ''' Agregate a dataframe by col_name and count their values.
    After it, create a column with label value.

    Parameters: 
    df (dataframe): Input dataframe
    col_name (string): Column name to aggregate and count lines
    label_name (string): Name of column that will be created with label_value 
    label_value (int): Label value

    Returns:
    data (dataframe): Dataframe with value counts of columns name
    '''

    data = df.copy()
    data = data[col_name].value_counts().reset_index()
    data.columns = [col_name, 'counts']
    data.sort_values(by=col_name, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data[label_name] = label_value
    
    return data

def compare_months(df_prev, df_ref, months):
    ''' Calculate the ratio between summed values in a year and
    summed values in a previews year in same period. The return will be
    a percentage.

    Parameters:
    df_prev (dataframe): DataFrame with counts per month of previus year
    df_ref (dataframe): DataFrame with counts per month of reference year 
    months (list): List with number of months included in analysis, 
    example ['01', '02', 03] - include Jan, Feb and Mar.

    Returns:
    pct (float): Percentage reduced based in previews year.
    '''
    
    data_prev = df_prev.copy()
    data_ref = df_ref.copy()
    result_prev = data_prev[data_prev.month.isin(months)].counts.sum()
    result_ref = data_ref[data_ref.month.isin(months)].counts.sum()
    pct = round((1 - result_ref/result_prev) * 100, 2)
    
    return pct