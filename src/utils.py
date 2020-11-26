import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def calc_delta_day(df, col_name, drop_col=False):
    '''
    Input:
    df - dataframe with features
    col_name - columns name with the date(string) to count days up to
    dataset data ref
    
    Output:
    data - dataframe with col_name replaced by delta calculated
    '''
    data = df.copy()
    date_aux = pd.to_datetime(data[col_name])
    date_ref = pd.to_datetime('2020-10-25')
    calc_delta = date_ref - date_aux
    calc_delta = calc_delta.apply(lambda x: x.days)
    data['delta_' + col_name] = calc_delta
    
    if drop_col:
        data.drop(col_name, axis=1, inplace=True)
    
    return data


def cast_pct_col(df, col_name):
    '''
    Input:
    df - dataframe with features
    col_name - columns name with the percentage(string) to cast to
    float
    
    Output:
    data - dataframe with col_name casted to float 
    '''
    data = df.copy()
    aux = data[col_name].str.replace('%', "")
    aux = aux.str.strip()
    aux = aux.astype('float')
    data[col_name] = aux
    
    return data

def cast_currency_col(df, col_name):
    '''
    Input:
    df - dataframe with features
    col_name - columns name with the currency(string) to cast to
    float
    
    Output:
    data - dataframe with col_name casted to float 
    '''
    data = df.copy()
 

    aux = data[col_name].str.replace('$', '')
    
    aux = aux.str.replace(',', '')
    aux = aux.str.strip()
    aux = aux.astype('float')
    data[col_name] = aux

    return data

def train_lmodel(X, y, test_size=0.3, random_state=42):
    """
    INPUT:
    X - Dataframe with features
    y - Series with target columns
    test_size - percentage of test size
    random_state - random seed
    
    OUTPUT:
    mae_train - Mean Absolute Error of train set
    mae_test - Mean Absolute Error of test set
    lmodel - Fitted model
    """
    
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
    
    print("MAE train: {}".format(round(mae_train, 2)))
    print("MAE test: {}".format(round(mae_test, 2)))
    
    return (mae_train, mae_test, lmodel)

def processes_mlb(df, col_name, drop_column=False):
    '''
    INPUT:
    df - Dataframe with features
    col_name - column to be transformed in one-hot encoding
    
    OUTPUT:
    data - Dataframe with col_name transformed in one-hot encoding
    mlb - MultiLabelBinarizer fitted
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



def cast_bool(df, col_name):
    '''
    Input:
    df - dataframe with features
    col_name - columns name with values 'f' or 't'(string) to cast to
    a flag 0 or 1 respectively
    
    Output:
    data - dataframe with col_name tranformed in a flag 
    '''
    
    data = df.copy()
    
    convert_bool = {'t': 1, 'f': 0}
    data[col_name] = data[col_name].map(convert_bool)
    
    return data

def flag_room_type(df, col_name, drop_col=False):
    '''
    Input:
    df - dataframe with features
    col_name - columns name with room type
    
    Output:
    data - dataframe with col_name tranformed in a flag 
    '''
    data = df.copy()
    data.loc[data[col_name] == 'Entire home/apt', 'flag_entire_home'] = 1
    data.flag_entire_home.fillna(0, inplace=True)
    
    if drop_col:
        data.drop(col_name, axis=1, inplace=True)
    
    return data

def create_dummies(df, col_name):
    '''
    INPUT:
    df - Dataframe with features
    col_name - column to be transformed in one-hot encoding
    
    OUTPUT:
    data - Dataframe with col_name transformed in one-hot encoding
    '''
    data = df.copy()
    data = pd.get_dummies(data, prefix=col_name, columns=[col_name])
    
    return data


def select_features(X, y, cuts):
    '''
    INPUT:
    X - Dataframe with features
    y - Series with target
    cuts - list with minimun percentage of class 1 in features
    
    OUTPUT:
    mae_trains - List of Mininum absolute error in train set for each cut configuration
    mae_tests - List of Mininum absolute error in test set for each cut configuration
    num_feats - Number of features for each cut configuration
    best_model - Best model fitted optimazing MAE test
    reduce_X - Data set with features used to fit best model
    
    '''
    X_ = X.copy()
    y_ = y.copy()
    
    features_pct = X_.mean().reset_index()
    features_pct.columns = ['feature', 'pct']
    
    mae_trains, mae_tests, num_feats = [], dict(), []
    
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
    print(mae_test)
        
    return (mae_trains, mae_tests, num_feats, best_model, reduce_X)


def create_month(df, col_date):
    '''
    INPUT:
    df - DataFrame with all vars
    col_date - String with column name with date 
    formatted [yyyy-mm-dd] and dtype String
    
    OUTPUT:
    data - DataFrame with an adicional colummn of month
    '''
    data = df.copy()
    
    data['month'] = data[col_date].apply(lambda x: x.split('-')[1])
    
    return data

def count_reviews(df, col_name, year):
    '''
    INPUT: 
    df - Dataframe with data
    col_name - column name (String) to aggregate and count lines
    year - INT year data ref
    OUTPUT:
    data - Dataframe with value counts of columns name
    
    '''
    data = df.copy()
    data = data[col_name].value_counts().reset_index()
    data.columns = [col_name, 'counts']
    data.sort_values(by=col_name, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data['year'] = year
    
    return data


def compare_tri(df_prev, df_ref, months):
    '''
    INPUT:
    df_prev - DataFrame with counts per month of previus year
    df_ref - DataFrame with counts per month of reference year 
    months - List with number of months included in analysis, 
    example ['01', '02', 03] - include Jan, Feb and Mar.
    OUTPUT:
    pct - Percentage reduced based in previews year.
    '''
    
    data_prev = df_prev.copy()
    data_ref = df_ref.copy()
    result_prev = data_prev[data_prev.month.isin(months)].counts.sum()
    result_ref = data_ref[data_ref.month.isin(months)].counts.sum()
    pct = round((1 - result_ref/result_prev) * 100, 2)
    
    return pct