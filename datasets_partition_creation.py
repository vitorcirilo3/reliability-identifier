# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from scipy import stats


# create folders

os.mkdir("dataset_partitions")

# path of CSVs to read
inputPath = "datasets"

# output folder for the partitions of test, validation and training created
output_folder = "dataset_partitions/"


from sklearn.model_selection import train_test_split

# function to split df in partitions
def get_dataset_partitions_pd(df, train_split=0.8, val_split=0.1, test_split=0.1, target_variable=None):
    
    assert (train_split + test_split + val_split) == 1
    
    # Only allows for equal validation and test splits
    assert val_split == test_split 

    ## result of similarity between training, test, and deploy
    result = []

    n_tries = 20

    X = df.copy()
    y = X['y']
    X = X.drop("y", axis=1)
    n_features = X.shape[1]

    # Specify seed to always have the same split distribution between runs
    # If target variable is provided, generate stratified sets
    if target_variable is not None:

        for random_state in range(1,n_tries):

            X_train, X_test_deploy, y_train, y_test_deploy = train_test_split(X, y, test_size=0.20, random_state=random_state, stratify=y)

            X_test, X_deploy, y_test, y_deploy = train_test_split(X_test_deploy, y_test_deploy, test_size=0.50, random_state=random_state, stratify=y_test_deploy)

            distances = list(map(lambda i : stats.ks_2samp(X_train.iloc[:,i],X_test.iloc[:,i]).statistic,range(n_features)))

            distances_2 = list(map(lambda i : stats.ks_2samp(X_test.iloc[:,i],X_deploy.iloc[:,i]).statistic,range(n_features)))

            distances += distances_2

            result.append((random_state,max(distances)))

    else:
        print("ERROR in target_variable is None")
    #   indices_or_sections = [int(train_split * len(df)), int((1 - val_split) * len(df))]
    #   train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)



    result = pd.DataFrame(result, columns =['seed', 'distance'])
    result = result.sort_values(by=['distance'], ascending=True)

    random_state = result.iloc[0,0]

    X_train, X_test_deploy, y_train, y_test_deploy = train_test_split(X, y, test_size=0.20, random_state=random_state, stratify=y)

    X_test, X_deploy, y_test, y_deploy = train_test_split(X_test_deploy, y_test_deploy, test_size=0.50, random_state=random_state, stratify=y_test_deploy)



    train_ds = X_train
    train_ds['y'] = y_train

    deploy_ds = X_deploy
    deploy_ds['y'] = y_deploy

    test_ds = X_test
    test_ds['y'] = y_test
    
    return train_ds, deploy_ds, test_ds

# function to return a list of files names of folder
def getInputFiles(input_path:str) -> list:
    path = input_path

    files = os.listdir(path)

    return files


# function to convert a list to lowerCase
def convert_to_lowerCase(list_of_strings:list) -> list:
    return [each_string.lower() for each_string in list(list_of_strings)]


# Function to verify if exist label name as "class"
def verify_labelIs_class(df:pd.DataFrame) -> bool:
    label_check = "class" in df.columns
    
    return label_check

# Function to verify if exist label name as "y"
def verify_labelIs_y(df:pd.DataFrame) -> bool:
    label_check = "y" in df.columns
    
    return label_check

# Function to verify if exist label name as "y"
def verify_labelIs_target(df:pd.DataFrame) -> bool:
    label_check = "target" in df.columns
    
    return label_check

def convertBinaryCategoric_toBinaryNumeric(list_values:list, target_names: list) -> list:
    y = list_values
    y = y.replace(target_names,[0,1])
    list_values = y

    return list_values

## convert the label to y, if possible. And verify if everything is ok about the label
def convertLabel_andVerifyLabelStatus(df:pd.DataFrame) -> (pd.DataFrame and bool):
    if verify_labelIs_target(df):
        df.rename(columns={'target':'y'}, inplace=True)
        y_check = True
    if verify_labelIs_class(df):
        df.rename(columns={'class':'y'}, inplace=True)
        y_check = True
    else:
        if not verify_labelIs_y(df):
            y_check = False
        else:
            y_check = True


    target_names = list(df.loc[:,'y'].unique())

    # print(list(df.loc[:,'y'].unique()))
    if len(target_names) == 2:
            binaryClass_check = True
    else:
        binaryClass_check = True

    label_check = y_check & binaryClass_check

    if label_check:
        # conver class to binary
        df["y"] = convertBinaryCategoric_toBinaryNumeric(df["y"], target_names)

    return df, label_check
    

def get_FeatureMostCorrelatedWithTarget(df:pd.DataFrame) -> pd.DataFrame:
    df_corr = df.corr()

    # get the absolute value. It doesn`t matter if the correlation is positive or not
    # sort the values from the minor to max
    y_correlation = abs(df_corr[df_corr['y'].index != 'y']['y']).sort_values()

    y_correlation = pd.DataFrame(y_correlation)
    y_correlation['features'] = y_correlation.index

    return y_correlation['features'][-1]

def identify_andCreate_hotEncode(df:pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    num_cols = df._get_numeric_data().columns
    # print(len(num_cols))

    categorical_colunns = list(set(cols) - set(num_cols))

    for i in categorical_colunns:
        df = pd.get_dummies(df, columns = [i])

    return df
    


def get_FeatureLessCorrelatedWithTarget(df:pd.DataFrame) -> pd.DataFrame:
    df_corr = df.corr()

    # get the absolute value. It doesn`t matter if the correlation is positive or not
    # sort the values from the minor to max
    y_correlation = abs(df_corr[df_corr['y'].index != 'y']['y']).sort_values()

    y_correlation = pd.DataFrame(y_correlation)
    y_correlation['features'] = y_correlation.index

    return y_correlation['features'][0]


# get name of files
inputFiles = getInputFiles(inputPath)


# MAIN 
for fileName in inputFiles:
    print(fileName)

    read_data = pd.read_csv(inputPath + "/" + fileName)

    # replace question mark for NA
    read_data.replace("?", np.nan, inplace = True)

    # drop rows with NA
    read_data.dropna(inplace=True)

    # ignore columns that are not be able to convert to numeric
    read_data = read_data.apply(pd.to_numeric, errors='ignore')

    print(read_data.shape)
    # print(read_data.groupby('Class').count().iloc[0].iloc[0]/read_data.shape[0])

    fileName_withFormatFile = fileName
    fileName = fileName_withFormatFile.replace(".csv", "")

    # convert all coluns to lower case
    read_data.columns = convert_to_lowerCase(read_data.columns) 
    # verify if df has a label called y, if not, try rename. Also verify if is binary and numerical 
    read_data, label_check = convertLabel_andVerifyLabelStatus(read_data)

    preprocess_done = label_check

    read_data = identify_andCreate_hotEncode(read_data)

    read_data = read_data.reindex(sorted(read_data.columns), axis=1)

    read_data = read_data.sort_values(['y'], ascending=True)

    if preprocess_done:
        # get the most correlated feature
        FeatureMostCorrelatedWithTarget = get_FeatureMostCorrelatedWithTarget(read_data)

        train_ds, val_ds, test_ds = get_dataset_partitions_pd(df = read_data, target_variable = "y")

        test_ds = test_ds.reindex(sorted(test_ds.columns), axis=1)
        val_ds = val_ds.reindex(sorted(val_ds.columns), axis=1)
        train_ds = train_ds.reindex(sorted(train_ds.columns), axis=1)

        test_ds.to_csv(output_folder+"default_"+fileName+"_deploy.csv")
        val_ds.to_csv(output_folder+"default_"+fileName+"_test.csv")
        train_ds.to_csv(output_folder+"default_"+fileName+"_train.csv")

        # it tries to create a test and validation similar to each other
        limit_training_plus_deploy= train_ds.shape[0] + test_ds.shape[0]
        size_test = test_ds.shape[0]

        FeatureLessCorrelatedWithTarget = get_FeatureLessCorrelatedWithTarget(read_data)

        if read_data.groupby('y').count().iloc[0,0] > read_data.groupby('y').count().iloc[1,0]:
            strat_Under_test_data = read_data.sort_values(['y'], ascending=True).iloc[limit_training_plus_deploy:,]
            strat_Under_train_deploy_data = read_data.sort_values(['y'], ascending=True).iloc[:limit_training_plus_deploy,]


            strat_Under_deploy_data = strat_Under_train_deploy_data.sort_values(['y'], ascending=True).iloc[train_ds.shape[0]:,]
            strat_Under_train_data = strat_Under_train_deploy_data.sort_values(['y'], ascending=True).iloc[:train_ds.shape[0],]
        else:
            strat_Under_test_data = read_data.sort_values(['y'], ascending=False).iloc[limit_training_plus_deploy:,]
            strat_Under_train_deploy_data = read_data.sort_values(['y'], ascending=False).iloc[:limit_training_plus_deploy,]


            strat_Under_deploy_data = strat_Under_train_deploy_data.sort_values(['y'], ascending=False).iloc[train_ds.shape[0]:,]
            strat_Under_train_data = strat_Under_train_deploy_data.sort_values(['y'], ascending=False).iloc[:train_ds.shape[0],]
        

        strat_Under_train_data = strat_Under_train_data.reset_index()
        del strat_Under_train_data['index']
        strat_Under_test_data = strat_Under_test_data.reset_index()
        del strat_Under_test_data['index']
        strat_Under_deploy_data = strat_Under_deploy_data.reset_index()
        del strat_Under_deploy_data['index']

        size_test_menos_dois = strat_Under_test_data.shape[0] - 2
        aux_from_train = strat_Under_train_data.iloc[size_test_menos_dois:strat_Under_test_data.shape[0],]
        aux_from_test = strat_Under_test_data.iloc[size_test_menos_dois:,]
        strat_Under_train_data.update(aux_from_test)
        strat_Under_test_data.update(aux_from_train)

        df_group = strat_Under_train_data.groupby("y").count()
        print("training unreliable: ", df_group.loc[:,df_group.columns[0]])

        strat_Under_test_data = strat_Under_test_data.reindex(sorted(strat_Under_test_data.columns), axis=1)
        strat_Under_deploy_data = strat_Under_deploy_data.reindex(sorted(strat_Under_deploy_data.columns), axis=1)
        strat_Under_train_data = strat_Under_train_data.reindex(sorted(strat_Under_train_data.columns), axis=1)


        strat_Under_train_data.to_csv(output_folder+"under_"+fileName+"_train.csv")
        strat_Under_test_data.to_csv(output_folder+"under_"+fileName+"_test.csv")
        strat_Under_deploy_data.to_csv(output_folder+"under_"+fileName+"_deploy.csv")
    else:
        print("it was not possible to conclude the process: Error in the following database csv: "+fileName)

    print("##########################")

        
print("done! " + str(len(inputFiles)) + " files!")
