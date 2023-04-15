import pandas as pd
import re
import os


inputPath = "dataset_partitions"

# function to return a list of files names of folder
def getInputFiles(input_path:str) -> list:
    path = input_path

    files = os.listdir(path)

    return files


# get name of files
inputFiles = getInputFiles(inputPath)

# MAIN 
for fileName in inputFiles:

    path_and_filename_111 = inputPath + "/" + fileName
    path_and_filename_222 = path_and_filename_111.replace("dataset_partitions","dataset_partitions2")

    read_data1 = pd.read_csv(path_and_filename_111)
    read_data2 = pd.read_csv(path_and_filename_222)

    if (read_data1.equals(read_data2)) == False:
        print(fileName)
        print("______________")




##############################
df1 = pd.read_csv('dataset_partitions/default_cylinder-bands_deploy.csv')

df2 = pd.read_csv('dataset_partitions2/default_cylinder-bands_deploy.csv')

print(df1.equals(df2)) 


df1 = pd.read_csv('dataset_partitions/under_cylinder-bands_deploy.csv')

df2 = pd.read_csv('dataset_partitions2/under_cylinder-bands_deploy.csv')

print(df1.equals(df2)) 

###########################################################
df1 = pd.read_csv('output/default_banknote-authentication_train_default_banknote-authentication_deploy/default_banknote-authentication_train_default_banknote-authentication_deploy_irt.csv')

df2 = pd.read_csv('output2/default_banknote-authentication_train_default_banknote-authentication_deploy/default_banknote-authentication_train_default_banknote-authentication_deploy_irt.csv')


print(df1.equals(df2)) 


df1 = pd.read_csv('output/default_pc1_train_default_pc1_test/default_pc1_train_default_pc1_test_irt.csv')

df2 = pd.read_csv('output2/default_pc1_train_default_pc1_test/default_pc1_train_default_pc1_test_irt.csv')


print(df1.equals(df2)) 


df1 = pd.read_csv('output/default_pc1_train_default_pc1_test/irt_item_param.csv')

df2 = pd.read_csv('output2/default_pc1_train_default_pc1_test/irt_item_param.csv')


print(df1.equals(df2)) 


df1 = pd.read_csv('output/under_pc1_train_under_pc1_deploy/irt_item_param.csv')

df2 = pd.read_csv('output2/under_pc1_train_under_pc1_deploy/irt_item_param.csv')


print(df1.equals(df2)) 



df1 = pd.read_csv('output/default_banknote-authentication_train_default_banknote-authentication_deploy/default_banknote-authentication_train_default_banknote-authentication_deploy_final.csv')

df2 = pd.read_csv('output2/default_banknote-authentication_train_default_banknote-authentication_deploy/default_banknote-authentication_train_default_banknote-authentication_deploy_final.csv')


print(df1.equals(df2)) 

###########################################################


