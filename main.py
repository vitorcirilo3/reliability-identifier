# -*- coding: utf-8 -*-

###################################################################################################
############################################## IMPORTS
###################################################################################################

import pandas as pd
import os
from subprocess import check_output
import pickle
import dill
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier #For Classification
from sklearn.ensemble import GradientBoostingRegressor #For Regression
from sklearn.ensemble import AdaBoostClassifier #For Classification
from sklearn.ensemble import AdaBoostRegressor #For Regression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

# ###################################################################################################
# ############################################## FUNCTIONS
# ###################################################################################################
def DataframeFeatures_fromHiperparameters(hiperparameters, dataset_name, dataset_type, dataset_partition):

    d = {'dataset_name': dataset_name, 
    'dataset_type': dataset_type, 
    'dataset_partition': dataset_partition,
    'discriminacao_mean': hiperparameters['Discriminacao'].mean(), 
    'dificuldade_mean': hiperparameters['Dificuldade'].mean(), 
    'adivinhacao_mean': hiperparameters['Adivinhacao'].mean(), 
    'discriminacao_median': hiperparameters['Discriminacao'].median(), 
    'dificuldade_median': hiperparameters['Dificuldade'].median(), 
    'adivinhacao_median': hiperparameters['Adivinhacao'].median(),
    'discriminacao_max': hiperparameters['Discriminacao'].max(), 
    'dificuldade_max': hiperparameters['Dificuldade'].max(),
    'adivinhacao_max': hiperparameters['Adivinhacao'].max(), 
    'discriminacao_min': hiperparameters['Discriminacao'].min(), 
    'dificuldade_min': hiperparameters['Dificuldade'].min(),
    'adivinhacao_min': hiperparameters['Adivinhacao'].min()}

    df = pd.DataFrame(data=d,index=[0])

    return df


def training(train, test, fold_no, model):
    x_train = train.drop(['dataset_type'],axis=1)
    y_train = train.dataset_type
    x_test = test.drop(['dataset_type'],axis=1)
    y_test = test.dataset_type
    model.fit(x_train, y_train)
    score = model.score(x_test,y_test)
    f1_score_value = f1_score(y_test,model.predict(x_test))
    acc_vector.append(score)
    f1_vector.append(f1_score_value)



###################################################################################################
############################################## EXECUTION OF METHODOLOGY
###################################################################################################

# csv with name of all names of datasets
input_filesName = pd.read_csv("file_inputs.csv")

path = "dataset_partitions/"

columns_name = ['data_train_name', 'data_test_or_deploy_name']

input_filesName['id'] = input_filesName['data_train_name'].astype(str) + "_" + input_filesName['data_test_or_deploy_name'].astype(str)

df_metadata = pd.DataFrame(columns=['id','qty_minorClass','percetagem_minorClass','qty_features','qty_instances'])

for file_name in columns_name:
    input_filesName[file_name] = path + input_filesName[file_name].astype(str) + '.csv'

## script 1
for i in range(0, input_filesName.shape[0]):

    print(str(input_filesName.iloc[i][0]))
    
    check_output("python decodIRT_OtML.py -data " +str(input_filesName.iloc[i][0])+ " -dataTest " +str(input_filesName.iloc[i][1]), shell=True)
    data_train = pd.read_csv(str(input_filesName.iloc[i][0]))

    qty_class_0 = list(data_train.groupby('y').count()['Unnamed: 0'])[0]
    qty_class_1 = list(data_train.groupby('y').count()['Unnamed: 0'])[1]
    qty_instances = qty_class_0 + qty_class_1
    qty_minorClass = 0

    if qty_class_0 > qty_class_1:
        qty_minorClass = qty_class_1
    else:
        qty_minorClass = qty_class_0

    percetagem_minorClass = qty_minorClass/qty_instances
    name_train_dataset = str(input_filesName.iloc[i][0])
    name_train_dataset = name_train_dataset.replace('dataset_partitions/','')
    name_train_dataset = name_train_dataset.replace('.csv','')
    qty_features = data_train.shape[1]

    data_train = data_train.drop(['Unnamed: 0','y'], axis=1)

    # mydata_z = data_train.apply(zscore)

    # km = KMeans(n_clusters=3, random_state=0)
    # mydata_z = mydata_z.iloc[:,:100]
    # mydata_z = mydata_z.fillna(0)
    # km.fit(mydata_z)

    # silhouette_value = silhouette_score(mydata_z, km.predict(mydata_z))

    df2 = {'id': name_train_dataset, 'qty_minorClass': qty_minorClass, 'percetagem_minorClass': percetagem_minorClass,
    'qty_instances':qty_instances, 'qty_features':qty_features}

    df_metadata = df_metadata.append(df2, ignore_index = True)
    


print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")    

## script 2
check_output("python decodIRT_MLtIRT.py", shell=True)


input_filesName_test = input_filesName[input_filesName["id"].str.contains("test")]
input_filesName_test = input_filesName_test[input_filesName_test["id"].str.contains("default")]


contador = 0

input_filesName_test = input_filesName_test.reset_index()

for i in range(0, input_filesName_test.shape[0]):

    contador = contador + 1

    file_name = input_filesName_test['id'][i]

    print(file_name)

    # dataset_name = input_filesName_test['data_test_or_deploy_name'][0].split("\\")[1].replace("_test.csv","").replace("default_","")
    dataset_name = input_filesName_test['data_train_name'][i].split("/")[1].replace("_train.csv","").replace("default_","")


    input_filesName_test['data_test_or_deploy_name'][i].split("/")[1].replace(".csv","")

    ## parametros IRT

    # IRT: default - test
    path_hiperparameters = f"{'output'}/{file_name}/{'irt_item_param.csv'}"
    hiperparameters = pd.read_csv(path_hiperparameters)
    df_hiperparameters = DataframeFeatures_fromHiperparameters(
        hiperparameters=hiperparameters, 
        dataset_name=dataset_name, 
        dataset_type="default",
        dataset_partition='test')
    df_hiperparameters_combined = df_hiperparameters

    # IRT: under - test
    path_hiperparameters = f"{'output'}/{file_name}/{'irt_item_param.csv'}"
    path_hiperparameters = path_hiperparameters.replace("default", "under")
    hiperparameters = pd.read_csv(path_hiperparameters)
    df_hiperparameters = DataframeFeatures_fromHiperparameters(
        hiperparameters=hiperparameters, 
        dataset_name=dataset_name, 
        dataset_type="under",
        dataset_partition='test')
    df_hiperparameters_combined = pd.concat([df_hiperparameters_combined, df_hiperparameters])
    
    # IRT: default - deploy
    path_hiperparameters = f"{'output'}/{file_name}/{'irt_item_param.csv'}"
    path_hiperparameters_deploy = path_hiperparameters.replace("test", "deploy")
    hiperparameters = pd.read_csv(path_hiperparameters_deploy)
    df_hiperparameters = DataframeFeatures_fromHiperparameters(
        hiperparameters=hiperparameters, 
        dataset_name=dataset_name, 
        dataset_type="default",
        dataset_partition='deploy')
    df_hiperparameters_combined = pd.concat([df_hiperparameters_combined, df_hiperparameters])

    # IRT: under - deploy
    path_hiperparameters = path_hiperparameters_deploy.replace("default", "under")
    hiperparameters = pd.read_csv(path_hiperparameters)
    df_hiperparameters = DataframeFeatures_fromHiperparameters(
        hiperparameters=hiperparameters, 
        dataset_name=dataset_name, 
        dataset_type="under",
        dataset_partition='deploy')
    df_hiperparameters_combined = pd.concat([df_hiperparameters_combined, df_hiperparameters])


    ## acuracia 
    path_test = f"{'output'}/{file_name}/{file_name}{'_final.csv'}"
    data_test = pd.read_csv(path_test)
    data_test["Metodo"] = data_test["Metodo"].astype(str) + '_test'
    data_test = data_test.set_index('Metodo').transpose().copy()

    path_test_under = path_test.replace("default", "under")
    data_test_under = pd.read_csv(path_test_under)
    data_test_under["Metodo"] = data_test_under["Metodo"].astype(str) + '_test_under'
    data_test_under = data_test_under.set_index('Metodo').transpose().copy()

    path_deploy = path_test.replace("test", "deploy")
    data_deploy = pd.read_csv(path_deploy)
    data_deploy["Metodo"] = data_deploy["Metodo"].astype(str) + '_deploy'
    data_deploy = data_deploy.set_index('Metodo').transpose().copy()

    path_deploy_under = path_deploy.replace("default", "under")
    data_deploy_under = pd.read_csv(path_deploy_under)
    data_deploy_under["Metodo"] = data_deploy_under["Metodo"].astype(str) + '_deploy_under'
    data_deploy_under = data_deploy_under.set_index('Metodo').transpose().copy()

    ## F1 - score
    path_test_f1 = f"{'output'}/{file_name}/{file_name}{'_final_f1Score.csv'}"
    data_test_f1 = pd.read_csv(path_test_f1)
    data_test_f1["Metodo"] = data_test_f1["Metodo"].astype(str) + '_test'
    data_test_f1 = data_test_f1.set_index('Metodo').transpose().copy()

    path_test_under_f1 = path_test_f1.replace("default", "under")
    data_test_under_f1 = pd.read_csv(path_test_under_f1)
    data_test_under_f1["Metodo"] = data_test_under_f1["Metodo"].astype(str) + '_test_under'
    data_test_under_f1 = data_test_under_f1.set_index('Metodo').transpose().copy()

    path_deploy_f1 = path_test_f1.replace("test", "deploy")
    data_deploy_f1 = pd.read_csv(path_deploy_f1)
    data_deploy_f1["Metodo"] = data_deploy_f1["Metodo"].astype(str) + '_deploy'
    data_deploy_f1 = data_deploy_f1.set_index('Metodo').transpose().copy()

    path_deploy_under_f1 = path_deploy_f1.replace("default", "under")
    data_deploy_under_f1 = pd.read_csv(path_deploy_under_f1)
    data_deploy_under_f1["Metodo"] = data_deploy_under_f1["Metodo"].astype(str) + '_deploy_under'
    data_deploy_under_f1 = data_deploy_under_f1.set_index('Metodo').transpose().copy()


    combined_df = pd.concat([data_test, data_test_under, data_deploy, data_deploy_under], axis=1)
    combined_df.insert(loc=0, column='id', value=dataset_name)

    combined_df_f1 = pd.concat([data_test_f1, data_test_under_f1, data_deploy_f1, data_deploy_under_f1], axis=1)
    combined_df_f1.insert(loc=0, column='id', value=dataset_name)

    if contador == 1:
        result = combined_df
        result = pd.concat([result, combined_df_f1])

        hiperparameters_result = df_hiperparameters_combined
    else:
        result = pd.concat([result, combined_df,combined_df_f1])
        hiperparameters_result = pd.concat([hiperparameters_result, df_hiperparameters_combined])
    
hiperparameters_result = hiperparameters_result.reset_index()
hiperparameters_result = hiperparameters_result.drop(columns=['index'])


# Save the entire session by creating a new pickle file
dill.dump_session('./your_bk_dill.pkl')

# Restore the entire session
# dill.load_session('./your_bk_dill.pkl')

selected_columns_result_default = [
 'GaussianNB_test',
 'BernoulliNB_test',
 'KNeighborsClassifier(2)_test',
 'KNeighborsClassifier(3)_test',
 'KNeighborsClassifier(5)_test',
 'KNeighborsClassifier(8)_test',
 'DecisionTreeClassifier()_test',
 'RandomForestClassifier(3_estimators)_test',
 'RandomForestClassifier(5_estimators)_test',
 'RandomForestClassifier_test',
 'SVM_test',
 'MLPClassifier_test',
 'majoritario_test',
 'minoritario_test',
 'GaussianNB_deploy',
 'BernoulliNB_deploy',
 'KNeighborsClassifier(2)_deploy',
 'KNeighborsClassifier(3)_deploy',
 'KNeighborsClassifier(5)_deploy',
 'KNeighborsClassifier(8)_deploy',
 'DecisionTreeClassifier()_deploy',
 'RandomForestClassifier(3_estimators)_deploy',
 'RandomForestClassifier(5_estimators)_deploy',
 'RandomForestClassifier_deploy',
 'SVM_deploy',
 'MLPClassifier_deploy',
 'majoritario_deploy',
 'minoritario_deploy']

selected_columns_result_under = [
 'GaussianNB_test_under',
 'BernoulliNB_test_under',
 'KNeighborsClassifier(2)_test_under',
 'KNeighborsClassifier(3)_test_under',
 'KNeighborsClassifier(5)_test_under',
 'KNeighborsClassifier(8)_test_under',
 'DecisionTreeClassifier()_test_under',
 'RandomForestClassifier(3_estimators)_test_under',
 'RandomForestClassifier(5_estimators)_test_under',
 'RandomForestClassifier_test_under',
 'SVM_test_under',
 'MLPClassifier_test_under',
 'majoritario_test_under',
 'minoritario_test_under',
 'GaussianNB_deploy_under',
 'BernoulliNB_deploy_under',
 'KNeighborsClassifier(2)_deploy_under',
 'KNeighborsClassifier(3)_deploy_under',
 'KNeighborsClassifier(5)_deploy_under',
 'KNeighborsClassifier(8)_deploy_under',
 'DecisionTreeClassifier()_deploy_under',
 'RandomForestClassifier(3_estimators)_deploy_under',
 'RandomForestClassifier(5_estimators)_deploy_under',
 'RandomForestClassifier_deploy_under',
 'SVM_deploy_under',
 'MLPClassifier_deploy_under',
 'majoritario_deploy_under',
 'minoritario_deploy_under']


compressed_result = result.copy()
compressed_result['median_default'] = compressed_result[selected_columns_result_default].median(axis=1)
compressed_result['median_under'] = compressed_result[selected_columns_result_under].median(axis=1)
compressed_result = compressed_result[['id','median_default','median_under']]
compressed_result['metodo'] = compressed_result.index


###################### resultados acc e f1 para default 
compressed_result_default = compressed_result.copy()
compressed_result_default = compressed_result_default.drop('median_under', axis=1)
compressed_result_default['id'] = 'default_' + compressed_result_default['id']
compressed_result_default_acc = compressed_result_default[compressed_result_default['metodo'] == 'Acuracia']
compressed_result_default_f1 = compressed_result_default[compressed_result_default['metodo'] == 'f1Score']

compressed_result_default_acc = compressed_result_default_acc.drop(['metodo'], axis = 1)
compressed_result_default_acc.columns = ['id', 'median_acc']

compressed_result_default_f1 = compressed_result_default_f1.drop(['metodo'], axis = 1)
compressed_result_default_f1.columns = ['id', 'median_f1']


###################### resultados acc e f1 para UNDER 
compressed_result_under = compressed_result.copy()
compressed_result_under = compressed_result_under.drop('median_default', axis=1)
compressed_result_under['id'] = 'under_' + compressed_result_under['id']
compressed_result_under_acc = compressed_result_under[compressed_result_under['metodo'] == 'Acuracia']
compressed_result_under_f1 = compressed_result_under[compressed_result_under['metodo'] == 'f1Score']

compressed_result_under_acc = compressed_result_under_acc.drop(['metodo'], axis = 1)
compressed_result_under_acc.columns = ['id', 'median_acc']

compressed_result_under_f1 = compressed_result_under_f1.drop(['metodo'], axis = 1)
compressed_result_under_f1.columns = ['id', 'median_f1']

## New id for hiperparameters
hiperparameters_result['id'] = hiperparameters_result['dataset_type'] + "_" + hiperparameters_result['dataset_name'] 

## results acc
compressed_result_acc = pd.concat([compressed_result_default_acc, compressed_result_under_acc])

## results f1
compressed_result_f1 = pd.concat([compressed_result_default_f1, compressed_result_under_f1])


hiperparameters_result = hiperparameters_result.merge(compressed_result_acc, left_on='id', right_on='id')
hiperparameters_result = hiperparameters_result.merge(compressed_result_f1, left_on='id', right_on='id')

df_metadata_bkp = df_metadata.copy()

df_metadata_bkp.drop_duplicates(keep='first',inplace=True)
df_metadata_bkp['id'] = df_metadata_bkp['id'].str.replace('_train','')

hiperparameters_result = hiperparameters_result.merge(df_metadata_bkp, left_on='id', right_on='id')

hiperparameters_result['qty_minorClass'] = pd.to_numeric(hiperparameters_result['qty_minorClass'])
hiperparameters_result['qty_features'] = pd.to_numeric(hiperparameters_result['qty_features'])
hiperparameters_result['qty_instances'] = pd.to_numeric(hiperparameters_result['qty_instances'])

df_train = hiperparameters_result.copy()
df_train.loc[df_train['dataset_type'] == 'default', 'dataset_type'] = 0
df_train.loc[df_train['dataset_type'] == 'under', 'dataset_type'] = 1

df_train = hiperparameters_result.copy()
df_train = df_train.drop(columns=['id','dataset_partition','dataset_name', 'adivinhacao_min', 'dificuldade_min', 'discriminacao_min', 'adivinhacao_max', 'dificuldade_max', 'discriminacao_max'])
df_train.loc[df_train['dataset_type'] == 'default', 'dataset_type'] = 0
df_train.loc[df_train['dataset_type'] == 'under', 'dataset_type'] = 1
df_train['dataset_type'] = pd.to_numeric(df_train['dataset_type'])

# seed_values = [1,3,7,9,30,50,99]
# for seed_i in seed_values:
#     seed_value = seed_i

seed_value = 50

models_and_results = pd.DataFrame(columns=['model','model_name','acc_mean','f1_mean'], index=range(9))
models_count = 0


dtree = DecisionTreeClassifier()

model = xgb.XGBClassifier(learning_rate=0.001, max_depth=10, n_estimators=100, random_state=seed_value)
models_and_results['model'][models_count] = model
models_and_results['model_name'][models_count] = 'XGboost'
models_count +=1

model = RandomForestClassifier(n_estimators=100, random_state = seed_value)
models_and_results['model'][models_count] = model
models_and_results['model_name'][models_count] = 'RandomForest'
models_count +=1

model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=5, random_state=seed_value)
models_and_results['model'][models_count] = model
models_and_results['model_name'][models_count] = 'GradientBoosting'
models_count +=1

model = GaussianNB()
models_and_results['model_name'][models_count] = 'GaussianNB'
models_and_results['model'][models_count] = model
models_count +=1

model = BernoulliNB()
models_and_results['model_name'][models_count] = 'BernolliNB'
models_and_results['model'][models_count] = model
models_count +=1

model = AdaBoostClassifier(n_estimators=100, base_estimator=dtree,learning_rate=1, random_state=seed_value)
models_and_results['model'][models_count] = model
models_and_results['model_name'][models_count] = 'AdaBoostClassifier'
models_count +=1

model = cb.CatBoostClassifier(random_state=seed_value)
models_and_results['model'][models_count] = model
models_and_results['model_name'][models_count] = 'CatBoostClassifier'
models_count +=1

model = lgb.LGBMClassifier(random_state=seed_value)
models_and_results['model'][models_count] = model
models_and_results['model_name'][models_count] = 'LGBMClassifier'
models_count +=1

model = LogisticRegression(solver='newton-cg', random_state=seed_value)
models_and_results['model'][models_count] = model
models_and_results['model_name'][models_count] = 'LogisticRegression'
models_count +=1


dataset = df_train.copy()
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed_value)

x = dataset.copy()
x.drop(['dataset_type'],axis=1)
y = dataset.dataset_type



acc_vector = []
f1_vector = []


for id_model in range(0, models_count):

    fold_no = 1
    for train_index,test_index in skf.split(x, y):
        train = dataset.iloc[train_index,:]
        test = dataset.iloc[test_index,:]
        training(train, test, fold_no, models_and_results["model"][id_model])
        fold_no += 1


    models_and_results["acc_mean"][id_model] = (sum(acc_vector)/len(acc_vector))
    models_and_results["f1_mean"][id_model] = (sum(f1_vector)/len(f1_vector))


# file_name = "results/result_models_" + str(seed_value) + ".csv"
file_name = "results/result_models.csv"
models_and_results.to_csv(file_name)



##################################################################################################
############################################# ANALYSIS OF DIFFERENCE BETWEEN NORMAL AND UNRELIABLE ENVIRONMENTS
##################################################################################################

classificadores = ['GaussianNB', 'BernoulliNB',
    'KNeighborsClassifier(2)',
    'KNeighborsClassifier(3)',
    'KNeighborsClassifier(5)',
    'KNeighborsClassifier(8)',
    'DecisionTreeClassifier()',
    'RandomForestClassifier(3_estimators)',
    'RandomForestClassifier(5_estimators)',
    'RandomForestClassifier', 'SVM',
    'MLPClassifier', 'rand1', 'rand2',
    'rand3', 'majoritario', 'minoritario',
    'pessimo', 'otimo']


for classificador_name in classificadores:
    new_column_name = classificador_name + "_default_diff"
    column_1 = classificador_name + "_test"
    column_2 = classificador_name + "_deploy"
    result[new_column_name] = result[column_2].sub(result[column_1], axis = 0)

    new_UnderColumn_name = classificador_name + "_under_diff"
    under_column_1 = classificador_name + "_test_under"
    under_column_2 = classificador_name + "_deploy_under"
    result[new_UnderColumn_name] = result[under_column_2].sub(result[under_column_1], axis = 0)



diferences_default = ['id','GaussianNB_default_diff', 'BernoulliNB_default_diff',
    'KNeighborsClassifier(2)_default_diff',
    'KNeighborsClassifier(3)_default_diff',
    'KNeighborsClassifier(5)_default_diff',
    'KNeighborsClassifier(8)_default_diff',
    'DecisionTreeClassifier()_default_diff',
    'RandomForestClassifier(3_estimators)_default_diff',
    'RandomForestClassifier(5_estimators)_default_diff',
    'RandomForestClassifier_default_diff', 'SVM_default_diff',
    'MLPClassifier_default_diff']


diferences_under = ['id','GaussianNB_under_diff', 'BernoulliNB_under_diff',
    'KNeighborsClassifier(2)_under_diff',
    'KNeighborsClassifier(3)_under_diff',
    'KNeighborsClassifier(5)_under_diff',
    'KNeighborsClassifier(8)_under_diff',
    'DecisionTreeClassifier()_under_diff',
    'RandomForestClassifier(3_estimators)_under_diff',
    'RandomForestClassifier(5_estimators)_under_diff',
    'RandomForestClassifier_under_diff', 'SVM_under_diff',
    'MLPClassifier_under_diff']

acc_result = result[result.index=="Acuracia"]
diferences_under_data = acc_result[diferences_under]
diferences_default_data = acc_result[diferences_default]

result_f1Score = result[result.index=="f1Score"]
diferences_under_data_f1Score = result_f1Score[diferences_under]
diferences_default_data_f1Score = result_f1Score[diferences_default]



###################################################### acuraria

plot1_visualization_default = diferences_default_data[diferences_default_data.id == 'breast-w']
plot1_visualization_under = diferences_under_data[diferences_under_data.id == 'breast-w']


colunms_name = list(plot1_visualization_default.columns)

new_colunms_name = []

for string in colunms_name:
    new_string = string.replace("_default", "")
    new_colunms_name.append(new_string)


plot1_visualization_default.columns = new_colunms_name
plot1_visualization_default.loc[:, ('env')] = 'normal'

plot1_visualization_under.columns = new_colunms_name
plot1_visualization_under.loc[:, ('env')] = 'unreliable'

plot1_visualization = pd.concat([plot1_visualization_default, plot1_visualization_under])

plot1_visualization = plot1_visualization.drop(columns=['id'])

plot1_visualization_t = plot1_visualization.transpose()
plot1_visualization_t['id'] = plot1_visualization_t.index
plot1_visualization_t = plot1_visualization_t[:-1]
plot1_visualization_t.columns = ['normal','unreliable', 'id']

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable

plt.rcParams["figure.figsize"] = (11,6)
plt.rcParams['font.size'] = 13
plot1_visualization_t[['normal','unreliable']] = abs(plot1_visualization_t[['normal','unreliable']])
ax = plot1_visualization_t.plot.barh()
make_axes_area_auto_adjustable(ax)
plt.savefig('results/breast-w.png')



###################################################### acuraria

plot2_visualization_default = diferences_default_data[diferences_default_data.id == 'wdbc']
plot2_visualization_under = diferences_under_data[diferences_under_data.id == 'wdbc']

colunms_name = list(plot2_visualization_default.columns)

new_colunms_name = []

for string in colunms_name:
    new_string = string.replace("_default", "")
    new_colunms_name.append(new_string)


plot2_visualization_default.columns = new_colunms_name
plot2_visualization_default.loc[:, ('env')] = 'normal'

plot2_visualization_under.columns = new_colunms_name
plot2_visualization_under.loc[:, ('env')] = 'unreliable'

plot2_visualization = pd.concat([plot2_visualization_default, plot2_visualization_under])

plot2_visualization = plot2_visualization.drop(columns=['id'])

plot2_visualization_t = plot2_visualization.transpose()
plot2_visualization_t['id'] = plot2_visualization_t.index
plot2_visualization_t = plot2_visualization_t[:-1]
plot2_visualization_t.columns = ['normal','unreliable', 'id']

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11,6)
plt.rcParams['font.size'] = 13
plot2_visualization_t[['normal','unreliable']] = abs(plot2_visualization_t[['normal','unreliable']])
ax = plot2_visualization_t.plot.barh()
make_axes_area_auto_adjustable(ax)
plt.savefig('results/wdbc.png')