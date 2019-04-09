import libraries.measures_calculation
import pandas as pd 
import numpy as np
import json
from libraries.model_var import ModelVar
from collections import Counter


def loadData(path_file):
    """
    return a pandasDatafram object with the loaded data in the path_file csv

    :param path_file: path of the csv file

    :type path_file: string - required

    :return: datafram object with all the data in the csv
    :rtype: Dataframe
    """
    data = pd.read_csv(path_file) 
    data.head()
    return data


def getSenSpeValuesByScores(datafram_scores):
    """
    return a a dataframe object with the sensitivity and specificity for all the models in the dataframe scores (tn,fp,fn and tp).
    E.G:
    +---------+---------+--------+-----+------+------+-----+-------------------+
    | param_a | param_b | CV_num | tn  |  fp  |  fn  | tp  |     file_name     |
    +---------+---------+--------+-----+------+------+-----+-------------------+
    |     1.0 |     0.0 |    0.0 | 3.0 | 46.0 | 72.0 | 6.0 | conf_A_C ... .ftt |
    +---------+---------+--------+-----+------+------+-----+-------------------+

    :param datafram_scores: datafram with all the tn,fp,fn and tp of all the models loaded

    :type datafram_scores: Dataframe - required


    E.G: return
    +-------+-------------+-------------+
    | index | Sensitivity | Specificity |
    +-------+-------------+-------------+
    |     0 |    0.076923 |    0.061224 |
    +-------+-------------+-------------+
    :return: datafram object with the score for the models in the :datafram_scores
    :rtype: Dataframe
    """
    sen_spe_values_vec = []
    for index, row in datafram_scores.iterrows():
        sensitivity = libraries.measures_calculation.calculateSensitivity(row['tn'], row['fp'], row['fn'], row['tp'])
        specificity = libraries.measures_calculation.calculateSpecificity(row['tn'], row['fp'], row['fn'], row['tp'])
        model_file_name = row['file_name']
        sen_spe_values_vec.append([index, sensitivity, specificity, model_file_name])
        
    df = pd.DataFrame(sen_spe_values_vec, columns=['index', 'Sensitivity', 'Specificity', 'file_name'])
    return df


def filterDataframeBySenSpeLimit(value_sen, value_spe, dataframe_values_models):
    """
    return a pandasDatafram filtered by the sensitivity and specificity

    :param value_sen: limit of the sensitivity
    :param value_spe: limit of the specificity
    :param dataframe_values_models: all the models 

    :type value_sen: float - required
    :type value_spe: float - required
    :type dataframe_values_models: pandas Dataframe - required

    :return: datafram object with the models that are higher than the sensitivity and specificity specified
    :rtype: Dataframe
    """

    datafram_values_filtered = dataframe_values_models.query('Sensitivity >= {0} and Specificity >= {1}'.format(value_sen, value_spe))
    return datafram_values_filtered

def filterDataframeBySenSpeLimitContrary(value_sen, value_spe, dataframe_values_models):
    """
    return a pandasDatafram filtered by the sensitivity and specificity. This return the inverse of the limits

    :param value_sen: limit of the sensitivity
    :param value_spe: limit of the specificity
    :param dataframe_values_models: all the models 

    :type value_sen: float - required
    :type value_spe: float - required
    :type dataframe_values_models: pandas Dataframe - required

    :return: datafram object with the models that are higher than the sensitivity and specificity specified
    :rtype: Dataframe
    """

    datafram_values_filtered = dataframe_values_models.query('Sensitivity < {0} or Specificity < {1}'.format(value_sen, value_spe))
    return datafram_values_filtered


def getModelsByFiltrationSenSpeDataframe(vec_models_filtered_data, vec_all_models_data):
    """
    return a list with all the files models filtered by the sensitivity and specificity

    :param vec_models_filtered_data: dataframe with the indexs, sen, and spe of the selected models
    :param vec_all_models_data: dataframe with all the models

    :type vec_models_filtered_data: pandas Dataframe - required
    :type vec_all_models_data: pandas Dataframe - required

    :return: list of strings of the models path
    :rtype: list[string]
    """
    index_models = np.array(vec_models_filtered_data['index'].values)
    models_delects_ds = vec_all_models_data[vec_all_models_data.index.isin(index_models)]
    list_models_path = models_delects_ds['file_name'].values.tolist()
    return list_models_path


def getListRulesPerModel(path_model):
    """
    return a list of the vector rules given a path model file

    :param path_model: path of the model

    :type path_model: string - required

    :return: list of model_var
    :rtype: list[model_var]
    """
    with open(path_model) as data_file:    
        data_json = json.load(data_file)
    #pprint(len(data_json['rules']))
    list_rules = data_json['rules']
    return list_rules

def transformListRulesToModelVar(model_path, list_rules):
    """
    return a list of model_var given a vector of rules (according to the structure)

    :param model_path: path of the model
    :param list_rules: list of the rules (according to the structure) [[[['10', 1], ['25', 2]], [1.0]], [[['20', 2], ['14', 1], ['21', 0]], [1.0]], [[['22', 0]], [1.0]], [[['0', 2], ['17', 2]], [0.0]]]

    :type model_path: string - required
    :type list_rules: list(list(list())) - required

    :return: list of model_var
    :rtype: list[model_var]
    """
    dict_var_qty = {}
    for rule in list_rules:
        variables_in_rule = rule[0]
        for variable in variables_in_rule:
            if variable[0] in dict_var_qty:
                dict_var_qty[variable[0]] = dict_var_qty[variable[0]] + 1
            else:
                dict_var_qty[variable[0]] = 1
    model_var = ModelVar(model_path, dict_var_qty)
    return model_var

def transformModelsToModelVarObj(list_models_path):
    """
    return a list model_var obj (see this class to understand what is it) based on a list of models path

    :param list_models_path: dlist of models path selected

    :type list_models_path: list[string] - required

    :return: list of model_var
    :rtype: list[model_var]
    """
    list_models_vars = []
    for file_name_path in list_models_path:
        list_rules = getListRulesPerModel(file_name_path)
        model_var = transformListRulesToModelVar(file_name_path,list_rules)
        list_models_vars.append(model_var)
    return list_models_vars


def countVarFreq(list_models_vars_freq):
    """
    return a dictionary with the frequencies of all the variables
    If the variable appear two times in a model it only count one

    :param list_models_path: list of model_var

    :type list_models_path: list[model_var] - required

    :return: dictionary with the frequencies for all the variables
    :rtype: dict{'var_name: frequence}
    """
    list_variables_total = []
    for model_var_freq in list_models_vars_freq:
        variables_names = list(model_var_freq.dict_freq_var.keys())
        list_variables_total.extend(variables_names)
    
    counter_frec_variables = Counter(list_variables_total)
    dict_frec_variables = dict(counter_frec_variables)
    return dict_frec_variables


def sort_reverse_dictionary_by_values(dicitonary_values):
    """
    return a dictionary reverse sorted by its values

    :param dicitonary_values: normal dictionary

    :type dicitonary_values: dict{string:int} - required

    :return: dictionary reverse sorted by values
    :rtype: dict{'var_name: frequence}
    """
    #Sort the dictionary by values
    sorted_dict_values = sorted(dicitonary_values.items(), key=lambda kv: kv[1], reverse=True)
    #Transform in a dictionary
    dict_sorted_values = dict((k[0],k[1]) for k in sorted_dict_values)
    return dict_sorted_values


def reduceQtyVars(nb_min_var:int, dict_values:dict, list_models_var):
    """
    return a list of model_var that the quantities of each variable are upper than the np_min_ar

    :param nb_min_var: quantity of the minimum variables that you want to save
    :param dict_values: dictionary with the frequency variables
    :param list_models_var: list of all the model_var objects

    :type nb_min_var: integer - required
    :type dict_values: dict{string:int} - required
    :type list_models_var: list[model_var] - required

    :return: list with all the model_Var saved
    :rtype: list[model_var]
    """

    dict2 = dict_values.copy()
    #On garde les variables qui ont une freq inferieur au seuil
    dict2 = {k: v for k, v in dict2.items() if v < nb_min_var}


    list_var_remove = list(dict2.keys())
    list_index_remove = []
    index_value = 0
    for model_var in list_models_var:
        var_in_models = list(model_var.dict_freq_var.keys())

        exists_var = any(x in var_in_models for x in list_var_remove)
        if exists_var == True:
            list_index_remove.append(index_value)

        index_value =index_value +1
    list_index_remove= reversed(list_index_remove)
    for element in list_index_remove:
        list_models_var.pop(element)
        
    return list_models_var


def createPlotQtyVarPerModelByMinimumFreq(dict_values, list_models_vars):
    #list_models_vars_cpopy = list_models_vars.copy()
    nb_min_var = 1
    qty_models = -1
    qty_variables = -1

    vec_qty_models = []
    vec_qty_variables = []

    while qty_models != 0:
        list_models_vars_cpopy = list_models_vars.copy()


        list_model_var_resultant = reduceQtyVars(nb_min_var, dict_values, list_models_vars_cpopy)
        dict_values_resulant = countVarFreq(list_model_var_resultant)

        #indication of the number of models and variables
        qty_models = len(list_model_var_resultant)
        qty_variables = len(dict_values_resulant)

        vec_qty_models.append(qty_models)
        vec_qty_variables.append(qty_variables)


        nb_min_var += 1

    vec_index = np.arange(1,nb_min_var)
    matrix_data = np.stack((vec_index, vec_qty_models, vec_qty_variables))

    headers = ['min freq var', 'number of models', 'quantity of variables']

    dataframe_result = pd.DataFrame(matrix_data.T, columns=headers)

    return dataframe_result



def plotSenSpeQtyModelsByThreshold(dataframe_models_sen_spe):
    value_sen = 0.0
    value_spe = 0.0
    array_results = []
    while value_sen < 1:
        value_spe = 0
        while value_spe < 1:
            dataframe_result = dataframe_models_sen_spe.loc[(dataframe_models_sen_spe['Sensitivity'] > value_sen) & (dataframe_models_sen_spe['Specificity'] > value_spe)]
            number_rows = dataframe_result.shape[0]
            array_results.append([value_sen, value_spe, number_rows])
            value_spe += 0.1
        value_sen += 0.1
    df_results = pd.DataFrame(array_results)
    df_results.columns= ['sensitivity','specificity','qty_models']

    #df_results = df_results.astype('int64')
    df_results = df_results.drop_duplicates()
    return df_results