from trefle.fitness_functions.output_thresholder import round_to_cls
from sklearn.metrics import confusion_matrix

import libraries.measures_calculation
import pandas as pd 
import numpy as np

def getConfusionMatrixValues(y_true, y_pred):
    """
    return tcross validation matrix

    :param y_true: True labels
    :param y_pred: Labels predicted by the algorithm

    :type y_true: [[int]] - required
    :type y_pred: [[int]] - required

    :return: The confusion matrix
    :rtype: Float
    """
    y_pred_bin = round_to_cls(y_pred, n_classes=2)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    return tn, fp, fn, tp



def calculateMean(dictionary_values):
    for key in dictionary_values:
        dictionary_values[key] = np.mean(dictionary_values[key])
        
    return dictionary_values

def calculateMeasures(df_values_experiments, vec_measures):
    dict_measures = { i : [] for i in vec_measures }
    for index, row in df_values_experiments.iterrows():
        tn, fp, fn, tp = row['tn'], row['fp'], row['fn'], row['tp']
        if 'acc' in vec_measures:
            acc = libraries.measures_calculation.calculateAccuracy(tn, fp, fn, tp)
            dict_measures['acc'].append(acc)

        if 'f1' in vec_measures:
            f1 = libraries.measures_calculation.calculateF1(tn, fp, fn, tp)
            dict_measures['f1'].append(f1)

        if 'sen' in vec_measures:
            sen = libraries.measures_calculation.calculateSensitivity(tn, fp, fn, tp)
            dict_measures['sen'].append(sen)

        if 'spe' in vec_measures:
            spe = libraries.measures_calculation.calculateSpecificity(tn, fp, fn, tp)
            dict_measures['spe'].append(spe)
            
    dict_measures = calculateMean(dict_measures)
    return dict_measures
    

def treatmentResultsValues(data_frame, parameter_a_name:str, parameter_b_name:str, vec_measures:list):
    df1 = data_frame.iloc[:,0:2]
    df1 = df1.drop_duplicates()
    #Get all different configurations
    qty_experiments = df1.shape[0]
    #Start tu calculate
    param_a_designation = 'param_a'
    param_b_designation = 'param_b'



    list_dict = []
    for index, row in df1.iterrows():
        df_experiment = data_frame.query("{0} == {1} and {2} == {3}".format(param_a_designation, row[0], param_b_designation, row[1]))
        results = calculateMeasures(df_experiment,vec_measures)
        list_dict.append(results)
    results_dataframe = pd.DataFrame(list_dict)
    param_a_list = df1['param_a'].tolist()
    param_b_list = df1['param_b'].tolist()

    results_dataframe.insert(loc=0, column='param_a', value=param_a_list)
    results_dataframe.insert(loc=1, column='param_b', value=param_b_list)

    return results_dataframe
