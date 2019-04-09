import numpy as np
from itertools import tee
import libraries.trefle_project
from sklearn.metrics import accuracy_score

from os import listdir
from os.path import isfile, join

import pandas as pd



class ModelTrain(object):

    def __init__(self, array_index_train_test, X_train, y_train, number_rule, var_per_rule, classifier_trefle, path_save_results, experience_name, path_save_results_values = 'values.csv'):
        ## 0: param_a value 1: param_b value 2: cv_number 3:tn 4:fp 5:fn 6:tp 
        self.results_scores_line = np.zeros([7])
        self.array_list_files_name = []
        self.results_scores_array = [np.zeros([7])]

        self.array_index_train_test = array_index_train_test
        self.X_train = X_train
        self.y_train = y_train

        self.number_rule = number_rule
        self.var_per_rule = var_per_rule

        self.classifier_trefle = classifier_trefle

        self.path_save_results = path_save_results

        self.path_save_results_values = path_save_results_values

        self.experience_name = experience_name



    #Save the results in the global dataframe
    def saveIntoDataFrame(self, tn, fp, fn, tp):
        
        self.results_scores_line[3] = tn
        self.results_scores_line[4] = fp
        self.results_scores_line[5] = fn
        self.results_scores_line[6] = tp


    #Train a model for each CV
    def TrainSystem(self, x_train, y_train, x_test, y_test, cv_number):
        
        y_sklearn = np.reshape(y_train, (-1, 1))
        
        
        #Changes parameters values
        self.classifier_trefle.n_rules = self.number_rule
        self.classifier_trefle.n_max_vars_per_rule = self.var_per_rule

        self.classifier_trefle.fit(x_train, y_sklearn)

        # Make predictions
        y_pred = self.classifier_trefle.predict_classes(x_test)

        self.classifier_trefle.print_best_fuzzy_system()

        # Evaluate accuracy
        score = accuracy_score(y_test, y_pred)
        #print("Score on test set: {:.3f}".format(score))
        
        
        
        tn, fp, fn, tp = libraries.trefle_project.getConfusionMatrixValues(y_test, y_pred)
        #print("TP: {0}, TN: {1}, FP: {2}, FN: {3}".format(tn, fp, fn, tp))
        

        tff = self.classifier_trefle.get_best_fuzzy_system_as_tff()

        # Export: save the fuzzy model to disk
        path_save = self.path_save_results
        name_model = self.experience_name + "_conf_A_CV_" + str(cv_number) + "_rule_" + str(self.number_rule) + "_var_per_rule_" + str(self.var_per_rule) + ".ftt"
        path_save = path_save + name_model
        with open(path_save, mode="w") as f:
            f.write(tff)
            
        self.saveIntoDataFrame(tn, fp, fn, tp)    
        self.results_scores_line[0] = self.number_rule
        self.results_scores_line[1] = self.var_per_rule
        
        self.array_list_files_name.append(path_save)    
        #self.array_list_files_name.append(name_model)    

        print('save end')

    def execute_cv(self):
        
        cv_number = 0
        array_index_train_test, array_index_train_test_copy = tee(self.array_index_train_test)
        #self.array_index_train_test = array_index_train_test_copy_a
        for train_index, test_index in array_index_train_test_copy:
            X_train_cv, X_test_cv = self.X_train[train_index], self.X_train[test_index]
            y_train_cv, y_test_cv = self.y_train[train_index], self.y_train[test_index]
            self.TrainSystem(X_train_cv, y_train_cv, X_test_cv, y_test_cv, cv_number)
            
                
        
            self.results_scores_line[2] = cv_number
            
            cv_number += 1
            
            self.results_scores_array = np.vstack([self.results_scores_array, self.results_scores_line])
        
        del array_index_train_test_copy

        self.saveResultsIntoCSVFile(self.path_save_results_values, self.path_save_results )


    def getListModels(self, path_models_directory):


        onlyfiles = [f for f in listdir(path_models_directory) if isfile(join(path_models_directory, f))]

        array_list_files_name = onlyfiles

        return array_list_files_name



    def saveResultsIntoCSVFile(self, path_to_save:str, path_directory_models:str):
        #Treatment of the list values results 
        results_scores_array = self.results_scores_array

        array_list_files_name = self.getListModels(path_directory_models)

        qty_exp = len(array_list_files_name)
        #print(len(array_list_files_name))
        array_list_files_re = np.array(array_list_files_name).reshape(qty_exp,1)
        results_scores_array = np.delete(results_scores_array, 0, 0)
        results_scores_array = np.hstack((results_scores_array, array_list_files_re))


        datafram_results = pd.DataFrame(results_scores_array ,columns=['param_a', 'param_b', 'CV_num', 'tn', 'fp','fn','tp','file_name'])

        #The results are savec in "values.csv" file. It is used in the second part of your analyse
        datafram_results.to_csv(path_or_buf=path_to_save, sep=','
                        , float_format="{:.2f}".format(5), columns=None, header=True, index=False)


