import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

def plotDataFrameValues(vec_values_models):
    plt.scatter(vec_values_models['Sensitivity'],vec_values_models['Specificity'],s=10, marker='o')

    plt.title('Treashold sen/spe')
    plt.xlabel('Sensitivity')
    plt.ylabel('Specificity')
    plt.savefig('ScatterPlot.png')

    plt.xlim(0,1)
    plt.ylim(0,1)

    return plt.show()


def plotDataFrameValuesFiltered(value_sen, value_spe, dataframe_values_models, dataframe_values_models_invert):
    
    
    plt.scatter(dataframe_values_models['Sensitivity'],dataframe_values_models['Specificity'],s=10, marker='o', color='blue')
    plt.scatter(dataframe_values_models_invert['Sensitivity'],dataframe_values_models_invert['Specificity'],s=10, marker='o', color='aqua')

    plt.title('Treashold sen/spe')
    plt.xlabel('Sensitivity')
    plt.ylabel('Specificity')
    plt.savefig('ScatterPlot.png')

    plt.axvline(value_sen, color='red')
    plt.axhline(value_spe, color='green')
    
    index_labels = dataframe_values_models.index.tolist()
    sen_values = dataframe_values_models['Sensitivity'].tolist()
    spe_values = dataframe_values_models['Specificity'].tolist()
    
    #for sen_value_posi, spe_value_posi, index_label in zip(sen_values, spe_values,index_labels):
    #    plt.text(sen_value_posi+.03, spe_value_posi+.03, index_label, fontsize=9)

    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()


def plotHistogramFreqVar(dict_var_freq):
    #Sort the dictionary by values
    sorted_dict_values = sorted(dict_var_freq.items(), key=lambda kv: kv[1], reverse=True)
    #Transform in a dictionary
    dict_sorted_values = dict((k[0],k[1]) for k in sorted_dict_values)

    ax = plt.figure().gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Variable name')
    ax.set_ylabel('Quantity of models where it appear')

    plt.bar(list(dict_sorted_values.keys()), dict_sorted_values.values(), color='g')
    plt.show()


def plotFreqVarPerFreqMinimum(datafram_values_freq_by_model):
    #plot dataframe
    # gca stands for 'get current axis'
    ax = plt.figure().gca()

    datafram_values_freq_by_model.plot(kind='line',x='min freq var',y='number of models',ax=ax)
    datafram_values_freq_by_model.plot(kind='line',x='min freq var',y='quantity of variables', color='red', ax=ax)

    plt.show()



