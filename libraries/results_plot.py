from libraries.ConfusionMatrix import ConfusionMatrix
import numpy as np



def plotCMByTreflePredictions(y_true, y_pred):
    results = []
    for element in y_pred:
        if element > 0.5:
            results.append(1)
        else:
            results.append(0)

    cm = confusion_matrix(y_true, results)
    n_classes = len(np.unique(y))
    ConfusionMatrix.plot(cm, classes=range(n_classes), title="Confusion Matrix")