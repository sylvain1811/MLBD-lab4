def calculateAccuracy(tn, fp, fn, tp):
    """
    return the accuracy based on the confusion matrix values

    :param tn: Quantity of True negative
    :param fp: Quantity of False positive
    :param fn: Quantity of False negative
    :param tp: Quantity of True positive

    :type tn: int - required
    :type fp: int - required
    :type fn: int - required
    :type tp: int - required

    :return: The accuracy value
    :rtype: Float
    """
    acc = (tp + tn) / (tp + tn + fp + fn)
    return acc

def calculateSensitivity(tn, fp, fn, tp):
    """
    return the Sensitivity based on the confusion matrix values

    :param tn: Quantity of True negative
    :param fp: Quantity of False positive
    :param fn: Quantity of False negative
    :param tp: Quantity of True positive

    :type tn: int - required
    :type fp: int - required
    :type fn: int - required
    :type tp: int - required

    :return: The sensitivity value
    :rtype: Float
    """
    sen = tp / (tp + fn)
    return sen

def calculateSpecificity(tn, fp, fn, tp):
    """
    return the Specificity based on the confusion matrix values

    :param tn: Quantity of True negative
    :param fp: Quantity of False positive
    :param fn: Quantity of False negative
    :param tp: Quantity of True positive

    :type tn: int - required
    :type fp: int - required
    :type fn: int - required
    :type tp: int - required

    :return: The speicificity value
    :rtype: Float
    """
    spe = tn / (tn + fp)
    return spe
    
def calculateF1(tn, fp, fn, tp):
    """
    return the F1 score based on the confusion matrix values

    :param tn: Quantity of True negative
    :param fp: Quantity of False positive
    :param fn: Quantity of False negative
    :param tp: Quantity of True positive

    :type tn: int - required
    :type fp: int - required
    :type fn: int - required
    :type tp: int - required

    :return: The F1 socre value
    :rtype: Float
    """
    f1_score = (2 * tp) / ( 2 * tp + fp + fn)
    return f1_score