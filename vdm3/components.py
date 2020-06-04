import numpy as np
import pandas as pd

class Error(Exception):
    """ base class
    """
    pass

class DimensionError(Error):
    """ raise when dimension mismatch
    """
    pass

def get_conditional_proba(x,y):
    """ get conditional probability of x given that y

    Parameters: x (nd-array) - an array of categorical data
                y (nd-array) - an array of categorical data, the dependent variable

    Returns: class_proba (dict) - a dictionary collects all the conditional probability
                                  {unique_y: pd.Sereis of conditional probability}
    """
    unique_y = np.unique(y)
    class_proba = dict()

    # creaet conditional probability
    for unique_x in np.unique(x):
        cond_data = y[x == unique_x]
        cond_proba = pd.Series(cond_data).value_counts() / len(cond_data)

        append_val = [i for i in unique_y if i not in cond_proba]
        if len(append_val) > 0:
            append_series = pd.Series(np.zeros(len(append_val)), index=append_val)
            cond_proba = cond_proba.append(append_series)

        # create new key
        class_proba[unique_x] = cond_proba

    return class_proba

def get_delta(cond_proba, val_1, val_2, norm=2):
    """ get delta for vdm

    Parameters: cond_proba (dict) - a dictionary of conditional probability, extract from get_conditional_proba()
                val_1 (str/int/float) - value 1
                val_2 (str/int/float) - value 2
                norm (int) - type of norm use, default is 2

    Returns: delta (float) - the delta value for this instance
    """
    proba = np.array([abs(cond_proba[val_1][uni_x] - cond_proba[val_2][uni_x]) for uni_x in cond_proba[val_1].index])
    delta = (proba**norm).sum()
    return delta

def get_conditional_proba_nd(X, y):
    """ given a nd array, return the dimensional conditional probability given that y

    Parameters: X (nd-array) - an nd-array of categorical data
                y (nd-array) - an array of categorical data, the dependent variable

    Returns: dim_proba (dict) - a dictionary collects all the conditional probability
                                {col_num:{unique_y: pd.Sereis of conditional probability}}
    """
    dim_proba = {col:get_conditional_proba(X[:,col],y) for col in range(X.shape[1])}
    return dim_proba

def get_delta_nd(cond_proba, ins_1, ins_2, norm=2):
    """ get deltas for vdm for multidimensional data

    Parameters: cond_proba (dict) - a dictionary of conditional probability, extract from get_conditional_proba_nd()
                ins_1 (nd-array) - array 1
                ins_2 (nd-array) - array 2
                norm (int) - type of norm use, default is 2

    Returns: deltas (array) - the delta values between ins_1 and ins_2
    """
    deltas = np.array([get_delta(cond_proba[col], ins_1[col], ins_2[col], norm) for col in cond_proba])
    return deltas

def get_cont_dist(ins_1, ins_2, norm=2):
    """ get dimensional distance for continuous data

    Parameters: ins_1 (nd-array) - array 1
                ins_2 (nd-array) - array 2
                norm (int) - type of norm use, default is 2

    Returns: dist (array) - the dimenional distance between ins_1 and ins_2
    """
    dist = abs(ins_1 - ins_2)**norm
    return dist
