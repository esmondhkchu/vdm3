import pytest
from vdm3 import *

import pandas as pd
import numpy as np

from scipy.special import comb

##########################################################
################# Test X variable type ###################
##########################################################

var1 = pd.DataFrame({'one':[1,2,3],'two':[2,3,4]})
var2 = np.array(var1)
var3 = var1.one

var_val = [var1, var2, var3]
var_id = ['dataframe','ndarray','pd.core.series']

@pytest.mark.parametrize('test_type', var_val, ids=var_id)
def test_good_Xinput_variable_type(test_type):
    """ check good variable type
        should all pass
    """
    test = ValueDifferenceMetric(test_type, [1,2,3])

var_val = [1,[1,],(1,),'test']
var_id = ['int', 'list', 'tuple','str']

@pytest.mark.parametrize('test_type', var_val, ids=var_id)
def test_bad_Xinput_variable_type(test_type):
    """ check bad variable type
        should raise TypeError if input is wrong type
        should all pass
    """
    with pytest.raises(TypeError):
        test = ValueDifferenceMetric(test_type, [1,2,3])

##########################################################
################# Test y variable type ###################
##########################################################

var4 = [1,2,3]
var5 = (1,2,3)
var_val = [var2, var3, var4, var5]
var_id = ['ndarray','pd.core.series','list', 'tuple']

@pytest.mark.parametrize('test_type', var_val, ids=var_id)
def test_good_yinput_variable_type(test_type):
    """ check good variable type
        should all pass
    """
    test = ValueDifferenceMetric(var1, test_type)

var_val = [1, '1']
var_id = ['int', 'str']

@pytest.mark.parametrize('test_type', var_val, ids=var_id)
def test_bad_yinput_variable(test_type):
    """ check bad variable type
        should raise TypeError if input is wrong type
        should all pass
    """
    with pytest.raises(TypeError):
        test = ValueDifferenceMetric(var1, test_type)

def test_y_dimension_mismatch():
    """ y should be a nx1 array
        if input is a multidimensional array
        should raise an error
    """
    with pytest.raises(TypeError):
        test = ValueDifferenceMetric(var1, var1)

##########################################################
########### Test "continuous" Variable Type ##############
##########################################################

dim_var_1 = (1,)
dim_var_2 = [1]
dim_var_3 = np.array([1])
dim_var_4 = pd.Series([1])

var_val = [dim_var_1, dim_var_2, dim_var_3, dim_var_4]
var_id = ['tuple', 'list', 'ndarray', 'pd.core.series']

@pytest.mark.parametrize('test_type', var_val, ids=var_id)
def test_continuous_parameter(test_type):
    """ test continuous parameter data type
        note: continuous parameter is a container collects the column number of continuous data
    """
    test = ValueDifferenceMetric(var1, [1,2,3], test_type)

var_val = [1, '1']
var_id = ['int', 'str']

@pytest.mark.parametrize('test_type', var_val, ids=var_id)
def test_bad_continuous_parameter(test_type):
    """ check bad variable type for continuous parameter
        should raise TypeError if input is wrong type
        should all pass
    """
    with pytest.raises(TypeError):
        test = ValueDifferenceMetric(var1, [1,2,3], test_type)

##########################################################
############# Test X and y dimension issue ###############
##########################################################

def test_X_y_dimension_mismatch():
    """ check if raise if X and y dimension mismatch
        X: 2x3
        y: 3x1
    """
    with pytest.raises(DimensionError):
        test = ValueDifferenceMetric(np.array([[1,2,3],[1,2,3]]), [1,2,3])

##########################################################
################# Test ins_1 and ins_2  ##################
##########################################################

X = pd.DataFrame({'color':['White','Red','Black','Red','Red','White'], \
                  'mpg':[23,28,32,42,40,20]})
y = np.array(['van','sport','sport','sedan','sedan','van'])

case = ValueDifferenceMetric(X,y,[1])
case.fit()

var = [('White',2),['White',2], np.array(['White',2]), pd.Series(['White',2])]
var_id = ['tuple','list','ndarray','pd.core.series']

@pytest.mark.parametrize('test_type', var, ids=var_id)
def test_get_distance_proper_case(test_type):
    """
    """
    case.get_distance(test_type, test_type) == 0

def test_get_distance_datatype():
    """ test if non proper data type in self.get_distance()
        with raise an TypeError error
    """
    with pytest.raises(TypeError):
        case.get_distance(1,2)

def test_get_distance_dimension():
    """ test if dimension mismatch with ins_1 and ins_2 with training data will raise a dimension error
    """
    with pytest.raises(DimensionError):
        case.get_distance([1,2,3,4],[1,2,3,4])

def test_get_distance_dimension_ins1_neq_ins2():
    """ test if dimension of ins_1 not equal to ins_2 will raise a dimension error
    """
    with pytest.raises(DimensionError):
        case.get_distance([1,2],[1,2,3])

##########################################################
###################### Test Values #######################
##########################################################

input_dim_0 = ['White','Red','Black','Red','Red','White']
input_dim_1 = [23,28,32,42,40,20]
expected = [5.153,0,4.110,14,12,8.097]

test_data = [[[i,j],k] for i,j,k in zip(input_dim_0, input_dim_1, expected)]

var_id = ['test_{}'.format(i) for i in range(6)]

@pytest.mark.parametrize('test_ins, expected', test_data, ids=var_id)
def test_vdm_calculation(test_ins, expected):
    assert round(case.get_distance(test_ins, np.array(['Red', 28])), 3) == expected
