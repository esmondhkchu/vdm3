import pytest
from vdm import *

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

var_val = [1,[1,],(1,)]
var_id = ['int', 'list', 'tuple']

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
##########################################################

columns = {
    'Gender':['F','F','F','M','F','F','F','F','M','F'],
    'Marital':['UN','S','M','M','S','M','M','S','D','M'],
    'Lead':['REF','INTINT','REF','INTINT','RADIO','REF','INTER','PPC','PPC','RADIO'],
    'PrevEd':['SOMECOLL','SOMECOLL','ASSOC','BACH','BACH','ASSOC','UN','SOMECOLL','BACH','SOMECOLL'],
    'Citizen':['US','US','US','US','US','ELNC','US','US','US','US']
}

X = pd.DataFrame(columns)
y = np.array([0,0,1,0,0,0,0,0,0,1])

##########################################################
################### Test get_cond_prob ###################
##########################################################

def test_get_cond_prob():
    test = ValueDifferenceMetric(X,y)
    cond_prob = test.get_cond_prob(np.array(['M','M','F','F']), np.array([1,0,1,0]))
    assert isinstance(cond_prob, dict)
    assert list(cond_prob.keys()) == ['F','M']
    assert list(cond_prob.get('F')) == [0.5,0.5]
    assert list(cond_prob.get('M')) == [0.5,0.5]

##########################################################
######################## Test vdm ########################
##########################################################

var_val = ['Gender','Marital','Lead','PrevEd','Citizen']
var_id = ['Gender','Marital','Lead','PrevEd','Citizen']

@pytest.mark.parametrize('test_type', var_val, ids=var_id)
def test_vdm(test_type):
    test = ValueDifferenceMetric(X,y)
    vdm = test.vdm(X[test_type], y)
    assert isinstance(vdm, dict)
    # check number of combination is correct
    n = len(np.unique(X[test_type]))
    assert len(vdm.keys()) == comb(n,2)

##########################################################
################### Test vdm_pairs_fit ###################
##########################################################

def test_vdm_pairs_fit():
    test = ValueDifferenceMetric(X,y)
    test.vdm_pairs_fit()
    assert isinstance(test.all_pairs, dict)
    assert len(test.all_pairs.keys()) == 5
    assert list(test.all_pairs.keys()) == list(X)

##########################################################
################### Test vdm_pairs_fit ###################
##########################################################

gd_point1 = ['F','D','INTER','ASSOC','ELNC']
gd_point2 = ['M', 'S', 'PPC', 'SOMECOLL', 'US']

bd_point1 = ['F','hello','world','ASSOC','ELNC']
bd_point2 = [1,2]

def test_get_points_distance_input():
    """ should raise an error if point dimensions are mismatch
    """
    test = ValueDifferenceMetric(X,y)
    test.vdm_pairs_fit()
    with pytest.raises(DimensionError):
        test.get_points_distance(bd_point2, gd_point1)

def test_get_points_distance():
    """ test good points
    """
    test = ValueDifferenceMetric(X,y)
    test.vdm_pairs_fit()
    assert isinstance(test.get_points_distance(gd_point1, gd_point2), float)

def test_get_wrong_points_distance():
    """ should raise an error if points are not in training
    """
    test = ValueDifferenceMetric(X,y)
    test.vdm_pairs_fit()
    with pytest.raises(ValueError):
        assert test.get_points_distance(gd_point1, bd_point1)

def test_get_points_distance_zero():
    """ if two points input are the same,
        resultant distance (return) should be 0
    """
    test = ValueDifferenceMetric(X,y)
    test.vdm_pairs_fit()
    assert test.get_points_distance(gd_point1, gd_point1) == 0

##########################################################
##################### Test 1xd array #####################
##########################################################

X1 = np.array(['a','b','a','a','c'])
X2 = pd.Series(['a','b','a','a','c'])
y1 = np.array([0,1,0,1,1])

def test_one_dim_array_np():
    test = ValueDifferenceMetric(X1,y1)
    test.vdm_pairs_fit()
    assert isinstance(test.get_points_distance('a','b'), (float, int))
    assert test.get_points_distance('a','a') == 0

def test_one_dim_array_pd():
    test = ValueDifferenceMetric(X2,y1)
    test.vdm_pairs_fit()
    assert isinstance(test.get_points_distance('a','b'), (float, int))
    assert test.get_points_distance('a','a') == 0

def test_special_case():
    specx = np.array(['White','Red','Black','Red','Red','White'])
    specy = np.array([0,1,1,2,2,0])
    test = ValueDifferenceMetric(specx, specy)
    test.vdm_pairs_fit()
    assert test.get_points_distance(['White'],['Black'])
    assert isinstance(test.get_points_distance(['White'],['Black']), (float, int))
    assert round(test.get_points_distance(['White'],['Red']), 3) == 1.556 
    with pytest.raises(ValueError):
        assert test.get_points_distance(['White'],['Pink'])