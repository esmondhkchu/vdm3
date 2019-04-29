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

class ValueDifferenceMetric:

    def __init__(self, X, y):
        # check input variable types
        if not isinstance(X, (pd.DataFrame, np.ndarray, pd.core.series.Series)):
            raise TypeError('Wrong variable type: X')
        else:
            pass

        if not isinstance(y, (list, tuple, np.ndarray, pd.core.series.Series)):
            raise TypeError('Wrong variable type: y')
        elif isinstance(y, pd.DataFrame) and (y.shape[1] > 1):
            raise ValueError('Dimension mismatch')
        else:
            pass

        # check dimension
        if X.shape[0] != len(y):
            raise DimensionError('Dimension mismatch')
        else:
            pass

        self.X = X
        self.y = np.array(y)
        self.col_name = list(X)

    @staticmethod
    def get_cond_prob(x,y):
        """ get conditional probability of an array

        arg: x (1d-array) - a single categorical variable
             y (1d-array) - class value

        return: dict - keys = class values
                       values = values of y

        >>> test = ValueDifferenceMetric(np.array(['a','b']), np.array([0,1])) 
        >>> test.get_cond_prob(np.array(['a','a','b','b']),np.array([0,1,1,0]))
        {'a': array([0.5, 0.5]), 'b': array([0.5, 0.5])}
        """

        def one_prob(val):
            val_arr = np.array(y)[x == val]
            count = np.unique(val_arr, return_counts=True)
            if all(count[0] == np.unique(y)):
                return count[1]/sum(count[1])
            else:
                new_count = np.array([list(val_arr).count(i) for i in np.unique(y)])
                return new_count/sum(new_count)

        unique_x = np.unique(x)
        cond_prob = {i:j for i,j in zip(unique_x, np.array([one_prob(i) for i in unique_x]))}
        return cond_prob

    def vdm(self,x,y):
        """ get vdm distances between each class of an array that's categorical in nature

        arg: x (1d-array) - a single categorical variable
             y (1d-array) - class value

        return: dict - keys = class pairs
                       values = vdm distances
        
        >>> test = ValueDifferenceMetric(np.array(['a','b']), np.array([0,1]))
        >>> test.vdm(np.array(['a','a','b','b']),np.array([0,1,1,0]))
        {('a', 'b'): 0.0}
        """
        unique_x = np.unique(x)
        cat_pairs = [(unique_x[i],unique_x[j]) for i in range(len(unique_x)) for j in range(len(unique_x)) if i<j]

        cond_prob = self.get_cond_prob(x,y)

        dist_paris = [sum((cond_prob.get(i[0]) - cond_prob.get(i[1]))**2) for i in cat_pairs]
        vdm_pairs = {i:j for i,j in zip(cat_pairs, dist_paris)}
        return vdm_pairs

    def vdm_pairs_fit(self):
        """ dictionary stored vdm distance of different pairs of each feature
        """
        all_pairs = {i:self.vdm(self.X[i], self.y) for i in self.col_name}
        self.all_pairs = all_pairs

    def get_points_distance(self, point1, point2, metric=2):
        """ input 2 points,
            return VDM distance
            return 0 if categorical features are the same

        arg: point1 (array)
             point2 (array)
             metric (int) - 1 for 1-norm, 2 for 2-norm (default)

        return: float
        """

        if (len(point1) != self.X.shape[1]) or (len(point2) != self.X.shape[1]):
            raise DimensionError('Dimension mismatch')
        else:
            pass

        dist_pat = list(zip(point1, point2))
        try:
            vdm_dist = list(self.all_pairs.values())
        except:
            temp1 = self.all_vdm_pairs()
            vdm_dist = list(self.all_pairs.values())

        # warning if any variables in point1 or point2 are not in trianing data
        for i in range(len(self.all_pairs.keys())):
            name = list(self.all_pairs.keys())[i]
            if (dist_pat[i] not in self.all_pairs.get(name).keys()) and (dist_pat[i][0] != dist_pat[i][1]):
                raise ValueError('{} is not a distance pairs in training.'.format(dist_pat[i]))
            else:
                pass

        result_dist = [vdm_dist[i].get(tuple(sorted(dist_pat[i]))) for i in range(len(dist_pat))]
        var_dist = np.array([i if i != None else 0 for i in result_dist])
        if metric == 2:
            dist = np.sqrt(np.sum(var_dist**2))
        elif metric == 1:
            dist = np.sum(np.absolute(var_dist))
        else:
            raise ValueError('Wrong distance metric input')
        return dist

