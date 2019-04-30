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
        if isinstance(self.X, pd.DataFrame):
            self.col_name = list(self.X)
        else:
            pass

    @staticmethod
    def get_cond_prob(x,y):
        """ get conditional probability of an array

        arg: x (1d-array) - a single categorical variable
             y (1d-array) - class value

        return: dict - keys = class values
                       values = values of y
        """

        def one_prob(val):
            val_arr = np.array(y)[x == val]
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
        if isinstance(self.X, pd.DataFrame):
            all_pairs = {i:self.vdm(self.X[i], self.y) for i in self.col_name}
            self.all_pairs = all_pairs
        elif isinstance(self.X, (np.ndarray, pd.core.series.Series)):
            if len(self.X.shape) == 1:
                all_pairs = {0:self.vdm(self.X, self.y)}
                self.all_pairs = all_pairs
            elif len(self.X.shape) == 2:
                all_pairs = {i:self.vdm(self.X[:,i], self.y) for i in range(self.X.shape[0])}
                self.all_pairs = all_pairs

    def get_points_distance(self, point1, point2):
        """ input 2 points,
            return VDM distance
            return 0 if categorical features are the same

        arg: point1 (array)
             point2 (array)

        return: float
        """

        if len(self.X.shape) == 1:
            if (len(point1) != 1) or (len(point2) != 1):
                raise DimensionError('Dimension mismatch')
            else:
                pass
        else:
            if (len(point1) != self.X.shape[1]) or (len(point2) != self.X.shape[1]):
                raise DimensionError('Dimension mismatch')
            else:
                pass
            
        dist_pat = [tuple(sorted(i)) for i in zip(point1, point2)]
        try:
            vdm_dist = list(self.all_pairs.values())
        except:
            temp1 = self.all_vdm_pairs()
            vdm_dist = list(self.all_pairs.values())

        # error if any variables in point1 or point2 are not in trianing data
        for i in range(len(self.all_pairs.keys())):
            name = list(self.all_pairs.keys())[i]
            if (dist_pat[i] not in self.all_pairs.get(name).keys()) and (dist_pat[i][0] != dist_pat[i][1]):
                raise ValueError('{} is not a distance pairs in training.'.format(dist_pat[i]))
            else:
                pass

        result_dist = [vdm_dist[i].get(dist_pat[i]) for i in range(len(dist_pat))]
        var_dist = np.array([i if i != None else 0 for i in result_dist])
        dist = np.sqrt(np.sum(var_dist))
        return dist
