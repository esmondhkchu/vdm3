from vdm3.components import *

class ValueDifferenceMetric:
    def __init__(self, X, y, continuous=None):
        """ input data to compute distance

        Parameters: X (pd.Series/pd.DataFrame/nd-array) - independent variables, can be mix-type
                    y (array-like) - dependent variable
                    continuous (list) - a list of column numbers denoting continuous columns
                                        defaul is None, if None, then will take all columns as categorical data
        """
        # check input variable types
        if not isinstance(X, (pd.core.series.Series, pd.DataFrame, np.ndarray)):
            raise TypeError('Wrong input data type: X')

        if not isinstance(y, (tuple, list, np.ndarray, pd.core.series.Series)):
            raise TypeError('Wrong input data type: y')

        if continuous is not None:
            if not isinstance(continuous, (tuple, list, np.ndarray, pd.core.series.Series)):
                raise TypeError('continuous must be a container type object')

        # check dimension
        if len(X) != len(y):
            raise DimensionError('Dimension of X != Dimension of y')

        self.X = np.array(X)
        self.y = np.array(y)
        self.continuous = continuous

        if continuous is not None:
            self.cont_X = self.X[:, self.continuous]

            self.categorical = [i for i in range(self.X.shape[-1]) if i not in self.continuous]
            self.cat_X = self.X[:, self.categorical]
        else:
            self.categorical = np.arange(self.X.shape[-1])
            self.cat_X = self.X

    def fit(self):
        """ fit data to creaet conditional probability for vdm calculation
        """
        self.cond_proba = get_conditional_proba_nd(self.cat_X, self.y)

    def get_distance(self, ins_1, ins_2, norm=2):
        """ calculate distance between two instances

        Parameters: ins_1 (array-like) - instance 1
                    ins_2 (array-like) - instance 2
                    norm (int) - type of norm use, default is 2

        Returns: dist (float) - vdm distance between the two instances
        """

        if not isinstance(ins_1, (tuple, list, np.ndarray, pd.core.series.Series)):
            raise TypeError('Wrong input data type: ins_1')

        if not isinstance(ins_2, (tuple, list, np.ndarray, pd.core.series.Series)):
            raise TypeError('Wrong input data type: ins_2')

        if len(ins_1) != self.X.shape[-1]:
            raise DimensionError('Dimension mismatch with training data: ins_1')

        if len(ins_2) != self.X.shape[-1]:
            raise DimensionError('Dimension mismatch with training data: ins_2')

        if len(ins_1) != len(ins_2):
            raise DimensionError('Dimension of ins_1 != Dimension of ins_2')

        ins_1 = np.array(ins_1)
        ins_2 = np.array(ins_2)

        if self.continuous is not None:
            ins_1_cat = ins_1[self.categorical]
            ins_2_cat = ins_2[self.categorical]

            ins_1_cont = ins_1[self.continuous].astype(float)
            ins_2_cont = ins_2[self.continuous].astype(float)
        else:
            ins_1_cat = ins_1
            ins_2_cat = ins_2

        cat_dist = get_delta_nd(self.cond_proba, ins_1_cat, ins_2_cat, norm)

        if self.continuous is not None:
            cont_dist = get_cont_dist(ins_1_cont, ins_2_cont, norm)
            dist = (np.concatenate([cat_dist, cont_dist]).sum())**(1/norm)
        else:
            dist = (cat_dist.sum())**(1/norm)

        return dist
