# vdm3

Value difference metric was introduced in 1986 to provide an appropriate distance function for symbolic attributes. It is based on the idea that the goal of finding the distance is to find the right class by looking at the following conditional probabilities. <br>
![](./equations/cond_prob.png) <br>
Then the distance is calculated by the Euclidean Distance or Manhattan Distance, for instance: <br>
![](./equations/distance.png) <br>

# Install

```
pip install vdm3
```

## Parameters:

```
ValueDifferenceMetric(X=X, y=y, continuous=None)
```

  - X: ndarray, DataFrame, Series
  - y: tuple, list, ndarray, Series
  - continuous: tuple, list, ndarray, Series - the column index of continuous variables, if default, will assume all the columns are categorical

# Usage
Consider the following example: <br>
```python
>>> X = pd.DataFrame({'color':['White','Red','Black','Red','Red','White'], 'mpg':[23,28,32,42,40,20]})
>>> y = np.array(['van','sport','sport','sedan','sedan','van'])
```
Initiate the example by: <br>
```python
>>> case = ValueDifferenceMetric(X=X, y=y, continuous=[1])
>>> case.fit()
```
Get the vdm distance of two points by:
```python
>>> one = np.array(['White',23])
>>> two = np.array(['Red',28])

>>> case.get_distance(ins_1=one, ins_2=two)
5.153208277913436
```
Return 0 if two points are the same: <br>
```python
>>> case.get_points_distance(ins_1=one, ins_2=one)
0.0
```
