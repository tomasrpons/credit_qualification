Relevant Information:

    This file concerns credit card applications.  All attribute names
    and values have been changed to meaningless symbols to protect
    confidentiality of the data.

## Dataset description

First things first we will take a look at the different variables presented to us, as well as their data type

Number of Attributes: 15 + class attribute

Attribute Information:

    A1:	b, a.
    A2:	continuous.
    A3:	continuous.
    A4:	u, y, l, t.
    A5:	g, p, gg.
    A6:	c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff.
    A7:	v, h, bb, j, n, z, dd, ff, o.
    A8:	continuous.
    A9:	t, f.
    A10:	t, f.
    A11:	continuous.
    A12:	t, f.
    A13:	g, p, s.
    A14:	continuous.
    A15:	continuous.
    A16: +,-         (class attribute)




Using this information we can identify our target variable (**A16**). We can also see that it is a discrete variable, therefore we are facing a **Classification Problem**. It only has two possible values, "+" or "-".

Missing Attribute Values:
    37 cases (5%) have one or more missing values.  The missing
    values from particular attributes are:

    A1:  12
    A2:  12
    A4:   6
    A5:   6
    A6:   9
    A7:   9
    A14: 13

We have around 5% of missing values, we will have to deal with those shortly. Now we'll take a look at the data.


```python
import pandas as pd
import numpy as np
```


```python
df = pd.read_csv("data/crx3.data")
columns = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
           "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16"]
df.columns = columns
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A1</th>
      <th>A2</th>
      <th>A3</th>
      <th>A4</th>
      <th>A5</th>
      <th>A6</th>
      <th>A7</th>
      <th>A8</th>
      <th>A9</th>
      <th>A10</th>
      <th>A11</th>
      <th>A12</th>
      <th>A13</th>
      <th>A14</th>
      <th>A15</th>
      <th>A16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>58.67</td>
      <td>4.460</td>
      <td>u</td>
      <td>g</td>
      <td>q</td>
      <td>h</td>
      <td>3.04</td>
      <td>t</td>
      <td>t</td>
      <td>6</td>
      <td>f</td>
      <td>g</td>
      <td>43.0</td>
      <td>560</td>
      <td>+</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>24.50</td>
      <td>0.500</td>
      <td>u</td>
      <td>g</td>
      <td>q</td>
      <td>h</td>
      <td>1.50</td>
      <td>t</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>280.0</td>
      <td>824</td>
      <td>+</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>27.83</td>
      <td>1.540</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>3.75</td>
      <td>t</td>
      <td>t</td>
      <td>5</td>
      <td>t</td>
      <td>g</td>
      <td>100.0</td>
      <td>3</td>
      <td>+</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>20.17</td>
      <td>5.625</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>1.71</td>
      <td>t</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>s</td>
      <td>120.0</td>
      <td>0</td>
      <td>+</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>32.08</td>
      <td>4.000</td>
      <td>u</td>
      <td>g</td>
      <td>m</td>
      <td>v</td>
      <td>2.50</td>
      <td>t</td>
      <td>f</td>
      <td>0</td>
      <td>t</td>
      <td>g</td>
      <td>360.0</td>
      <td>0</td>
      <td>+</td>
    </tr>
  </tbody>
</table>
</div>



First glance at our data, we can see that numerical variables differ significantly in scale.


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 689 entries, 0 to 688
    Data columns (total 16 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   A1      677 non-null    object 
     1   A2      677 non-null    float64
     2   A3      689 non-null    float64
     3   A4      683 non-null    object 
     4   A5      683 non-null    object 
     5   A6      680 non-null    object 
     6   A7      680 non-null    object 
     7   A8      689 non-null    float64
     8   A9      689 non-null    object 
     9   A10     689 non-null    object 
     10  A11     689 non-null    int64  
     11  A12     689 non-null    object 
     12  A13     689 non-null    object 
     13  A14     676 non-null    float64
     14  A15     689 non-null    int64  
     15  A16     689 non-null    object 
    dtypes: float64(4), int64(2), object(10)
    memory usage: 86.2+ KB
    

We can check which columns presented the most amount of missing values.


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A2</th>
      <th>A3</th>
      <th>A8</th>
      <th>A11</th>
      <th>A14</th>
      <th>A15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>677.000000</td>
      <td>689.000000</td>
      <td>689.000000</td>
      <td>689.000000</td>
      <td>676.000000</td>
      <td>689.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>31.569261</td>
      <td>4.765631</td>
      <td>2.224819</td>
      <td>2.402032</td>
      <td>183.988166</td>
      <td>1018.862119</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11.966670</td>
      <td>4.978470</td>
      <td>3.348739</td>
      <td>4.866180</td>
      <td>173.934087</td>
      <td>5213.743149</td>
    </tr>
    <tr>
      <th>min</th>
      <td>13.750000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>22.580000</td>
      <td>1.000000</td>
      <td>0.165000</td>
      <td>0.000000</td>
      <td>74.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>28.420000</td>
      <td>2.750000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>160.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>38.250000</td>
      <td>7.250000</td>
      <td>2.625000</td>
      <td>3.000000</td>
      <td>277.000000</td>
      <td>396.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80.250000</td>
      <td>28.000000</td>
      <td>28.500000</td>
      <td>67.000000</td>
      <td>2000.000000</td>
      <td>100000.000000</td>
    </tr>
  </tbody>
</table>
</div>



Using describe we can confirm that these numerical variables have notable differences in scale, that is something we'll have to keep in mind.

## Data Visualization


```python
# Definition of a function to visualize correlation between variables
import seaborn as sn
import matplotlib.pyplot as plt


def plot_correlation(df):
    corr_matrix = df.corr()
    heat_map = sn.heatmap(corr_matrix, annot=False)
    plt.show(heat_map)
```


```python
plot_correlation(df)
```


![png](output_16_0.png)


Between this particular variables there seem to be no clear correlation that could indicate we should delete any of the variables.


```python
pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(10, 10));
```


![png](output_18_0.png)


Using this scatter matrix we can see some outlier values mostly on **A15**. The problem is that we cannot know if those are error or real values. If we were to see that the precision of our model is not as high as we expected, we could always come back and try removing them.

## Data Processing
Our first step now will be separating our data into train set and test set.


```python
from sklearn.model_selection import train_test_split

X = df.drop(axis=1, columns='A16')

# replacing target variable possible values with 1 and 0
y = df['A16'].replace(to_replace=["+", "-"], value=[1, 0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=123)
```


```python
# Categorical variable names
cat_var = X_train.select_dtypes(include=['object', 'bool']).columns
# Numerical variable names
num_var = X_train.select_dtypes(np.number).columns
```

### Data Cleaning
Now we will impute missing values


```python
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
```

For this project we used a KNN Imputer for numerical variables and a most frequent imputer for categorical variables


```python
# Creating both numerical and categorical imputer
t1 = ('num_imputer', KNNImputer(n_neighbors=5), num_var)
t2 = ('cat_imputer', SimpleImputer(strategy='most_frequent'),
      cat_var)

column_transformer_cleaning = ColumnTransformer(
    transformers=[t1, t2], remainder='passthrough')

column_transformer_cleaning.fit(X_train)

Train_transformed = column_transformer_cleaning.transform(X_train)
Test_transformed = column_transformer_cleaning.transform(X_test)

# Here we update the order in wich variables are located in the dataframe, given that after transforming, we will have all
# numerical variables first, followed by all the categorical variables.
var_order = num_var.tolist() + cat_var.tolist()

# And finally we recreate the Data Frames
X_train_clean = pd.DataFrame(Train_transformed, columns=var_order)
X_test_clean = pd.DataFrame(Test_transformed, columns=var_order)
```

### Normalizing and encoding data
Next step is to normalize numerical data and encode categorical variables (One Hot Encoding or creating "dummy" variables)


```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
```


```python
# We obtain the diferent values in all categorical variables
dif_values = [df[column].dropna().unique() for column in cat_var]
```


```python
# Now we create the transformers
t_norm = ("normalizer", MinMaxScaler(feature_range=(0, 1)), num_var)
t_nominal = ("onehot", OneHotEncoder(
    sparse=False, categories=dif_values), cat_var)             
# As the dataset isn't huge, we will set sparse=false
```


```python
column_transformer_norm_enc = ColumnTransformer(transformers=[t_norm, t_nominal],
                                                remainder='passthrough')

column_transformer_norm_enc.fit(X_train_clean);
```


```python
X_train_transformed = column_transformer_norm_enc.transform(X_train_clean)
X_test_transformed = column_transformer_norm_enc.transform(X_test_clean)
```

And with this transformations, we end our data preprocessing

## Model Selection
We all know learning from any test set is a huge mistake that will compromise the precision of our estimations of performance. That is why we will separate even further our train set into validation train and validation test sets.


```python
X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(
    X_train_transformed, y_train, test_size=0.20, random_state=123)
```

As our performance metrics, we will use accuracy


```python
from sklearn.metrics import accuracy_score

from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
```

Now it begins the process of training different models to see which one performs the best and with what hyperparameters. For this project, we selected **K-Nearest Neighbors**, **Decision Tree Classifier** and **Logistic Regression**.


```python
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report
```

### Decision Tree Classifier


```python
from sklearn import tree

acc_scorer = make_scorer(accuracy_score, greater_is_better=True)
pipe_tree = make_pipeline(tree.DecisionTreeClassifier(random_state=1))

depths = np.arange(1, 31)
num_leafs = [1, 5, 10, 20, 50, 100]

param_grid_tree = [{'decisiontreeclassifier__max_depth': depths,
                    'decisiontreeclassifier__min_samples_leaf': num_leafs}]
```


```python
from sklearn.model_selection import GridSearchCV

gs_tree = GridSearchCV(estimator=pipe_tree,
                       param_grid=param_grid_tree, scoring='accuracy', cv=10)
best_tree = gs_tree.fit(X_train_transformed, y_train)
```


```python
print(classification_report(
    best_tree.best_estimator_.predict(X_train_transformed), y_train))
print(best_tree.best_params_)
print(best_tree.best_score_)
```

                  precision    recall  f1-score   support
    
               0       0.85      0.93      0.89       289
               1       0.92      0.82      0.87       262
    
        accuracy                           0.88       551
       macro avg       0.88      0.88      0.88       551
    weighted avg       0.88      0.88      0.88       551
    
    {'decisiontreeclassifier__max_depth': 4, 'decisiontreeclassifier__min_samples_leaf': 20}
    0.8674350649350648
    

### K-Nearest Neighbors


```python
from sklearn import neighbors

acc_scorer = make_scorer(accuracy_score, greater_is_better=True)
pipe_knn = make_pipeline(neighbors.KNeighborsClassifier())

n_neighbors = [number for number in np.arange(1, 32) if number % 2 == 1]
weights = ['uniform', 'distance']
metrics = ['euclidean', 'manhattan']

param_grid_knn = [{'kneighborsclassifier__n_neighbors': n_neighbors, 'kneighborsclassifier__weights': weights,
                   'kneighborsclassifier__metric': metrics}]
```


```python
gs_knn = GridSearchCV(estimator=pipe_knn,
                      param_grid=param_grid_knn, scoring='accuracy', cv=10)
best_knn = gs_knn.fit(X_train_transformed, y_train)
```


```python
print(classification_report(
    best_knn.best_estimator_.predict(X_train_transformed), y_train))
print(best_knn.best_params_)
print(best_knn.best_score_)
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00       315
               1       1.00      1.00      1.00       236
    
        accuracy                           1.00       551
       macro avg       1.00      1.00      1.00       551
    weighted avg       1.00      1.00      1.00       551
    
    {'kneighborsclassifier__metric': 'manhattan', 'kneighborsclassifier__n_neighbors': 27, 'kneighborsclassifier__weights': 'distance'}
    0.883733766233766
    

### Logistic Regression


```python
from sklearn import linear_model

import warnings
warnings.filterwarnings('ignore')

pipe_lr = make_pipeline(linear_model.LogisticRegression())

penalty = ['l1', 'l2']
c = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

param_grid_lr = {'logisticregression__penalty': penalty,
                 'logisticregression__C': c}

gs_lr = GridSearchCV(
    pipe_lr, param_grid=param_grid_lr, scoring='accuracy')
gs_lr.fit(X_train_transformed, y_train);
```


```python
print(classification_report(gs_lr.best_estimator_.predict(X_train_transformed), y_train))
print(gs_lr.best_params_)
print(gs_lr.best_score_)
```

                  precision    recall  f1-score   support
    
               0       0.88      0.93      0.90       298
               1       0.91      0.85      0.88       253
    
        accuracy                           0.89       551
       macro avg       0.90      0.89      0.89       551
    weighted avg       0.89      0.89      0.89       551
    
    {'logisticregression__C': 1, 'logisticregression__penalty': 'l2'}
    0.8783456183456184
    

## Model Training
We identified (it was close, mostly because of our data preprocessing) that **K-Nearest Neighbors** was the winner. Now it's time to train that model with all of our train data (keeping the best model parameters) to obtain the *down to earth* performance of our model.


```python
model = neighbors.KNeighborsClassifier(metric='manhattan',n_neighbors=27,weights='distance')
model.fit(X_train_transformed,y_train)
y_pred = model.predict(X_test_transformed)
acc = accuracy_score(y_test,y_pred)
print('The accuracy of our model is :',round(acc,4))
```

    The accuracy of our model is : 0.8261
    

## Conclusion

The tool we developed to help determining which clients can access a bank loan has an estimated accuracy of 83%
