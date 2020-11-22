# Credit Qualification

This tool was designed to provide crucial information in determining which clients can access a bank loan and which clients should not. Real world data was provided for the making of this tool, so a data privacy disclaimer was needed.

Relevant Information:

    This file concerns credit card applications.  All attribute names
    and values have been changed to meaningless symbols to protect
    confidentiality of the data.

## Dataset description

First things first we will take a look at the diferent variables presented to us, as well as their data type

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



First glance at our data, we can see that numerical variables differ significatly in scale.


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
    

We can check wich columns presented the most amount of missing values.


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



Using describe we can confirm that these numerical variables have notable diferences in scale, that is something we'll have to keep in mind.

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
    sparse=False, categories=dif_values), cat_var)             # As the dataset isn't huge, we will set sparse=false
```


```python
column_transformer_norm_enc = ColumnTransformer(transformers=[t_norm, t_nominal],
                                                remainder='passthrough')

column_transformer_norm_enc.fit(X_train_clean)
```




    ColumnTransformer(remainder='passthrough',
                      transformers=[('normalizer', MinMaxScaler(),
                                     Index(['A2', 'A3', 'A8', 'A11', 'A14', 'A15'], dtype='object')),
                                    ('onehot',
                                     OneHotEncoder(categories=[array(['a', 'b'], dtype=object),
                                                               array(['u', 'y', 'l'], dtype=object),
                                                               array(['g', 'p', 'gg'], dtype=object),
                                                               array(['q', 'w', 'm', 'r', 'cc', 'k', 'c', 'd', 'x', 'i', 'e', 'aa', 'ff',
           'j'], dtype=object),
                                                               array(['h', 'v', 'bb', 'ff', 'j', 'z', 'o', 'dd', 'n'], dtype=object),
                                                               array(['t', 'f'], dtype=object),
                                                               array(['t', 'f'], dtype=object),
                                                               array(['f', 't'], dtype=object),
                                                               array(['g', 's', 'p'], dtype=object)],
                                                   sparse=False),
                                     Index(['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13'], dtype='object'))])




```python
X_train_transformed = column_transformer_norm_enc.transform(X_train_clean)
X_test_transformed = column_transformer_norm_enc.transform(X_test_clean)
```

And with this transformations, we end our data preprocessing

## Model Selection
We all know learning from any test set is a huge mistake that will compromise the precission of our estimations of performance. That is why we will separate even further our train set into validation train and validation test sets.


```python
X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(
    X_train_transformed, y_train, test_size=0.20, random_state=123)
```

As our performance metrics, we will use accuracy


```python
from sklearn.metrics import accuracy_score
```

Now it begings the process of training different models to see wich one performs the best and with what hyperparameters.
For this project, we selected K-Nearest Neighbors (weighted and not weighted), Decision Tree Classifier and Logistic Regression.

### Decision Tree Classifier


```python
from sklearn.tree import DecisionTreeClassifier

best_Tree_model = None
best_AC_T = 0

for i in range(1, 100):
    model_T = DecisionTreeClassifier(max_depth=i)
    model_T.fit(X_val_train, y_val_train)
    y_pred_T = model_T.predict(X_val_test)
    AC_Tree = accuracy_score(y_val_test, y_pred_T)

    if AC_Tree > best_AC_T:
        best_AC_T = AC_Tree
        best_Tree_model = model_T

# print('The best Decision Tree Classifier had a depth of: ',
#       best_Tree_model.max_depth)
# print('With and Accuracy of: ', round(best_AC_T, 3))
```

The best Decision Tree Classifier had a depth of:  6    
With and Accuracy of:  0.928

### Distance Weighted K-Nearest Neighbors


```python
from sklearn import neighbors

best_KNN_D = None
best_AC_KNN_D = 0

for i in range(1, 100):
    KNN_D_model = neighbors.KNeighborsClassifier(
        n_neighbors=i, weights='distance')
    KNN_D_model.fit(X_val_train, y_val_train)
    y_pred_KNN_D = KNN_D_model.predict(X_val_test)

    AC_KNN_D = accuracy_score(y_val_test, y_pred_KNN_D)

    if AC_KNN_D > best_AC_KNN_D:
        best_AC_KNN_D = AC_KNN_D
        best_KNN_D = KNN_D_model

# print('The best distance weighted KNN model had: ',
#       best_KNN_D.n_neighbors, ' neighbors')
# print('With an accuracy of: ', round(best_AC_KNN_D, 3))
```

The best distance weighted KNN model had:  40  neighbors  
With an accuracy of:  0.919

### Not Weighted K-Nearest Neighbors


```python
from sklearn import neighbors

best_KNN_U = None
best_AC_KNN_U = 0

for i in range(1, 100):
    KNN_U_model = neighbors.KNeighborsClassifier(
        n_neighbors=i, weights='uniform')
    KNN_U_model.fit(X_val_train, y_val_train)
    y_pred_KNN_U = KNN_U_model.predict(X_val_test)

    AC_KNN_U = accuracy_score(y_val_test, y_pred_KNN_U)

    if AC_KNN_U > best_AC_KNN_U:
        best_AC_KNN_U = AC_KNN_U
        best_KNN_U = KNN_U_model

# print('The best not weighted KNN model had: ',
#       best_KNN_D.n_neighbors, ' neighbors')
# print('With an accuracy of: ', round(best_AC_KNN_D, 3))
```

The best not weighted KNN model had: 40 neighbors  
With an accuracy of: 0.919

### Logistic Regression


```python
from sklearn import linear_model

LogR_model = linear_model.LogisticRegression(
    max_iter=20000, penalty='none', fit_intercept=True, random_state=123)
LogR_model.fit(X_val_train, y_val_train)
y_pred_LogR = LogR_model.predict(X_val_test)

AC_LogR = accuracy_score(y_val_test, y_pred_LogR)

print("Logistic Regression model had the following coeficients: \n", LogR_model.coef_)
print("Accuracy: ", round(AC_LogR, 5))
```

    Logistic Regression model had the following coeficients: 
     [[  0.54512457  -1.13751313   1.78327063   9.05423909  -5.9583447
       15.93291865  -0.08989456  -0.28535717   0.07024966  -0.4455014
        0.           0.07024966  -0.4455014    0.          -0.51642635
        0.33065104  -0.51281162   4.16481675   1.78493893  -0.63033784
       -0.07534405   0.17111417   1.93482775   0.35350374   2.31896792
       -0.43179881  -4.39978618  -4.86756717   1.65236382   1.37986192
       -0.48524295   3.89445997   5.78178813  -3.8028768  -12.15040702
       -1.31685497   4.67165616   1.86432309  -2.23957482   0.15237269
       -0.52762442  -0.12488661  -0.25036513  -1.05918775  -1.03367138
        1.71760741]]
    Accuracy:  0.9009
    

## Model Training
We identified (it was close, mostly because of our great data preprocessing) that **Decision Tree Classifier** was the winner. Now it's time to train that model with all of our train data (keeping the winning depth of 6) to obtain the *down to earth* performance of our model.


```python
model = DecisionTreeClassifier(max_depth=6)
model.fit(X_train_transformed, y_train)
y_pred = model.predict(X_test_transformed)
accuracy = accuracy_score(y_test, y_pred)

# print('Model Accuracy: ', round(accuracy, 3))
```

Model Accuracy:  0.804

## Conclusion

The tool we developed to help determining which clients can access a bank loan has an estimated accuracy of 80%
