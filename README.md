{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Credit Qualification"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This tool was designed to provide crucial information in determining which clients can access a bank loan and which clients should not. Real world data was provided for the making of this tool, so a data privacy disclaimer was needed."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Relevant Information:\n",
    "\n",
    "    This file concerns credit card applications.  All attribute names\n",
    "    and values have been changed to meaningless symbols to protect\n",
    "    confidentiality of the data."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Dataset description"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "First things first we will take a look at the diferent variables presented to us, as well as their data type"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Number of Attributes: 15 + class attribute\n",
    "\n",
    "Attribute Information:\n",
    "\n",
    "    A1:\tb, a.\n",
    "    A2:\tcontinuous.\n",
    "    A3:\tcontinuous.\n",
    "    A4:\tu, y, l, t.\n",
    "    A5:\tg, p, gg.\n",
    "    A6:\tc, d, cc, i, j, k, m, r, q, w, x, e, aa, ff.\n",
    "    A7:\tv, h, bb, j, n, z, dd, ff, o.\n",
    "    A8:\tcontinuous.\n",
    "    A9:\tt, f.\n",
    "    A10:\tt, f.\n",
    "    A11:\tcontinuous.\n",
    "    A12:\tt, f.\n",
    "    A13:\tg, p, s.\n",
    "    A14:\tcontinuous.\n",
    "    A15:\tcontinuous.\n",
    "    A16: +,-         (class attribute)\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Using this information we can identify our target variable (**A16**). We can also see that it is a discrete variable, therefore we are facing a **Classification Problem**. It only has two possible values, \"+\" or \"-\"."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Missing Attribute Values:\n",
    "    37 cases (5%) have one or more missing values.  The missing\n",
    "    values from particular attributes are:\n",
    "\n",
    "    A1:  12\n",
    "    A2:  12\n",
    "    A4:   6\n",
    "    A5:   6\n",
    "    A6:   9\n",
    "    A7:   9\n",
    "    A14: 13"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We have around 5% of missing values, we will have to deal with those shortly. Now we'll take a look at the data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>A1</th>\n",
       "      <th>A2</th>\n",
       "      <th>A3</th>\n",
       "      <th>A4</th>\n",
       "      <th>A5</th>\n",
       "      <th>A6</th>\n",
       "      <th>A7</th>\n",
       "      <th>A8</th>\n",
       "      <th>A9</th>\n",
       "      <th>A10</th>\n",
       "      <th>A11</th>\n",
       "      <th>A12</th>\n",
       "      <th>A13</th>\n",
       "      <th>A14</th>\n",
       "      <th>A15</th>\n",
       "      <th>A16</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>a</td>\n",
       "      <td>58.67</td>\n",
       "      <td>4.460</td>\n",
       "      <td>u</td>\n",
       "      <td>g</td>\n",
       "      <td>q</td>\n",
       "      <td>h</td>\n",
       "      <td>3.04</td>\n",
       "      <td>t</td>\n",
       "      <td>t</td>\n",
       "      <td>6</td>\n",
       "      <td>f</td>\n",
       "      <td>g</td>\n",
       "      <td>43.0</td>\n",
       "      <td>560</td>\n",
       "      <td>+</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>a</td>\n",
       "      <td>24.50</td>\n",
       "      <td>0.500</td>\n",
       "      <td>u</td>\n",
       "      <td>g</td>\n",
       "      <td>q</td>\n",
       "      <td>h</td>\n",
       "      <td>1.50</td>\n",
       "      <td>t</td>\n",
       "      <td>f</td>\n",
       "      <td>0</td>\n",
       "      <td>f</td>\n",
       "      <td>g</td>\n",
       "      <td>280.0</td>\n",
       "      <td>824</td>\n",
       "      <td>+</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>b</td>\n",
       "      <td>27.83</td>\n",
       "      <td>1.540</td>\n",
       "      <td>u</td>\n",
       "      <td>g</td>\n",
       "      <td>w</td>\n",
       "      <td>v</td>\n",
       "      <td>3.75</td>\n",
       "      <td>t</td>\n",
       "      <td>t</td>\n",
       "      <td>5</td>\n",
       "      <td>t</td>\n",
       "      <td>g</td>\n",
       "      <td>100.0</td>\n",
       "      <td>3</td>\n",
       "      <td>+</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>b</td>\n",
       "      <td>20.17</td>\n",
       "      <td>5.625</td>\n",
       "      <td>u</td>\n",
       "      <td>g</td>\n",
       "      <td>w</td>\n",
       "      <td>v</td>\n",
       "      <td>1.71</td>\n",
       "      <td>t</td>\n",
       "      <td>f</td>\n",
       "      <td>0</td>\n",
       "      <td>f</td>\n",
       "      <td>s</td>\n",
       "      <td>120.0</td>\n",
       "      <td>0</td>\n",
       "      <td>+</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>b</td>\n",
       "      <td>32.08</td>\n",
       "      <td>4.000</td>\n",
       "      <td>u</td>\n",
       "      <td>g</td>\n",
       "      <td>m</td>\n",
       "      <td>v</td>\n",
       "      <td>2.50</td>\n",
       "      <td>t</td>\n",
       "      <td>f</td>\n",
       "      <td>0</td>\n",
       "      <td>t</td>\n",
       "      <td>g</td>\n",
       "      <td>360.0</td>\n",
       "      <td>0</td>\n",
       "      <td>+</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  A1     A2     A3 A4 A5 A6 A7    A8 A9 A10  A11 A12 A13    A14  A15 A16\n",
       "0  a  58.67  4.460  u  g  q  h  3.04  t   t    6   f   g   43.0  560   +\n",
       "1  a  24.50  0.500  u  g  q  h  1.50  t   f    0   f   g  280.0  824   +\n",
       "2  b  27.83  1.540  u  g  w  v  3.75  t   t    5   t   g  100.0    3   +\n",
       "3  b  20.17  5.625  u  g  w  v  1.71  t   f    0   f   s  120.0    0   +\n",
       "4  b  32.08  4.000  u  g  m  v  2.50  t   f    0   t   g  360.0    0   +"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(\"data/crx3.data\")\n",
    "columns = [\"A1\", \"A2\", \"A3\", \"A4\", \"A5\", \"A6\", \"A7\", \"A8\",\n",
    "           \"A9\", \"A10\", \"A11\", \"A12\", \"A13\", \"A14\", \"A15\", \"A16\"]\n",
    "df.columns = columns\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "First glance at our data, we can see that numerical variables differ significatly in scale."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 689 entries, 0 to 688\n",
      "Data columns (total 16 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   A1      677 non-null    object \n",
      " 1   A2      677 non-null    float64\n",
      " 2   A3      689 non-null    float64\n",
      " 3   A4      683 non-null    object \n",
      " 4   A5      683 non-null    object \n",
      " 5   A6      680 non-null    object \n",
      " 6   A7      680 non-null    object \n",
      " 7   A8      689 non-null    float64\n",
      " 8   A9      689 non-null    object \n",
      " 9   A10     689 non-null    object \n",
      " 10  A11     689 non-null    int64  \n",
      " 11  A12     689 non-null    object \n",
      " 12  A13     689 non-null    object \n",
      " 13  A14     676 non-null    float64\n",
      " 14  A15     689 non-null    int64  \n",
      " 15  A16     689 non-null    object \n",
      "dtypes: float64(4), int64(2), object(10)\n",
      "memory usage: 86.2+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can check wich columns presented the most amount of missing values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>A2</th>\n",
       "      <th>A3</th>\n",
       "      <th>A8</th>\n",
       "      <th>A11</th>\n",
       "      <th>A14</th>\n",
       "      <th>A15</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>677.000000</td>\n",
       "      <td>689.000000</td>\n",
       "      <td>689.000000</td>\n",
       "      <td>689.000000</td>\n",
       "      <td>676.000000</td>\n",
       "      <td>689.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>31.569261</td>\n",
       "      <td>4.765631</td>\n",
       "      <td>2.224819</td>\n",
       "      <td>2.402032</td>\n",
       "      <td>183.988166</td>\n",
       "      <td>1018.862119</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>11.966670</td>\n",
       "      <td>4.978470</td>\n",
       "      <td>3.348739</td>\n",
       "      <td>4.866180</td>\n",
       "      <td>173.934087</td>\n",
       "      <td>5213.743149</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>13.750000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>22.580000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.165000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>74.500000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>28.420000</td>\n",
       "      <td>2.750000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>160.000000</td>\n",
       "      <td>5.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>38.250000</td>\n",
       "      <td>7.250000</td>\n",
       "      <td>2.625000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>277.000000</td>\n",
       "      <td>396.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>80.250000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>28.500000</td>\n",
       "      <td>67.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>100000.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               A2          A3          A8         A11          A14  \\\n",
       "count  677.000000  689.000000  689.000000  689.000000   676.000000   \n",
       "mean    31.569261    4.765631    2.224819    2.402032   183.988166   \n",
       "std     11.966670    4.978470    3.348739    4.866180   173.934087   \n",
       "min     13.750000    0.000000    0.000000    0.000000     0.000000   \n",
       "25%     22.580000    1.000000    0.165000    0.000000    74.500000   \n",
       "50%     28.420000    2.750000    1.000000    0.000000   160.000000   \n",
       "75%     38.250000    7.250000    2.625000    3.000000   277.000000   \n",
       "max     80.250000   28.000000   28.500000   67.000000  2000.000000   \n",
       "\n",
       "                 A15  \n",
       "count     689.000000  \n",
       "mean     1018.862119  \n",
       "std      5213.743149  \n",
       "min         0.000000  \n",
       "25%         0.000000  \n",
       "50%         5.000000  \n",
       "75%       396.000000  \n",
       "max    100000.000000  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Using describe we can confirm that these numerical variables have notable diferences in scale, that is something we'll have to keep in mind."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Processing\n",
    "Our first step now will be separating our data into train set and test set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X = df.drop(axis=1, columns='A16')\n",
    "# replacing target variable possible values with 1 and 0\n",
    "y = df['A16'].replace(to_replace=[\"+\", \"-\"], value=[1, 0])\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.20, random_state=123)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Categorical variable names\n",
    "cat_var = X_train.select_dtypes(include=['object', 'bool']).columns\n",
    "# Numerical variable names\n",
    "num_var = X_train.select_dtypes(np.number).columns"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data Cleaning\n",
    "Now we will impute missing values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.impute import SimpleImputer  \n",
    "from sklearn.impute import KNNImputer\n",
    "from sklearn.compose import ColumnTransformer"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For this project we used a KNN Imputer for numerical variables and a most frequent imputer for categorical variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Creating both numerical and categorical imputer\n",
    "t1 = ('num_imputer', KNNImputer(n_neighbors=5), num_var)\n",
    "t2 = ('cat_imputer', SimpleImputer(strategy='most_frequent'),\n",
    "      cat_var)\n",
    "\n",
    "column_transformer_cleaning = ColumnTransformer(\n",
    "    transformers=[t1, t2], remainder='passthrough')\n",
    "\n",
    "column_transformer_cleaning.fit(X_train)\n",
    "\n",
    "Train_transformed = column_transformer_cleaning.transform(X_train)\n",
    "Test_transformed = column_transformer_cleaning.transform(X_test)\n",
    "\n",
    "# Here we update the order in wich variables are located in the dataframe, given that after transforming, we will have all\n",
    "# numerical variables first, followed by all the categorical variables.\n",
    "var_order = num_var.tolist() + cat_var.tolist()\n",
    "\n",
    "# And finally we recreate the Data Frames\n",
    "X_train_clean = pd.DataFrame(Train_transformed, columns=var_order)\n",
    "X_test_clean = pd.DataFrame(Test_transformed, columns=var_order)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Normalizing and encoding data\n",
    "Next step is to normalize numerical data and encode categorical variables (One Hot Encoding or creating \"dummy\" variables)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from sklearn.preprocessing import OneHotEncoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# We obtain the diferent values in all categorical variables\n",
    "dif_values = [df[column].dropna().unique() for column in cat_var]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Now we create the transformers\n",
    "t_norm = (\"normalizer\", MinMaxScaler(feature_range=(0, 1)), num_var)\n",
    "t_nominal = (\"onehot\", OneHotEncoder(\n",
    "    sparse=False, categories=dif_values), cat_var)             # As the dataset isn't huge, we will set sparse=false"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ColumnTransformer(remainder='passthrough',\n",
       "                  transformers=[('normalizer', MinMaxScaler(),\n",
       "                                 Index(['A2', 'A3', 'A8', 'A11', 'A14', 'A15'], dtype='object')),\n",
       "                                ('onehot',\n",
       "                                 OneHotEncoder(categories=[array(['a', 'b'], dtype=object),\n",
       "                                                           array(['u', 'y', 'l'], dtype=object),\n",
       "                                                           array(['g', 'p', 'gg'], dtype=object),\n",
       "                                                           array(['q', 'w', 'm', 'r', 'cc', 'k', 'c', 'd', 'x', 'i', 'e', 'aa', 'ff',\n",
       "       'j'], dtype=object),\n",
       "                                                           array(['h', 'v', 'bb', 'ff', 'j', 'z', 'o', 'dd', 'n'], dtype=object),\n",
       "                                                           array(['t', 'f'], dtype=object),\n",
       "                                                           array(['t', 'f'], dtype=object),\n",
       "                                                           array(['f', 't'], dtype=object),\n",
       "                                                           array(['g', 's', 'p'], dtype=object)],\n",
       "                                               sparse=False),\n",
       "                                 Index(['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13'], dtype='object'))])"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "column_transformer_norm_enc = ColumnTransformer(transformers=[t_norm, t_nominal],\n",
    "                                                remainder='passthrough')\n",
    "\n",
    "column_transformer_norm_enc.fit(X_train_clean)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_transformed = column_transformer_norm_enc.transform(X_train_clean)\n",
    "X_test_transformed = column_transformer_norm_enc.transform(X_test_clean)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "And with this transformations, we end our data preprocessing"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Model Selection\n",
    "We all know learning from any test set is a huge mistake that will compromise the precission of our estimations of performance. That is why we will separate even further our train set into validation train and validation test sets."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(\n",
    "    X_train_transformed, y_train, test_size=0.20, random_state=123)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As our performance metrics, we will use accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now it begings the process of training different models to see wich one performs the best and with what hyperparameters.\n",
    "For this project, we selected K-Nearest Neighbors (weighted and not weighted), Decision Tree Classifier and Logistic Regression."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Decision Tree Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "best_Tree_model = None\n",
    "best_AC_T = 0\n",
    "\n",
    "for i in range(1, 100):\n",
    "    model_T = DecisionTreeClassifier(max_depth=i)\n",
    "    model_T.fit(X_val_train, y_val_train)\n",
    "    y_pred_T = model_T.predict(X_val_test)\n",
    "    AC_Tree = accuracy_score(y_val_test, y_pred_T)\n",
    "\n",
    "    if AC_Tree > best_AC_T:\n",
    "        best_AC_T = AC_Tree\n",
    "        best_Tree_model = model_T\n",
    "\n",
    "# print('The best Decision Tree Classifier had a depth of: ',\n",
    "#       best_Tree_model.max_depth)\n",
    "# print('With and Accuracy of: ', round(best_AC_T, 3))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The best Decision Tree Classifier had a depth of:  6    \n",
    "With and Accuracy of:  0.928"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Distance Weighted K-Nearest Neighbors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn import neighbors\n",
    "\n",
    "best_KNN_D = None\n",
    "best_AC_KNN_D = 0\n",
    "\n",
    "for i in range(1, 100):\n",
    "    KNN_D_model = neighbors.KNeighborsClassifier(\n",
    "        n_neighbors=i, weights='distance')\n",
    "    KNN_D_model.fit(X_val_train, y_val_train)\n",
    "    y_pred_KNN_D = KNN_D_model.predict(X_val_test)\n",
    "\n",
    "    AC_KNN_D = accuracy_score(y_val_test, y_pred_KNN_D)\n",
    "\n",
    "    if AC_KNN_D > best_AC_KNN_D:\n",
    "        best_AC_KNN_D = AC_KNN_D\n",
    "        best_KNN_D = KNN_D_model\n",
    "\n",
    "# print('The best distance weighted KNN model had: ',\n",
    "#       best_KNN_D.n_neighbors, ' neighbors')\n",
    "# print('With an accuracy of: ', round(best_AC_KNN_D, 3))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The best distance weighted KNN model had:  40  neighbors  \n",
    "With an accuracy of:  0.919"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Not Weighted K-Nearest Neighbors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn import neighbors\n",
    "\n",
    "best_KNN_U = None\n",
    "best_AC_KNN_U = 0\n",
    "\n",
    "for i in range(1, 100):\n",
    "    KNN_U_model = neighbors.KNeighborsClassifier(\n",
    "        n_neighbors=i, weights='uniform')\n",
    "    KNN_U_model.fit(X_val_train, y_val_train)\n",
    "    y_pred_KNN_U = KNN_U_model.predict(X_val_test)\n",
    "\n",
    "    AC_KNN_U = accuracy_score(y_val_test, y_pred_KNN_U)\n",
    "\n",
    "    if AC_KNN_U > best_AC_KNN_U:\n",
    "        best_AC_KNN_U = AC_KNN_U\n",
    "        best_KNN_U = KNN_U_model\n",
    "\n",
    "# print('The best not weighted KNN model had: ',\n",
    "#       best_KNN_D.n_neighbors, ' neighbors')\n",
    "# print('With an accuracy of: ', round(best_AC_KNN_D, 3))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The best not weighted KNN model had: 40 neighbors  \n",
    "With an accuracy of: 0.919"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic Regression model had the following coeficients: \n",
      " [[  0.54512457  -1.13751313   1.78327063   9.05423909  -5.9583447\n",
      "   15.93291865  -0.08989456  -0.28535717   0.07024966  -0.4455014\n",
      "    0.           0.07024966  -0.4455014    0.          -0.51642635\n",
      "    0.33065104  -0.51281162   4.16481675   1.78493893  -0.63033784\n",
      "   -0.07534405   0.17111417   1.93482775   0.35350374   2.31896792\n",
      "   -0.43179881  -4.39978618  -4.86756717   1.65236382   1.37986192\n",
      "   -0.48524295   3.89445997   5.78178813  -3.8028768  -12.15040702\n",
      "   -1.31685497   4.67165616   1.86432309  -2.23957482   0.15237269\n",
      "   -0.52762442  -0.12488661  -0.25036513  -1.05918775  -1.03367138\n",
      "    1.71760741]]\n",
      "Accuracy:  0.9009\n"
     ]
    }
   ],
   "source": [
    "from sklearn import linear_model\n",
    "\n",
    "LogR_model = linear_model.LogisticRegression(\n",
    "    max_iter=20000, penalty='none', fit_intercept=True, random_state=123)\n",
    "LogR_model.fit(X_val_train, y_val_train)\n",
    "y_pred_LogR = LogR_model.predict(X_val_test)\n",
    "\n",
    "AC_LogR = accuracy_score(y_val_test, y_pred_LogR)\n",
    "\n",
    "print(\"Logistic Regression model had the following coeficients: \\n\", LogR_model.coef_)\n",
    "print(\"Accuracy: \", round(AC_LogR, 5))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Model Training\n",
    "We identified (it was close, mostly because of our great data preprocessing) that **Decision Tree Classifier** was the winner. Now it's time to train that model with all of our train data (keeping the winning depth of 6) to obtain the *down to earth* performance of our model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = DecisionTreeClassifier(max_depth=6)\n",
    "model.fit(X_train_transformed, y_train)\n",
    "y_pred = model.predict(X_test_transformed)\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "\n",
    "# print('Model Accuracy: ', round(accuracy, 3))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Model Accuracy:  0.804"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Conclusion"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The tool we developed to help determining which clients can access a bank loan has an estimated accuracy of 80%"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
