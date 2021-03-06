<h1><center> Artificial Intelligence Computer Assignment 4</center></h1>
<h2><center> Sepehr Ghobadi / Student No: 810098009 </center></h2>


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from hazm import Normalizer, Stemmer, Lemmatizer, utils, word_tokenize
from nltk import RegexpTokenizer
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.metrics import make_scorer,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV
from copy import deepcopy
```

# Aim of Project

In this project, we apply some of the simplest algorithms to predict the price of cars based on dataset collected from <a href="divar.com">Divar</a>. we use the **K-Nearest Neighbors**, the **decision trees**, and the **linear regression** and finally, we apply some of the ensemble learning methods based on the individual predictors to handle the task.

The methodology and codes are all explained throughout this report and the effect of several related parameters and techniques are discussed.

# 0.Dataset

The dataset contains the information of almost 130,000 sale ads from Divar website in vehicles category including the following details:

1. Brand
2. Category
3. Ad's Submit Date
4. Title and description
5. Ad's images count
6. Mileage
7. Price
8. Model year

The first rows of the dataset are shown below:


```python
dataset = pd.read_csv("./vehicles.csv")
dataset.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>brand</th>
      <th>category</th>
      <th>created_at</th>
      <th>description</th>
      <th>image_count</th>
      <th>mileage</th>
      <th>price</th>
      <th>title</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>heavy</td>
      <td>Tuesday 07PM</td>
      <td>?????????? 43j$NUM???????? ???????????? ???? ???????? ???????????? ???? ???? ...</td>
      <td>4</td>
      <td>NaN</td>
      <td>-1</td>
      <td>???????? ???????? ?????????? 950</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>????????</td>
      <td>light</td>
      <td>Wednesday 04AM</td>
      <td>???????????? ???? ???? ???? ???????? ???? ?????????? ???????? \n???????????? ????...</td>
      <td>3</td>
      <td>180000.0</td>
      <td>-1</td>
      <td>???????????? ???? ???????????? ???????? ??????</td>
      <td>1366</td>
    </tr>
    <tr>
      <th>2</th>
      <td>?????? ??????::Peugeot 405</td>
      <td>light</td>
      <td>Wednesday 11AM</td>
      <td>?????? 2000?????? ???????? ?????????? ???????? ???????? ?????? ???? ???????? ??...</td>
      <td>0</td>
      <td>290000.0</td>
      <td>8500000</td>
      <td>?????? ?????? 81 ????????</td>
      <td>1381</td>
    </tr>
    <tr>
      <th>3</th>
      <td>??????????::Nissan</td>
      <td>light</td>
      <td>Wednesday 01PM</td>
      <td>????????.\n?????????? ?????????? ???????? ?????? ???? ???????????? ???????? ???? ...</td>
      <td>3</td>
      <td>175000.0</td>
      <td>19500000</td>
      <td>???????????? 2????</td>
      <td>1372</td>
    </tr>
    <tr>
      <th>4</th>
      <td>????????::Samand</td>
      <td>light</td>
      <td>Thursday 07AM</td>
      <td>???????? ???? ???????? ???????? ?????? ?????????? ?????????? ???? ?????????? ??????...</td>
      <td>4</td>
      <td>80000.0</td>
      <td>23900000</td>
      <td>???????? ???????????? ??????????????</td>
      <td>1391</td>
    </tr>
  </tbody>
</table>
</div>



# 1. Preprocess

Here, we do some general preprocessing operations that are necessary for all of the regressors inputs. More specific processing for each algorithm's input will be done in the next sections.

The date of submit is represented in string format. first we add two columns 'Day' and 'Hour' based on 'Created_at' column. in 'Year' column for numbers less that 1366 the term "<1366" is used in dataset so we change it to 1365 for better numerical interpretations. also we replace "-1" values of 'Price' column with "NaN" to reach a uniform representation for Null values


```python
dataset['day'] = dataset['created_at'].apply(lambda s: s.split(" ")[0]).replace({"Monday":1, "Tuesday":2, "Wednesday":3, "Thursday":4 ,"Friday":5, "Saturday":6, "Sunday":7})
dataset['hour'] = dataset['created_at'].apply(lambda s: s.split(" ")[1]).apply(lambda s:int(s[0:2])+12 if s[2:]=="PM" else int(s[0:2]))
dataset = dataset.drop(['created_at'], axis=1)
dataset['year'].replace({"<1366":1365}, inplace=True)
dataset['year'] = pd.to_numeric(dataset['year'])
dataset['price'].replace({-1:np.NaN}, inplace=True)
```

### 1.1 Categorical Features <a id="Q2"></a>

In ML models we are often required to convert the categorical i.e text features to its numeric representation. The two most common ways to do this is __Label Encoding__ or __OneHot Encoding__. a label encoder encode labels with a value between 0 and N-1 where N is the number of distinct labels. If a label repeats it assigns the same value to as assigned earlier. in one-hot encoding the target column is replaced by a one-hot vector which is a 1 ?? N matrix (vector) used to distinguish each label in values from every other label in the column. The vector consists of 0s in all cells with the exception of a single 1 in a cell used uniquely to identify the label.

Depending on the data, label encoding introduces a new problem. For example, in this project we will encode a set of car brands into numerical data. This is actually categorical data and there is no relation, of any kind, between the rows. The problem here is since there are different numbers in the same column, the model will misunderstand the data to be in some kind of order, 0 < 1 <2. The model may derive a correlation like as the brand number increases the population increases but this clearly may not be the scenario in some other data or the prediction set. This can show most negative impacts in algorithms like those working based on the distance between instances like KNN.

The main problem with one-hot encoding is the large increase in the consumed memory and so the one-hot encoding must not be used unless it is necessary. we will compare performance of some models based on different methods of encoding but in this project since dataset is not too large and number of different values in categorical columns (30 different brand and 2 class in category column) is relatively smaller than final number of features (as we see in next sections we extract at least 200 features from text-based features) we can predict that OneHot encoding gives better results .


```python
class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, *, column, method=None):
        super().__init__()
        self.column = column
        self.method = method
    def fit(self, X, y=None):
        return self
    def transform(self, original_dataset, y=None):
        dataset = original_dataset.copy()
        column = self.column
        method = self.method
        values = dataset[column].dropna().unique().tolist()
        if method == "OneHot":
            onehot_columns = pd.get_dummies(dataset[column])
            onehot_columns.columns = values
            dataset = pd.concat([dataset, onehot_columns], axis=1, sort=False)
            dataset = dataset.drop([column], axis=1)

        if method == "Label":
            dataset[column] = dataset[column].replace({values[i]:int(i) for i in range(len(values))})
        return dataset

```

### 1.2 Missing Values  <a id="Q4"></a>

The real-world data often has a lot of missing values. The cause of missing values can be data corruption or failure to record data. The handling of missing data is very important during the preprocessing of the dataset as many machine learning algorithms do not support missing values. There are different methods to handle missing values. One can simply __delete rows with missing values__. it is a easy approach and the model trained with the removal of all missing values creates a robust model. but with this method we lose lot of information and also model works poorly if the percentage of missing values is excessive in comparison to the complete dataset.

Another approach could be __imputing missing values__. with this strategy missing cells of columns in the dataset which are having numeric values can be replaced with the mean, median, or mode of remaining values in the column. This method can prevent the loss of data compared to the earlier method but this method has some problems too. it causes data leakage and also does not factor the covariance between features.

A more complex method that has probably better results than earlier ones is to predict missing values using some ML algorithm or neural networks. this method can take into account the covariance between missing value column and other columns and also lightens the effects of data leakage of statistical method like mean imputation. in this project we use KNN imputing method that show much better results than mean imputing.

since the removing rows containing null values doesnt make dataset too small in this project we first use this method and the we will compare results given from using other methods too.


```python
class MissingValuesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, *, method=None):
        super().__init__()
        self.method = method
    def fit(self, X, y=None):
        return self
    def transform(self, dataset, y=None):
        dataset = dataset.copy()
        method = self.method
        if method == "drop":
            dataset=dataset.dropna()
            dataset=dataset.reset_index()
            dataset=dataset.drop(["index"], axis=1)
        if method == "fill_mean":
            dataset[col+"_missed"] = np.where(pd.isnull(dataset[col]), 1, 0)
            dataset[col+"_not_missed"] = np.where(pd.isnull(dataset[col]), 0, 1)
            dataset[col] = dataset[col].replace(np.NaN, int(dataset[col].mean()))
        if method == "fill_knn":
            imputer = KNNImputer(n_neighbors=320, add_indicator=True)
            text_features = dataset[["title","description"]]
            numerical_features = dataset.drop(["title", "description"], axis=1)
            numerical_columns= numerical_features.columns.tolist()+["missed_mileage", "missed_price", "missed_year"]
            numerical_features = imputer.fit_transform(numerical_features)
            numerical_features = pd.DataFrame(numerical_features, columns=numerical_columns)
            dataset = pd.concat([numerical_features, text_features], axis=1, sort=False)
        return dataset
```

### 1.3 Feature Extraction From Text <a id="Q31"></a>

'Title' and 'Description' columns have textual data in them so they cant be used in regression models. we can drop that columns but they can contain valuable informations about price of car. so we should extract some numerical features from them.

__Vectorization__ is the general process of turning a collection of text documents into numerical feature vectors. This specific strategy (tokenization, counting and normalization) is called the Bag of Words or ???Bag of n-grams??? representation. Documents are described by word occurrences while completely ignoring the relative position information of the words in the document. there are two methods based on vectorization:

1. __Count Vectorization__: in count vectorization we convert a collection of text documents to a matrix of token counts: the occurrences of tokens in each document. This algoithm produces a sparse representation of the counts.


2. __TF-IDF Vectorization__: Another approach for estimating importance of each word is using the **TF-IDF** measure of each word. **TF-IDF** stands for **term frequency-inverse document frequency** and is a statistical measure used to evaluate how important a word is to a document in a collection. The importance increases proportionally to **the number of times a word appears in the document** but is offseted by the **frequency of the word in the collection**. The goal of using TF-IDF instead of the empirical probabilities with smoothing is to scale down the impact of words that occur very frequently in different classes and that are hence empirically less informative than features that occur in a small fraction of the training set.

Also for easier process of textual columns we should first normalize the documents. we will remove stop words and also lemmatize them in order to extract features better and faster.


```python
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, *, method="TfidfVectorizer", max_features=None, min_freq_threshold=0.005, max_freq_threshold=0.9, final_column_names=[]):
        super().__init__()
        self.method = method
        self.max_features = max_features
        self.min_freq_threshold = min_freq_threshold
        self.max_freq_threshold = max_freq_threshold
        self.final_column_names = final_column_names

    def normalize_text_data(self, dataset):
        normalizer = Normalizer()
        lemmatizer = Lemmatizer()
        tokenizer = RegexpTokenizer(r'\w+')
        stopwords = utils.stopwords_list()
        result = dataset.copy()
        for col in ["title", "description"]:
            normalized_data = [normalizer.normalize(row) for row in result[col]]
            result[col] = [ " ".join([lemmatizer.lemmatize(word) for word in tokenizer.tokenize(row) if word not in stopwords]) for row in normalized_data]
        return result;

    def fit(self, X, y=None):
        return self

    def transform(self, dataset, y=None):
        dataset = dataset.copy()
        dataset = self.normalize_text_data(dataset)
        corpus = [d["title"]+" "+d["description"] for _,d in dataset.iterrows()]
        token_pattern=r'\b\w+\b'
        if self.method == "CountVectorizer":
            vectorizer = CountVectorizer(ngram_range=(1,2), token_pattern=token_pattern, max_features=self.max_features, min_df=self.min_freq_threshold, max_df=self.max_freq_threshold)
        if self.method == "TfidfVectorizer":
            vectorizer = TfidfVectorizer(smooth_idf=True, ngram_range=(1,2), token_pattern=token_pattern, max_features=self.max_features, min_df=self.min_freq_threshold, max_df=self.max_freq_threshold)
        features = pd.DataFrame(vectorizer.fit_transform(corpus).toarray(), columns=vectorizer.get_feature_names())
        dataset = pd.concat([dataset, features], axis=1, sort=False)
        dataset = dataset.drop(["title","description"], axis=1)
        self.final_column_names = dataset.columns
        return dataset

```

#### 1.3.1 Effect of feature extraction

here we compare three models: one without using features exteacted from textual columns, one using CountVectorizee feature extraction and last one using TfIdfVectorizer: (the codes for models evaluation will be explained later)


```python
TEST_PROPORTION = 0.15
SEED = 29
no_feature_dataset = CategoricalTransformer(column="brand", method="OneHot").fit_transform(dataset.copy().drop(["title","description"], axis=1))
no_feature_dataset = MissingValuesTransformer(method="drop").fit_transform(no_feature_dataset)
dataset_split = split_data(no_feature_dataset, 0.15, SEED)
preprocess_pipeline = Pipeline( steps = [
    ('ctg_preprocess', CategoricalTransformer(column="brand", method="OneHot")),
    ('imputation', MissingValuesTransformer(method="drop")),
    ('feature_extract', FeatureExtractor(method="TfidfVectorizer", max_features=700, min_freq_threshold=0.002, max_freq_threshold=0.5, final_column_names=[])),
])
tfidf_dataset, _ = clean_data(preprocess_pipeline, dataset.copy(), get_scale=False)
preprocess_pipeline.set_params(feature_extract__method="CountVectorizer")
cv_dataset, _ = clean_data(preprocess_pipeline, dataset.copy(), get_scale=False)
```


```python
dt_errors = []
dataset_names = ["No Text Feature", "Count Vectorizer", "TF-IDF Vectorizer"]
datasets = [no_feature_dataset, tfidf_dataset, cv_dataset]
max_depths = range(2,50,2)
all_train_rmses = []
all_test_rmses = []
for index, dt in enumerate(datasets):
    train_rmses = []
    test_rmses = []
    for depth in max_depths:
        predictor = DecisionTreeRegressor(max_depth=depth)
        errors = model_evaluator(split_data(dt, TEST_PROPORTION, SEED), predictor=predictor, scale=1)
        train_rmses.append(errors[2])
        test_rmses.append(errors[3])
    all_train_rmses.append(train_rmses)
    all_test_rmses.append(test_rmses)
```


```python
double_parameter_plot("Max Depth",list(max_depths), "Dataset", dataset_names, "Test RMSE", all_test_rmses, print_best_score=False)
```



![png](./images/output_23_0.png)



obviously title and description have valuable information about price and we should extract features from them and as we see CountVectorizer has a slightly better performance than TfIdfVectorizer

### 1.4 Scaling

**Feature Scaling** is a technique to standardize the independent features present in the data in a fixed range. In feature scaling, we subtract the mean of all features from each of them and divide them by the standard deviation to obtain a feature set with zero mean and standard deviation of 1.

Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization. Generally speaking, if feature scaling is not done, then a machine learning algorithm tends to weigh greater values, higher and consider smaller values as the lower values, regardless of the unit of the values.

As an example, **K-Nearest neighbors** simply work based on computing the distance between two points by the Euclidean distance and if one of the features has a broad range of values, the distance will be governed by this particular feature. Therefore, the range of all features should be normalized so that each feature contributes proportionately to the final distance.

Also, it must be mentioned that some of the algorithms which mostly work based on comparing single features from different instances are not sensitive to scaling, and scaling has no effect on their performance. As an example, scaling does not enhance the performance of the **decision tree** but it doesnt have negative effect also. So we will scale the dataset for entire project.

Here we apply the feature scaling to all of the features using __sklearn.preprocessing.StandardScaler__ class. we also use Scaler's __var__ attribute to revert scalling on MSE scores.

# 2. Computing the Information Gains

## 2.1. Information Entropy for a Dataset

The entropy $??$ of a dataset $X$ with $C$ possible classes like ${x_1, x_2, ??? ,x_c}$  and class probability mass function $P(X)$ is defined as:

$$H(X) = E[-log_b(P(X))] =-\sum_{i=1}^c P(x_i)log_b P(x_i)$$
$$  $$

Also, the conditional entropy of X when it is split based on the values of a discrete attribute $a$ with possible values of ${a_1, a_2, ??? ,a_n}$ is defined as the weighted sum of the entropies of all the subsets of the dataset while the weights are the probabilities of happening a specific value for $a$:

$$ H (X|a) = ??? \sum_{i=1}^n P(a=a_i) H(\{X|a=a_i\}) $$

where $\{X|a=a_i\}$ is the set of all the instances form X with $a=a_i$.

In the above formulas, $b$ is the base of the logarithm. Common values of $b$ are 2, Euler's number e, and 10, and the corresponding units of entropy are the bits for $b = 2$, nats for $b = e$, and bans for $b = 10$.

Information Entropy can be thought of as how unpredictable a dataset is.

## 2.2. Information Gain <a id="Q1"></a>

In a dataset, **information gain**  of an attribute $a$ is the change in information entropy ?? of a dataset when is split based on the possible values for $a$:

$$IG (T, a) = H (T) ??? H(T|a)$$

where H ( T | a ) is the conditional entropy of T given the value of attribute a.

The higher Information gain, the more Entropy we removed, which is what we want. In the perfect case, after splitting a dataset subset contain only a specific class of data, which causes zero entropy.

Information gain can be interpreted as the amount of information gained about a random variable from observing another random variable and it is referred to as mutual information between the two random variables. The information gain is often used for feature selection, by evaluating the gain of each variable in the context of the target variable.

## 2.3. Computing the Information Gains for the Dataset

For computing the information gain for each of the features in the context of the target variable (price) we use the  ***mutual_info_regression***     function from the Python sklearn library as below:


```python
ig_dataset = dataset.copy()
ig_dataset = CategoricalTransformer(column="brand", method="Label").transform(ig_dataset)
ig_dataset = CategoricalTransformer(column="category", method="Label").transform(ig_dataset)
ig_dataset = MissingValuesTransformer(method="drop").transform(ig_dataset)

FEATURES_LIST = [col for col in ig_dataset.columns.tolist() if col not in ['price', 'title', 'description']]
TARGET = 'price'

features_array = ig_dataset[FEATURES_LIST].values
target_array = ig_dataset[TARGET].values

mutual_info = pd.DataFrame(mutual_info_regression(features_array, target_array, discrete_features=True),
                           index=FEATURES_LIST, columns=['Mutual Information Gain'])

mutual_info = mutual_info.sort_values(by=['Mutual Information Gain'], ascending=False)
```

The values of the information gains are shown bellow:


```python
mutual_info
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mutual Information Gain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>brand</th>
      <td>0.728899</td>
    </tr>
    <tr>
      <th>year</th>
      <td>0.555465</td>
    </tr>
    <tr>
      <th>mileage</th>
      <td>0.254629</td>
    </tr>
    <tr>
      <th>image_count</th>
      <td>0.007779</td>
    </tr>
    <tr>
      <th>category</th>
      <td>0.000154</td>
    </tr>
    <tr>
      <th>day</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>hour</th>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



Also, the figure for the information gains is as follow:m


```python
mutual_info.plot.bar()
```




    <AxesSubplot:>





![png](./images/output_38_1.png)



## 2.4. Interpretation of the Information Gains

As we expected features like 'Brand', 'Year' and 'Mileage' are very determinative in a car price. and we can see that day and hour of posting an ad doesnt affect price of a car too much so we can remove these features from dataset. also as shown in plot a feature like category is not too much predictive. it can be because of the fact that description or brand of a car is more illustrator of car's features and a single word category ('heavy' or 'light') is not a feature that cant be realized from other features. image count is not as weak as other features mentioned so we wont drop it.


```python
dataset = dataset.drop(['category','day','hour','image_count'], axis=1)
```

# 3. Models Evaluation

Here we define some functions that will be used during evaluation process of different models and also a basic pipeline for preprocessing dataset. We will modify different parts of preprocess pipeline for each algorithm later.


```python
def clean_data(pipeline, dataset, get_scale=True):
    clean_dataset = pipeline.fit_transform(dataset)
    clean_dataset = pd.DataFrame(clean_dataset, columns=pipeline.get_params()["feature_extract__final_column_names"])
    price_scale=1
    if get_scale:
        price_scale = pipeline.get_params()['scale'].var_[clean_dataset.columns.get_loc("price")]
    return (clean_dataset, price_scale)

def split_data(dataset, test_proportion, seed):
    features = dataset.drop(['price'], axis=1)
    target = dataset['price']
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=test_proportion, random_state=seed)
    return {"x_train":x_train, "x_test":x_test, "y_train":y_train, "y_test":y_test}

def model_evaluator(dataset, predictor, scale=1):
    predictor.fit(dataset["x_train"], dataset["y_train"])
    y_predicted_train = predictor.predict(dataset["x_train"])
    y_predicted_test = predictor.predict(dataset["x_test"])
    train_mse = mean_squared_error(dataset["y_train"], y_predicted_train)*scale
    train_rmse = np.sqrt(train_mse)
    test_mse = mean_squared_error(dataset["y_test"], y_predicted_test)*scale
    test_rmse = np.sqrt(test_mse)
    return (train_mse, test_mse, train_rmse, test_rmse)

def plot_errors(x_label, x_values, errors,  print_best_score=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))
    ax1.plot(x_values, [error[0] for error in errors], label='MSE of train data')
    ax1.plot(x_values, [error[1] for error in errors], label='MSE of test data')
    ax1.set( xlabel=x_label, ylabel='MSE')
    ax1.legend()

    ax2.plot(x_values, [error[2] for error in errors], label='RMSE of train data')
    ax2.plot(x_values, [error[3] for error in errors], label='RMSE of test data')
    ax2.set( xlabel=x_label, ylabel='RMSE')
    ax2.legend()

    if print_best_score:
        print("Best RMSE = {:.2f} Milion Toman".format(min([x[3] for x in errors])/10**6))

def double_parameter_eval(split_dataset, p1, p1_values_range, p2, p2_values_range, predictor, scale):
    all_train_rmses = []
    all_test_rmses = []

    for p2_value in p2_values_range:
        train_rmses = []
        test_rmses = []
        for p1_value in p1_values_range:
            predictor.set_params(**{p1:p1_value, p2:p2_value})
            errors = model_evaluator(split_dataset, predictor=predictor, scale=scale)
            train_rmses.append(errors[2])
            test_rmses.append(errors[3])
        all_train_rmses.append(train_rmses)
        all_test_rmses.append(test_rmses)
    return (all_train_rmses, all_test_rmses)

def double_parameter_plot(p1, p1_values, p2, p2_values, y_label, scores, print_best_score=True):
    plt.figure(figsize=(15,7))
    for idx in range(len(scores)):
        plt.plot(p1_values, scores[idx], label='{}={}'.format(p2, p2_values[idx]))
    plt.xlabel(p1)
    plt.ylabel(y_label)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    if print_best_score:
        print("Best {} = {:.2f} Milion Toman".format(y_label, min([min(x) for x in scores])/10**6))

preprocess_pipeline = Pipeline( steps = [
    ('ctg_preprocess', CategoricalTransformer(column="brand", method="OneHot")),
    ('imputation', MissingValuesTransformer(method="drop")),
    ('feature_extract', FeatureExtractor(method="TfidfVectorizer", max_features=700, min_freq_threshold=0.002, max_freq_threshold=0.5, final_column_names=[])),
    ('scale', preprocessing.StandardScaler()),
])
```

# 4. Decision Tree

A Decision Tree is a simple supervised algorithm where the data is continuously split based on the parameter having the highest information gain into tree branches until reaching a maximum depth.


```python
dt_clean_dataset, dt_price_scale = clean_data(preprocess_pipeline, dataset.copy())
TEST_PROPORTION = 0.15
SEED = 29
split_dt_dataset = split_data(dt_clean_dataset, 0.15, SEED)
dt_clean_dataset.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mileage</th>
      <th>price</th>
      <th>year</th>
      <th>????????</th>
      <th>?????? ??????::Peugeot 405</th>
      <th>??????????::Nissan</th>
      <th>????????::Samand</th>
      <th>??????????????????::MVM</th>
      <th>?????????? ???????????????????::Pride</th>
      <th>?????????? ?????????????::Pride</th>
      <th>...</th>
      <th>????????</th>
      <th>????</th>
      <th>????</th>
      <th>????</th>
      <th>????</th>
      <th>???? ??????</th>
      <th>????</th>
      <th>????</th>
      <th>????</th>
      <th>???? ??????????</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.909658</td>
      <td>-0.562708</td>
      <td>-1.199293</td>
      <td>-0.162830</td>
      <td>-0.065691</td>
      <td>-0.058416</td>
      <td>-0.129499</td>
      <td>-0.088075</td>
      <td>-0.167143</td>
      <td>-0.073385</td>
      <td>...</td>
      <td>-0.081941</td>
      <td>-0.202290</td>
      <td>-0.137523</td>
      <td>-0.133741</td>
      <td>-0.182891</td>
      <td>-0.076001</td>
      <td>-0.151217</td>
      <td>-0.152054</td>
      <td>-0.247089</td>
      <td>-0.073288</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.735591</td>
      <td>-0.145700</td>
      <td>-2.821352</td>
      <td>-0.162830</td>
      <td>-0.065691</td>
      <td>-0.058416</td>
      <td>-0.129499</td>
      <td>-0.088075</td>
      <td>-0.167143</td>
      <td>-0.073385</td>
      <td>...</td>
      <td>-0.081941</td>
      <td>-0.202290</td>
      <td>-0.137523</td>
      <td>-0.133741</td>
      <td>-0.182891</td>
      <td>-0.076001</td>
      <td>-0.151217</td>
      <td>-0.152054</td>
      <td>-0.247089</td>
      <td>-0.073288</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.234290</td>
      <td>0.021104</td>
      <td>0.602994</td>
      <td>-0.162830</td>
      <td>-0.065691</td>
      <td>-0.058416</td>
      <td>-0.129499</td>
      <td>-0.088075</td>
      <td>-0.167143</td>
      <td>-0.073385</td>
      <td>...</td>
      <td>-0.081941</td>
      <td>3.173145</td>
      <td>-0.137523</td>
      <td>-0.133741</td>
      <td>-0.182891</td>
      <td>-0.076001</td>
      <td>-0.151217</td>
      <td>-0.152054</td>
      <td>-0.247089</td>
      <td>-0.073288</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.225128</td>
      <td>-0.562708</td>
      <td>-0.658607</td>
      <td>-0.162830</td>
      <td>-0.065691</td>
      <td>-0.058416</td>
      <td>-0.129499</td>
      <td>-0.088075</td>
      <td>-0.167143</td>
      <td>-0.073385</td>
      <td>...</td>
      <td>-0.081941</td>
      <td>-0.202290</td>
      <td>-0.137523</td>
      <td>-0.133741</td>
      <td>-0.182891</td>
      <td>-0.076001</td>
      <td>-0.151217</td>
      <td>-0.152054</td>
      <td>-0.247089</td>
      <td>-0.073288</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.938729</td>
      <td>-0.183609</td>
      <td>0.963452</td>
      <td>6.141378</td>
      <td>-0.065691</td>
      <td>-0.058416</td>
      <td>-0.129499</td>
      <td>-0.088075</td>
      <td>-0.167143</td>
      <td>-0.073385</td>
      <td>...</td>
      <td>-0.081941</td>
      <td>-0.202290</td>
      <td>-0.137523</td>
      <td>-0.133741</td>
      <td>2.817337</td>
      <td>-0.076001</td>
      <td>-0.151217</td>
      <td>-0.152054</td>
      <td>-0.247089</td>
      <td>-0.073288</td>
    </tr>
  </tbody>
</table>
<p>5 rows ?? 729 columns</p>
</div>




```python
dt_errors = []
max_depths = range(2,51,2)
for depth in max_depths:
    predictor = DecisionTreeRegressor(max_depth=depth)
    dt_errors.append( model_evaluator(split_dt_dataset, predictor=predictor, scale=dt_price_scale) )
```

<a id="Q6"></a>


```python
plot_errors("Max Depth", max_depths, dt_errors, print_best_score=True)
```

    Best RMSE = 11.85 Milion Toman




![png](./images/output_50_1.png)



### 4.1 Overfitting

As we see in above plot, after a certain point, increasing depth of decision tree only result in better training accuracy but results of test set get worse sllightly and gap between trian and test results increases. The common problem with Decision trees, especially having a table full of columns, they fit a lot. Sometimes it looks like the tree memorized the training data set. If there is no limit set on a decision tree, it will give you 100% accuracy on the training data set because in the worse case it will end up making 1 leaf for each observation. Thus this affects the accuracy when predicting samples that are not part of the training set.

there are two main methods to overcome overfit in decision trees:

1. __Pruning__: The splitting process results in fully grown trees until the stopping criteria are reached. But, the fully grown tree is likely to overfit the data, leading to poor accuracy on unseen data. there are different types of pruning. in one method, you trim off the branches of the tree, i.e., remove the decision nodes starting from the leaf node such that the overall accuracy is not disturbed. This is done by segregating the actual training set into two sets: training data set and validation data set. Prepare the decision tree using the segregated training data set. Then continue trimming the tree accordingly to optimize the accuracy of the validation data set. an easier method can be pruning based of number of samples belonging to a node in process of making tree. if we set a min_threshold for number of samples in a node when a node gets much precise about feature values we can stop from deeping to prevent overfitting
2. __Random Forests__ (see Section 7)

In this section we try to find a proper value for minimum threshold of samples (__min_samples_split__ in sklearn DecisionTreeRegressor model) in a node to prune decision tree in order to prevent model from overfitting:


```python
min_samples_split_range = list(range(60,201,20))+[240,300]
max_depth_range = range(2,31,2)

dt_prune_train_rmses ,dt_prune_test_rmses = double_parameter_eval(split_dt_dataset, 'max_depth', max_depth_range,
                                                                    'min_samples_split', min_samples_split_range,
                                                                    DecisionTreeRegressor(), dt_price_scale)
```


```python
double_parameter_plot('Max Depth', list(max_depth_range), "min_samples_split", list(min_samples_split_range), "Train RMSE", dt_prune_train_rmses, print_best_score=False)
double_parameter_plot('Max Depth', list(max_depth_range), "min_samples_split", list(min_samples_split_range), "Test RMSE", dt_prune_test_rmses)
```



![png](./images/output_54_0.png)





![png](./images/output_54_1.png)



    Best Test RMSE = 11.12 Milion Toman



```python
plt.figure(figsize=(15,7))
train_test_rmse_difference = []
for idx, min_samples_split in enumerate(min_samples_split_range):
    train_test_rmse_difference.append( [ dt_prune_test_rmses[idx][i]-dt_prune_train_rmses[idx][i] for i in range(len(dt_prune_test_rmses[idx])) ] )
double_parameter_plot('Max Depth', list(max_depth_range), "min_samples_split", list(min_samples_split_range), "Test RMSE - Train RMSE", train_test_rmse_difference, print_best_score=False)
```


    <Figure size 1080x504 with 0 Axes>




![png](./images/output_55_1.png)



As we see in "test RMSE - train RMSE" plot by increasing the min_sample_split threshold the difference between test and train gets smaller and also by increasing max depth the differnce inccreases with smaller rate. so we can see that pruning has positive effect on overfitting. but on the other side by looking at RMSE scores we can conclude that increasing min_samples_split from a certain point (200 in this case) increases RMSE again as learning has stopped in higher levels of tree.

# 5. Linear Regression

LinearRegression fits a linear model with coefficients w = (w1, ???, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.


```python
preprocess_pipeline.set_params(feature_extract__max_features=150) # 1111
lr_clean_dataset, lr_price_scale = clean_data(preprocess_pipeline, dataset.copy())
lr_clean_dataset.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mileage</th>
      <th>price</th>
      <th>year</th>
      <th>????????</th>
      <th>?????? ??????::Peugeot 405</th>
      <th>??????????::Nissan</th>
      <th>????????::Samand</th>
      <th>??????????????????::MVM</th>
      <th>?????????? ???????????????????::Pride</th>
      <th>?????????? ?????????????::Pride</th>
      <th>...</th>
      <th>??</th>
      <th>??????</th>
      <th>??</th>
      <th>????</th>
      <th>????</th>
      <th>????</th>
      <th>????</th>
      <th>????</th>
      <th>????</th>
      <th>????</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.909658</td>
      <td>-0.562708</td>
      <td>-1.199293</td>
      <td>-0.162830</td>
      <td>-0.065691</td>
      <td>-0.058416</td>
      <td>-0.129499</td>
      <td>-0.088075</td>
      <td>-0.167143</td>
      <td>-0.073385</td>
      <td>...</td>
      <td>-0.173558</td>
      <td>-0.188319</td>
      <td>-0.178832</td>
      <td>-0.148278</td>
      <td>-0.150128</td>
      <td>-0.204833</td>
      <td>-0.184998</td>
      <td>-0.152886</td>
      <td>-0.153733</td>
      <td>-0.248908</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.735591</td>
      <td>-0.145700</td>
      <td>-2.821352</td>
      <td>-0.162830</td>
      <td>-0.065691</td>
      <td>-0.058416</td>
      <td>-0.129499</td>
      <td>-0.088075</td>
      <td>-0.167143</td>
      <td>-0.073385</td>
      <td>...</td>
      <td>-0.173558</td>
      <td>-0.188319</td>
      <td>-0.178832</td>
      <td>-0.148278</td>
      <td>-0.150128</td>
      <td>-0.204833</td>
      <td>-0.184998</td>
      <td>-0.152886</td>
      <td>-0.153733</td>
      <td>-0.248908</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.234290</td>
      <td>0.021104</td>
      <td>0.602994</td>
      <td>-0.162830</td>
      <td>-0.065691</td>
      <td>-0.058416</td>
      <td>-0.129499</td>
      <td>-0.088075</td>
      <td>-0.167143</td>
      <td>-0.073385</td>
      <td>...</td>
      <td>-0.173558</td>
      <td>-0.188319</td>
      <td>-0.178832</td>
      <td>-0.148278</td>
      <td>-0.150128</td>
      <td>3.307756</td>
      <td>-0.184998</td>
      <td>-0.152886</td>
      <td>-0.153733</td>
      <td>-0.248908</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.225128</td>
      <td>-0.562708</td>
      <td>-0.658607</td>
      <td>-0.162830</td>
      <td>-0.065691</td>
      <td>-0.058416</td>
      <td>-0.129499</td>
      <td>-0.088075</td>
      <td>-0.167143</td>
      <td>-0.073385</td>
      <td>...</td>
      <td>-0.173558</td>
      <td>-0.188319</td>
      <td>-0.178832</td>
      <td>-0.148278</td>
      <td>-0.150128</td>
      <td>-0.204833</td>
      <td>-0.184998</td>
      <td>-0.152886</td>
      <td>-0.153733</td>
      <td>-0.248908</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.938729</td>
      <td>-0.183609</td>
      <td>0.963452</td>
      <td>6.141378</td>
      <td>-0.065691</td>
      <td>-0.058416</td>
      <td>-0.129499</td>
      <td>-0.088075</td>
      <td>-0.167143</td>
      <td>-0.073385</td>
      <td>...</td>
      <td>-0.173558</td>
      <td>-0.188319</td>
      <td>-0.178832</td>
      <td>-0.148278</td>
      <td>-0.150128</td>
      <td>-0.204833</td>
      <td>2.636153</td>
      <td>-0.152886</td>
      <td>-0.153733</td>
      <td>-0.248908</td>
    </tr>
  </tbody>
</table>
<p>5 rows ?? 179 columns</p>
</div>




```python
test_set_sizes = [0.02, 0.11, 0.6]
SEED = 42
lr_errors = []
for test_proportion in test_set_sizes:
    predictor = LinearRegression(n_jobs=-1)
    lr_errors.append(model_evaluator(split_data(lr_clean_dataset, test_proportion, SEED), predictor=predictor, scale=lr_price_scale))
```


```python
plot_errors("Test Set Proportion", test_set_sizes, lr_errors, print_best_score=True)
```

    Best RMSE = 14.80 Milion Toman




![png](./images/output_61_1.png)



<a id="Q5"></a>
The plots show that when we use a large amount of datas as training set (98%) we have lowest training RMSE but the test set error is very high and the difference between train and test errors is higher than other ratios which shows high __overfit__. Also when training set contains smaller fraction of dataset (40%) __underfitting__ occurs that cause the rmse for both test and train set increasr and also difference became more than difference in best proportion(90/10). Best split ratio for train and test set depends on many factors but we can do an exhaustive search between values of 5% to 20% (based on size of dataset values can change) and find best value that has small RMSE scores and doesnt overfit.

### 5.1 Polynomial and Interaction Features

We can add some features to consider co-realtion between different features and generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, if an input sample is two dimensional and of the form $[a, b]$, the degree-2 polynomial features are $[1, a, b, a^2, ab, b^2]$. interaction features are a subset of polynomial features that are product of n_degree different input features so terms like $a^2$ and $b^2$ are not interaction features.


```python
class PlynomialFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
    def fit(self, X, y=None):
        return self
    def transform(self, original_dataset, y=None):
        dataset = original_dataset.copy()
        text_features = dataset[["title","description"]]
        numerical_features = dataset.drop(["title", "description"], axis=1)
        numerical_columns = numerical_features.columns
        transformer = preprocessing.PolynomialFeatures(include_bias=True, interaction_only=True)
        numerical_features = transformer.fit_transform(numerical_features)
        numerical_columns = transformer.get_feature_names(numerical_columns)
        numerical_features = pd.DataFrame(numerical_features, columns=numerical_columns)
        dataset = pd.concat([numerical_features, text_features], axis=1, sort=False)
        return dataset
```


```python
lr_poly_preprocess_pipeline = deepcopy(preprocess_pipeline)
lr_poly_preprocess_pipeline.set_params(ctg_preprocess__method="Label")
lr_poly_preprocess_pipeline.set_params(feature_extract__max_features=1)
lr_poly_preprocess_pipeline.steps.insert(2,['poly_features',PlynomialFeaturesTransformer()])
lr_clean_dataset, lr_price_scale = clean_data(lr_poly_preprocess_pipeline, dataset.copy())
lr_clean_dataset.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>brand</th>
      <th>mileage</th>
      <th>price</th>
      <th>year</th>
      <th>brand mileage</th>
      <th>brand price</th>
      <th>brand year</th>
      <th>mileage price</th>
      <th>mileage year</th>
      <th>price year</th>
      <th>????????</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.099053</td>
      <td>1.909658</td>
      <td>-0.562708</td>
      <td>-1.199293</td>
      <td>-0.423431</td>
      <td>-0.371074</td>
      <td>-1.099574</td>
      <td>0.220463</td>
      <td>1.904263</td>
      <td>-0.563655</td>
      <td>0.837213</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>-0.907425</td>
      <td>0.735591</td>
      <td>-0.145700</td>
      <td>-2.821352</td>
      <td>-0.355020</td>
      <td>-0.313717</td>
      <td>-0.911466</td>
      <td>0.594651</td>
      <td>0.719768</td>
      <td>-0.154879</td>
      <td>-1.194439</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>-0.715797</td>
      <td>-0.234290</td>
      <td>0.021104</td>
      <td>0.602994</td>
      <td>-0.480441</td>
      <td>-0.252223</td>
      <td>-0.714249</td>
      <td>0.002071</td>
      <td>-0.231580</td>
      <td>0.021827</td>
      <td>-1.194439</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>-1.099053</td>
      <td>0.225128</td>
      <td>-0.562708</td>
      <td>-0.658607</td>
      <td>-0.611562</td>
      <td>-0.371074</td>
      <td>-1.099160</td>
      <td>-0.333415</td>
      <td>0.224228</td>
      <td>-0.562961</td>
      <td>0.837213</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>-0.524169</td>
      <td>-0.938729</td>
      <td>-0.183609</td>
      <td>0.963452</td>
      <td>-0.703917</td>
      <td>-0.247898</td>
      <td>-0.521172</td>
      <td>-0.672652</td>
      <td>-0.940231</td>
      <td>-0.181654</td>
      <td>0.837213</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_set_sizes = [0.01*x for x in range(2,80,1)]
SEED = 62
lr_errors = []
for test_proportion in test_set_sizes:
    predictor = LinearRegression(n_jobs=-1)
    lr_errors.append(model_evaluator(split_data(lr_clean_dataset, test_proportion, SEED), predictor=predictor, scale=lr_price_scale))
```


```python
plot_errors("Test Set Proportion", test_set_sizes, lr_errors, print_best_score=True)
```

    Best RMSE = 0.06 Milion Toman




![png](./images/output_68_1.png)



# 6. K Nearest Neighbours (KNN)

The k-nearest neighbor algorithm (K-NN) is a non-parametric method used for classification and regression. In the K-NN regressor, a target is predicted by local interpolation of the targets associated of the nearest neighbors in the training. Regression based on k-nearest neighbors (k is a positive integer, typically small).


```python
preprocess_pipeline.set_params(feature_extract__max_features=None, feature_extract__min_freq_threshold=0.002, feature_extract__max_freq_threshold=0.8)
knn_clean_dataset, knn_price_scale = clean_data(preprocess_pipeline, dataset.copy())
knn_clean_dataset.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mileage</th>
      <th>price</th>
      <th>year</th>
      <th>????????</th>
      <th>?????? ??????::Peugeot 405</th>
      <th>??????????::Nissan</th>
      <th>????????::Samand</th>
      <th>??????????????????::MVM</th>
      <th>?????????? ???????????????????::Pride</th>
      <th>?????????? ?????????????::Pride</th>
      <th>...</th>
      <th>????</th>
      <th>???? ????????</th>
      <th>???? ??????????</th>
      <th>???? ??????</th>
      <th>???? ????????????</th>
      <th>???? ??</th>
      <th>???? ??</th>
      <th>???? ??</th>
      <th>???? ??</th>
      <th>???? ??</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.909658</td>
      <td>-0.562708</td>
      <td>-1.199293</td>
      <td>-0.162830</td>
      <td>-0.065691</td>
      <td>-0.058416</td>
      <td>-0.129499</td>
      <td>-0.088075</td>
      <td>-0.167143</td>
      <td>-0.073385</td>
      <td>...</td>
      <td>-0.248969</td>
      <td>-0.050734</td>
      <td>-0.073522</td>
      <td>-0.045647</td>
      <td>-0.049264</td>
      <td>-0.054709</td>
      <td>-0.056937</td>
      <td>-0.052866</td>
      <td>-0.051149</td>
      <td>-0.043048</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.735591</td>
      <td>-0.145700</td>
      <td>-2.821352</td>
      <td>-0.162830</td>
      <td>-0.065691</td>
      <td>-0.058416</td>
      <td>-0.129499</td>
      <td>-0.088075</td>
      <td>-0.167143</td>
      <td>-0.073385</td>
      <td>...</td>
      <td>-0.248969</td>
      <td>-0.050734</td>
      <td>-0.073522</td>
      <td>-0.045647</td>
      <td>-0.049264</td>
      <td>-0.054709</td>
      <td>-0.056937</td>
      <td>-0.052866</td>
      <td>-0.051149</td>
      <td>-0.043048</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.234290</td>
      <td>0.021104</td>
      <td>0.602994</td>
      <td>-0.162830</td>
      <td>-0.065691</td>
      <td>-0.058416</td>
      <td>-0.129499</td>
      <td>-0.088075</td>
      <td>-0.167143</td>
      <td>-0.073385</td>
      <td>...</td>
      <td>-0.248969</td>
      <td>-0.050734</td>
      <td>-0.073522</td>
      <td>-0.045647</td>
      <td>-0.049264</td>
      <td>-0.054709</td>
      <td>-0.056937</td>
      <td>-0.052866</td>
      <td>-0.051149</td>
      <td>-0.043048</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.225128</td>
      <td>-0.562708</td>
      <td>-0.658607</td>
      <td>-0.162830</td>
      <td>-0.065691</td>
      <td>-0.058416</td>
      <td>-0.129499</td>
      <td>-0.088075</td>
      <td>-0.167143</td>
      <td>-0.073385</td>
      <td>...</td>
      <td>-0.248969</td>
      <td>-0.050734</td>
      <td>-0.073522</td>
      <td>-0.045647</td>
      <td>-0.049264</td>
      <td>-0.054709</td>
      <td>-0.056937</td>
      <td>-0.052866</td>
      <td>-0.051149</td>
      <td>-0.043048</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.938729</td>
      <td>-0.183609</td>
      <td>0.963452</td>
      <td>6.141378</td>
      <td>-0.065691</td>
      <td>-0.058416</td>
      <td>-0.129499</td>
      <td>-0.088075</td>
      <td>-0.167143</td>
      <td>-0.073385</td>
      <td>...</td>
      <td>-0.248969</td>
      <td>-0.050734</td>
      <td>-0.073522</td>
      <td>-0.045647</td>
      <td>-0.049264</td>
      <td>-0.054709</td>
      <td>-0.056937</td>
      <td>-0.052866</td>
      <td>-0.051149</td>
      <td>-0.043048</td>
    </tr>
  </tbody>
</table>
<p>5 rows ?? 1675 columns</p>
</div>




```python
n_neighbours = list(range(1,40,2))
TEST_PROPORTION = 0.1
SEED=13
knn_errors = []
for n in n_neighbours:
    predictor = KNeighborsRegressor(n_neighbors=n, n_jobs=-1)
    knn_errors.append(model_evaluator(split_data(k_clean_dataset, test_proportion, SEED), predictor=predictor, scale=k_price_scale))
    print(n)
```


```python
plot_errors("Number of Neighbours" ,n_neighbours, knn_errors, print_best_score=True)
```

    Best RMSE = 16.13 Milion Toman




![png](./images/output_73_1.png)



and if we plot errors for all possible values for N (that are smaller than regressor's n_samples_fit_) we can see that RMSE is ascending (results are imported from execution on google colab):


```python
n_to_infinity_neighbours = [10+100*x for x in range(1,61,1)]
plot_errors("Number of Neighbours" ,n_to_infinity_neighbours, n_to_infinity_errors)
```



![png](./images/output_75_0.png)



# 7. Random Forest

**Random forest** or **random decision forest** is a method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random Forest is an example of ensemble learning, in which we combine multiple machine learning algorithms to obtain better predictive performance.

Two key concepts that give it the name random:

A random sampling of training data set when building trees.
Random subsets of features considered when splitting nodes.
A technique known as bagging is used to create an ensemble of trees where multiple training sets are generated with replacement.

In the bagging technique, a data set is divided into N samples using randomized sampling. Then, using a single learning algorithm a model is built on all samples. Later, the resultant predictions are combined using voting or averaging in parallel.

### 7.1 Hyperparameteres  <a id="low_variance"></a>

In random forests, each tree in the ensemble is built from a sample drawn with replacement (i.e., a bootstrap sample) from the training set. Furthermore, when splitting each node during the construction of a tree, the best split is found either from all input features or a random subset of size __max_features__.

The purpose of these two sources of randomness is to decrease the variance of the forest estimator. Indeed, individual decision trees typically exhibit high variance and tend to overfit. The injected randomness in forests yield decision trees with somewhat decoupled prediction errors. By taking an average of those predictions, some errors can cancel out. Random forests achieve a reduced variance by combining diverse trees, sometimes at the cost of a slight increase in bias. In practice the variance reduction is often significant hence yielding an overall better model.

### 7.2 Number Of Estimators


```python
n_estimators_range = range(1, 15, 2)
max_depth_range = range(2, 30, 3)

RF_est_train_rmses, RF_est_test_rmses = double_parameter_eval(split_dt_dataset, "max_depth", max_depth_range, "n_estimators", n_estimators_range, RandomForestRegressor(n_jobs=-1), dt_price_scale)
```


```python
double_parameter_plot('Max Depth', list(max_depth_range), "n_estimators", list(n_estimators_range),"Train RMSE", RF_est_train_rmses)
double_parameter_plot('Max Depth', list(max_depth_range), "n_estimators", list(n_estimators_range),"Test RMSE", RF_est_test_rmses)
```



![png](./images/output_82_0.png)



    Best Train RMSE = 5.54 Milion Toman




![png](./images/output_82_2.png)



    Best Test RMSE = 10.27 Milion Toman


<b>now we raise number of estimators to higher values (the excecution result from colab):</b>


```python
n_estimators_range = range(30, 180, 15)
max_depth_range = range(2, 30, 3)
RF_high_n_estimatores_train_rmses = pd.read_csv('./data/rf_train_rmse.csv').drop(["Unnamed: 0"], axis=1).values
RF_high_n_estimatores_test_rmses = pd.read_csv('./data/rf_test_rmse.csv').drop(["Unnamed: 0"], axis=1).values
double_parameter_plot('Max Depth', list(max_depth_range), "n_estimators", list(n_estimators_range),"Test RMSE", RF_high_n_estimatores_test_rmses)
```



![png](./images/output_84_0.png)



    Best Test RMSE = 10.12 Milion Toman


<b>as we expected there is no significant change in higher number of estimators </b>

### 7.2 Max Features


```python
max_features_range = range(50,701,50)
n_estimators_range = range(1, 20, 3)
RF_mx_train_rmses, RF_mx_test_rmses = double_parameter_eval(split_dt_dataset, "n_estimators", n_estimators_range, "max_features", max_features_range, RandomForestRegressor(max_depth=18, min_samples_split=2, n_jobs=-1), dt_price_scale)

```


```python
double_parameter_plot('n_estimators', list(n_estimators_range), "max_features", list(max_features_range),"Train RMSE", RF_mx_train_rmses, print_best_score=False)
double_parameter_plot('n_estimators', list(n_estimators_range), "max_features", list(max_features_range),"Test RMSE", RF_mx_test_rmses)
```



![png](./images/output_88_0.png)





![png](./images/output_88_1.png)



    Best Test RMSE = 10.28 Milion Toman


Like we expected increasing max_feature results in better performance. Increasing max_features generally improves the performance of the model as at each node now we have a higher number of options to be considered. However, this is not necessarily true as this decreases the diversity of individual tree which is the USP of random forest. But, for sure, you decrease the speed of algorithm by increasing the max_features. Hence, you need to strike the right balance and choose the optimal max_features.

### 7.3 Bootstraping

Bootstrapping is sampling with replacement from the original data, and taking the not chosen data points as test cases. The procedure is taken several times and the calculated average score is represented as an estimation of our model performance. Bootstrapping is usually used when there are not enough samples for training.

In ensemble learning, bootstrapping can be used to train different models using different bootstrap samples of the data, where each bootstrap sample is a random sample of the data drawn with replacement and treated as if it was independently drawn from the underlying distribution.

In the case of proper use, bootstrapping can marginally decrease both bias and variance but if it is applied more than a certain amount it can lead to overfitting and increase in variance while decreasing the bias.

### 7.4 Comparing the Ensemble Learning Models with Individual Regressors

An ensemble-based system is obtained by combining different models and their decision is based on the collective opinion of individual models to reach more certain opinions. However, it is important to emphasize that there is no guarantee of the combination of multiple models to always perform better than the best individual models in the ensemble. Indeed, an improvement in the ensemble's average performance can not be guaranteed except for certain special cases. Hence combining models may not necessarily beat the performance of the best model.

For the ensemble to outperform the individual models, regressors must exhibit some level of diversity among themselves. Within the prediction context, then, the diversity in the predictors allows individual models to generate different decision boundaries. If proper diversity is achieved, a different error is made by each model, a strategic combination of which can then reduce the total error. After all, if all models provided the same output, correcting a possible mistake would not be possible. Therefore, individual regressors in an ensemble system need to make different errors in different instances.

### 7.5 Random Forest vs. Decision Tree <a id="Q7"></a>

Decision trees are prone to overfitting, especially when a tree is particularly deep. This is due to the amount of specificity we look at leading to smaller sample of events that meet the previous assumptions. This small sample could lead to unsound conclusions.

As explained <a href="low_variance">here</a> Random Forests reduce variance by training on different samples of the data. A second way is by using a random subset of features. This means random forests will only use a certain number of those features in each model and we have omitted other features that could be useful. But as stated, a random forest is a collection of decision trees. Thus, in each tree we can utilize five random features. If we use many trees in our forest, eventually many or all of our features will have been included. This inclusion of many features will help limit our error due to bias and error due to variance. If features weren???t chosen randomly, base trees in our forest could become highly correlated. This is because a few features could be particularly predictive and thus, the same features would be chosen in many of the base trees. If many of these trees included the same features we would not be combating error due to variance.

but random forests can increase bias for somme reason:

1. More data produces better models, and since we only use part of the whole training data to train the model (bootstrap), higher bias occurs in each tree

2. In the Random Forests algorithm, we limit the number of variables to split on in each split - i.e. we limit the number of variables to explain our data with. Again, higher bias occurs in each tree.

so in general we expect that random forest decreses variance significantly and a small amount of increase in bias is acceptable. as we see in plots of decision trees in some cases the train RMSE reaches to $5*10^6$ while test RMSE is higher that $11*10^6$ but in random forest the difference is much smaller (on average 2 or 3 million) while fortunately the ensemble model has lower bias too. as we see random forest has reached test RMSE of $10.12*10^6$ milions while the difference with train set is around $2*10^6$:


```python
n_estimators_range = range(30, 180, 15)
max_depth_range = range(2, 30, 3)
RF_high_n_estimatores_train_rmses = pd.read_csv('./data/rf_train_rmse.csv').drop(["Unnamed: 0"], axis=1).values
RF_high_n_estimatores_test_rmses = pd.read_csv('./data/rf_test_rmse.csv').drop(["Unnamed: 0"], axis=1).values
double_parameter_plot('Max Depth', list(max_depth_range), "n_estimators", list(n_estimators_range),"Test RMSE", RF_high_n_estimatores_test_rmses)
```



![png](./images/output_96_0.png)



    Best Test RMSE = 10.12 Milion Toman



```python
plt.figure(figsize=(15,7))
train_test_rmse_difference = []
for idx, _ in enumerate(n_estimators_range):
    train_test_rmse_difference.append( [ RF_high_n_estimatores_test_rmses[idx][i]-RF_high_n_estimatores_train_rmses[idx][i] for i in range(len(RF_high_n_estimatores_test_rmses[idx])) ] )
double_parameter_plot('Max Depth', list(max_depth_range), "n_estimators", list(n_estimators_range), "Test RMSE - Train RMSE", train_test_rmse_difference, print_best_score=False)
```


    <Figure size 1080x504 with 0 Axes>




![png](./images/output_97_1.png)
