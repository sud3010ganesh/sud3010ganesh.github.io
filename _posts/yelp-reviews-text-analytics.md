---
title: "Text Analytics on Yelp Reviews"
permalink: /yelp-reviews-text-analytics/
---

Yelp Reviews Data - Text Analytics

### Sudharsan Ganesh T

## Introduction

This project uses a small subset of the data from Kaggle's [Yelp Business Rating Prediction](https://www.kaggle.com/c/yelp-recsys-2013) competition.

**Description of the data:**

- **`yelp.csv`** contains the dataset. It is stored in the course repository (in the **`data`** directory), so there is no need to download anything from the Kaggle website.
- Each observation (row) in this dataset is a review of a particular business by a particular user.
- The **stars** column is the number of stars (1 through 5) assigned by the reviewer to the business. (Higher stars is better.) In other words, it is the rating of the business by the person who wrote the review.
- The **text** column is the text of the review.

**Goal:** Predict the star rating of a review using **only** the review text.

**Tip:** After each task, I recommend that you check the shape and the contents of your objects, to confirm that they match your expectations.


```python
# for Python 2: use print only as a function
# from __future__ import print_function
```


Read **`yelp.csv`** into a Pandas DataFrame and examine it.


```python
import pandas as pd
```


```python
path = 'yelp.csv'
yelp = pd.read_csv(path)
###Check Shape of Dataset###
yelp.shape
```




    (10000, 10)




Create a new DataFrame that only contains the **5-star** and **1-star** reviews.

- **Hint:** [How do I apply multiple filter criteria to a pandas DataFrame?](https://www.youtube.com/watch?v=YPItfQ87qjM&list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y&index=9) explains how to do this.


```python
yelp_best_worst = yelp[(yelp.stars==5) | (yelp.stars==1)]
yelp_best_worst.shape
```




    (4086, 10)




```python
# examine the class distribution
yelp_best_worst.stars.value_counts().sort_index()
```




    1     749
    5    3337
    Name: stars, dtype: int64




```python
type(yelp_best_worst)
```




    pandas.core.frame.DataFrame




Define X and y from the new DataFrame, and then split X and y into training and testing sets, using the **review text** as the only feature and the **star rating** as the response.

- **Hint:** Keep in mind that X should be a Pandas Series (not a DataFrame), since we will pass it to CountVectorizer in the task that follows.


```python
# define X and y
X = yelp_best_worst.text
y = yelp_best_worst.stars
```


```python
# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```

    C:\Users\Sudarshan\Anaconda3\lib\site-packages\sklearn\cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
# examine the object shapes
print(X_train.shape)
print(X_test.shape)
```

    (3064,)
    (1022,)



Use CountVectorizer to create **document-term matrices** from X_train and X_test.


```python
# use CountVectorizer to create document-term matrices from X_train and X_test
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
```


```python
# fit and transform X_train
X_train_dtm = vect.fit_transform(X_train)
```


```python
# only transform X_test
X_test_dtm = vect.transform(X_test)
```


```python
# examine the shapes
print(X_train_dtm.shape)
print(X_test_dtm.shape)
```

    (3064, 16825)
    (1022, 16825)



Use Multinomial Naive Bayes to **predict the star rating** for the reviews in the testing set, and then **calculate the accuracy** and **print the confusion matrix**.

- **Hint:** [Evaluating a classification model](https://github.com/justmarkham/scikit-learn-videos/blob/master/09_classification_metrics.ipynb) explains how to interpret both classification accuracy and the confusion matrix.


```python
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
```


```python
# print the number of features that were generated
print('Features: ', X_train_dtm.shape[1])

# use Multinomial Naive Bayes to predict the star rating
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
y_pred_class = nb.predict(X_test_dtm)

# print the accuracy of its predictions
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_class))

# print the confusion matrix
print('Confusion Matrix: ')
print(metrics.confusion_matrix(y_test, y_pred_class))
```

    Features:  16825
    Accuracy:  0.918786692759
    Confusion Matrix:
    [[126  58]
     [ 25 813]]



Calculate the **null accuracy**, which is the classification accuracy that could be achieved by always predicting the most frequent class.

- **Hint:** [Evaluating a classification model](https://github.com/justmarkham/scikit-learn-videos/blob/master/09_classification_metrics.ipynb) explains null accuracy and demonstrates two ways to calculate it, though only one of those ways will work in this case. Alternatively, you can come up with your own method to calculate null accuracy!


```python
####Null Accuracy####
###NULL Accuracy : No of correct classifications when we predict all record to be star rating 5####
y_test.value_counts().head(1)/y_test.shape
```




    5    0.819961
    Name: stars, dtype: float64





Browse through the review text of some of the **false positives** and **false negatives**. Based on your knowledge of how Naive Bayes works, do you have any ideas about why the model is incorrectly classifying these reviews?

- **Hint:** [Evaluating a classification model](https://github.com/justmarkham/scikit-learn-videos/blob/master/09_classification_metrics.ipynb) explains the definitions of "false positives" and "false negatives".
- **Hint:** Think about what a false positive means in this context, and what a false negative means in this context. What has scikit-learn defined as the "positive class"?



In our classification problem,  Naive Bayes has taken 5 star rating as positive class and 1 star rating as a negative class.

**FALSE POSITIVE**
A false positive is a scenario where the actual label is 0 but the predicted label is 1. In this scenario if we predict an actual bad rating(1 star) as a good rating(5 star) we can call the observation to be a false positive

**FALSE NEGATIVE**
A false negative is a scenario where the actual label is 1 but the predicted label is 0. In this scenario if we predict an actual good rating(5 star) to be a bad rating(1 star) we can call the observation to be a false negative


```python
#####Filter out a sample of false positives#####
X_test[y_test < y_pred_class].sample(10, random_state=6)
```




    2175    This has to be the worst restaurant in terms o...
    1899    Buca Di Beppo is literally, italian restaurant...
    6222    My mother always told me, if I didn't have any...
    8833    The owner has changed hands & this place isn't...
    8000    Still a place that is unacceptable in my book-...
    943     Don't waste your time...Arrowhead mall on the ...
    7631    this is a business located in the fry's grocer...
    3755    Have been going to LGO since 2003 and have alw...
    9299    The salad plates were not chilled... As they u...
    9984    Went last night to Whore Foods to get basics t...
    Name: text, dtype: object




```python
######Filter out a sample of false negatives#####
X_test[y_test > y_pred_class].sample(10, random_state=6)
```




    2494    What a great surprise stumbling across this ba...
    2504    I've passed by prestige nails in walmart 100s ...
    3448    I was there last week with my sisters and whil...
    6050    I went to sears today to check on a layaway th...
    2475    This place is so great! I am a nanny and had t...
    6318    Since I have ranted recently on poor customer ...
    7148    I now consider myself an Arizonian. If you dri...
    763     Here's the deal. I said I was done with OT, bu...
    5565    I`ve had work done by this shop a few times th...
    402     Once again Wildflower proves why it's my favor...
    Name: text, dtype: object



**Hypothesis :**
1) Naive Bayes classifier makes an assumption that features are independent given the target variable.
Features thar are marginally correlated can result in misclassification. This results in Naive Bayes missing out on sarcastic reviews which do cannot be detected by assuming feature independence. This is one of the primary reasons for misclassification. Another example could be the case of double negation being used to indicate something positive. Two negative tokens can be combined to talk about something positive.
This correlation between features will result in false classification as naive bayes assumes features independence.

2) Naive Bayes does not work well when there is class imbalance. The current data we trained one has 2499 reviews with 5 stars and 565 reviews with 1 star. This can be one of the possible hypothesis for misclassification.

3) We can also notice that there is a tendancy for negative reviews to be much longer in detail. A quick examination shows some of the positive reviews have been pretty long and have been misclassified.

4) Naive Bayes also has a tendancy to make extreme classifications with probabilties close to zero or one. There are some reviews that are too close to call which Naive Bayes misclassifies as a result of this property.



```python
###Class Imbalance##

y_train.value_counts()
```




    5    2499
    1     565
    Name: stars, dtype: int64




Calculate which 10 tokens are the most predictive of **5-star reviews**, and which 10 tokens are the most predictive of **1-star reviews**.

- **Hint:** Naive Bayes automatically counts the number of times each token appears in each class, as well as the number of observations in each class. You can access these counts via the `feature_count_` and `class_count_` attributes of the Naive Bayes model object.


```python
X_train_tokens = vect.get_feature_names()
len(X_train_tokens)
```




    16825




```python
# Naive Bayes counts the number of times each token appears in each class
nb.feature_count_
```




    array([[ 26.,   4.,   1., ...,   0.,   0.,   0.],
           [ 39.,   5.,   0., ...,   1.,   1.,   1.]])




```python
# rows represent classes, columns represent tokens
nb.feature_count_.shape
```




    (2, 16825)




```python
# number of times each token appears across all one star reviews
one_star_token_count = nb.feature_count_[0, :]
one_star_token_count
```




    array([ 26.,   4.,   1., ...,   0.,   0.,   0.])




```python
# number of times each token appears all 5 star reviews
five_star_token_count = nb.feature_count_[1, :]
five_star_token_count
```




    array([ 39.,   5.,   0., ...,   1.,   1.,   1.])




```python
# create a DataFrame of tokens with their separate bad review and good review counts
tokens = pd.DataFrame({'token':X_train_tokens, 'one_star':one_star_token_count, 'five_star':five_star_token_count}).set_index('token')
```


```python
# examine 5 random DataFrame rows
tokens.sample(5, random_state=3)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>five_star</th>
      <th>one_star</th>
    </tr>
    <tr>
      <th>token</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>amazed</th>
      <td>12.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>polytechnic</th>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>sheared</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>impersonal</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sane</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# add 1 to avoid dividing by 0
tokens['one_star'] = tokens.one_star + 1
tokens['five_star'] = tokens.five_star + 1
tokens.sample(5, random_state=3)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>five_star</th>
      <th>one_star</th>
    </tr>
    <tr>
      <th>token</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>amazed</th>
      <td>13.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>polytechnic</th>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sheared</th>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>impersonal</th>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>sane</th>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Naive Bayes counts the number of observations in each class
nb.class_count_
```




    array([  565.,  2499.])



**We convert the count of a token to frequency of the token by dividing it with the total number of tokens in the respective class(five star or one star). This is done to compute the goodness of each token as a predictor relative to other tokens in its class.**


```python
# convert the  counts into frequencies
tokens['one_star'] = tokens.one_star / nb.class_count_[0]
tokens['five_star'] = tokens.five_star / nb.class_count_[1]
tokens.sample(5, random_state=3)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>five_star</th>
      <th>one_star</th>
    </tr>
    <tr>
      <th>token</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>amazed</th>
      <td>0.005202</td>
      <td>0.00177</td>
    </tr>
    <tr>
      <th>polytechnic</th>
      <td>0.000800</td>
      <td>0.00177</td>
    </tr>
    <tr>
      <th>sheared</th>
      <td>0.000400</td>
      <td>0.00354</td>
    </tr>
    <tr>
      <th>impersonal</th>
      <td>0.000400</td>
      <td>0.00354</td>
    </tr>
    <tr>
      <th>sane</th>
      <td>0.000400</td>
      <td>0.00354</td>
    </tr>
  </tbody>
</table>
</div>




```python
# calculate the ratio of fivestar to one star for each token
tokens['fivestar_to_onestar_ratio'] = tokens.five_star / tokens.one_star
```

**The top 10 tokens that help predict five star reviews can be seen below. Words like fantastic, positive, yum, favorite, outstanding etc. which have a positive connotation tend to be more useful in predicting five star reviews**


```python
tokens.sort_values('fivestar_to_onestar_ratio', ascending=False).head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>five_star</th>
      <th>one_star</th>
      <th>fivestar_to_onestar_ratio</th>
    </tr>
    <tr>
      <th>token</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fantastic</th>
      <td>0.077231</td>
      <td>0.003540</td>
      <td>21.817727</td>
    </tr>
    <tr>
      <th>perfect</th>
      <td>0.098039</td>
      <td>0.005310</td>
      <td>18.464052</td>
    </tr>
    <tr>
      <th>yum</th>
      <td>0.024810</td>
      <td>0.001770</td>
      <td>14.017607</td>
    </tr>
    <tr>
      <th>favorite</th>
      <td>0.138055</td>
      <td>0.012389</td>
      <td>11.143029</td>
    </tr>
    <tr>
      <th>outstanding</th>
      <td>0.019608</td>
      <td>0.001770</td>
      <td>11.078431</td>
    </tr>
    <tr>
      <th>brunch</th>
      <td>0.016807</td>
      <td>0.001770</td>
      <td>9.495798</td>
    </tr>
    <tr>
      <th>gem</th>
      <td>0.016006</td>
      <td>0.001770</td>
      <td>9.043617</td>
    </tr>
    <tr>
      <th>mozzarella</th>
      <td>0.015606</td>
      <td>0.001770</td>
      <td>8.817527</td>
    </tr>
    <tr>
      <th>pasty</th>
      <td>0.015606</td>
      <td>0.001770</td>
      <td>8.817527</td>
    </tr>
    <tr>
      <th>amazing</th>
      <td>0.185274</td>
      <td>0.021239</td>
      <td>8.723323</td>
    </tr>
  </tbody>
</table>
</div>



**The top 10 tokens that help predict one star reviews can be seen below. Words like refused, disgusting, filthy etc. which have a negative connotation tend to be more useful in predicting one star reviews. The most useful token for predicting one star review is staff person. This indicates that most people who give a poor rating are driven by poor customer service by the staff.**


```python
tokens['onestar_to_fivestar_ratio'] = tokens.one_star / tokens.five_star
tokens.sort_values('onestar_to_fivestar_ratio', ascending=False).head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>five_star</th>
      <th>one_star</th>
      <th>fivestar_to_onestar_ratio</th>
      <th>onestar_to_fivestar_ratio</th>
    </tr>
    <tr>
      <th>token</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>staffperson</th>
      <td>0.0004</td>
      <td>0.030088</td>
      <td>0.013299</td>
      <td>75.191150</td>
    </tr>
    <tr>
      <th>refused</th>
      <td>0.0004</td>
      <td>0.024779</td>
      <td>0.016149</td>
      <td>61.922124</td>
    </tr>
    <tr>
      <th>disgusting</th>
      <td>0.0008</td>
      <td>0.042478</td>
      <td>0.018841</td>
      <td>53.076106</td>
    </tr>
    <tr>
      <th>filthy</th>
      <td>0.0004</td>
      <td>0.019469</td>
      <td>0.020554</td>
      <td>48.653097</td>
    </tr>
    <tr>
      <th>unacceptable</th>
      <td>0.0004</td>
      <td>0.015929</td>
      <td>0.025121</td>
      <td>39.807080</td>
    </tr>
    <tr>
      <th>acknowledge</th>
      <td>0.0004</td>
      <td>0.015929</td>
      <td>0.025121</td>
      <td>39.807080</td>
    </tr>
    <tr>
      <th>unprofessional</th>
      <td>0.0004</td>
      <td>0.015929</td>
      <td>0.025121</td>
      <td>39.807080</td>
    </tr>
    <tr>
      <th>ugh</th>
      <td>0.0008</td>
      <td>0.030088</td>
      <td>0.026599</td>
      <td>37.595575</td>
    </tr>
    <tr>
      <th>yuck</th>
      <td>0.0008</td>
      <td>0.028319</td>
      <td>0.028261</td>
      <td>35.384071</td>
    </tr>
    <tr>
      <th>fuse</th>
      <td>0.0004</td>
      <td>0.014159</td>
      <td>0.028261</td>
      <td>35.384071</td>
    </tr>
  </tbody>
</table>
</div>



These results align with our intuition that positive connotations are more prevalent in 5 star reviews and negative connotations are more prevalent in one star reviews.


```python

```
