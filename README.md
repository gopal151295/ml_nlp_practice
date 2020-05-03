- [ML Notes](#ml-notes)
  - [Part 1: Classification](#part-1-classification)
    - [1: Machine learning basics](#1-machine-learning-basics)
      - [1.2: Key Terminology](#12-key-terminology)
      - [1.3: Key tasks of machine learning](#13-key-tasks-of-machine-learning)
      - [1.4: How to choose the right algorithm](#14-how-to-choose-the-right-algorithm)
      - [1.5: Steps in developing a machine learning application](#15-steps-in-developing-a-machine-learning-application)
      - [1.8: Summary](#18-summary)
    - [2: Classifying with k-Nearest Neighbors](#2-classifying-with-k-nearest-neighbors)
      - [2.1: Classifying with distance measurements](#21-classifying-with-distance-measurements)
        - [2.1.3: How to test a classifier](#213-how-to-test-a-classifier)
      - [2.2: Example: improving matches from a dating site with kNN](#22-example-improving-matches-from-a-dating-site-with-knn)
        - [2.2.3: Prepare: normalizing numeric values](#223-prepare-normalizing-numeric-values)
        - [2.2.4: Test: Testing the classifier as a whole](#224-test-testing-the-classifier-as-a-whole)
      - [2.4: Summary](#24-summary)
    - [3: Decision trees](#3-decision-trees)
      - [3.1: Tree Construction](#31-tree-construction)
        - [3.1.1: Information gain](#311-information-gain)
        - [3.1.2: Splitting the dataset](#312-splitting-the-dataset)
      - [3.3: Testing and storing the classifier](#33-testing-and-storing-the-classifier)
        - [3.3.2: Persisting the decision tree](#332-persisting-the-decision-tree)
      - [3.5: Summary](#35-summary)
    - [4: Naïve Bayes](#4-na%c3%afve-bayes)
      - [4.4: Document classification with naïve Bayes](#44-document-classification-with-na%c3%afve-bayes)
      - [4.5: Classifying text with Python](#45-classifying-text-with-python)
        - [4.5.3: Modifying the classifier for real-world conditions](#453-modifying-the-classifier-for-real-world-conditions)
        - [4.5.4: The bag-of-words document model](#454-the-bag-of-words-document-model)
      - [4.8: Summary](#48-summary)
    - [5: Logistic regression](#5-logistic-regression)
      - [5.1: Classification with logistic regression and the sigmoid function](#51-classification-with-logistic-regression-and-the-sigmoid-function)
      - [5.2: Using optimization to find the best regression coefficients](#52-using-optimization-to-find-the-best-regression-coefficients)
        - [5.2.1: Gradient Ascent](#521-gradient-ascent)
        - [5.2.2: Using gradient ascent to find the best parameters](#522-using-gradient-ascent-to-find-the-best-parameters)
        - [5.2.4: Stochastic gradient ascent](#524-stochastic-gradient-ascent)
      - [5.3: Estimating horse fatalities from colic](#53-estimating-horse-fatalities-from-colic)
        - [5.3.1: Dealing with missing values in the data](#531-dealing-with-missing-values-in-the-data)
      - [5.4: Summary](#54-summary)
    - [6: Support vector machines](#6-support-vector-machines)
      - [6.1: Separating data with the maximum margin](#61-separating-data-with-the-maximum-margin)
      - [6.2: Finding the maximum margin](#62-finding-the-maximum-margin)
      - [6.3: Efficient optimization with the SMO algorithm](#63-efficient-optimization-with-the-smo-algorithm)
        - [6.3.1: Platt’s SMO algorithm](#631-platts-smo-algorithm)
        - [6.3.2: Solving small datasets with the simplified SMO](#632-solving-small-datasets-with-the-simplified-smo)
      - [6.4: Speeding up optimization with the full Platt SMO](#64-speeding-up-optimization-with-the-full-platt-smo)

# ML Notes

## Part 1: Classification

### 1: Machine learning basics

#### 1.2: Key Terminology
The four things we’ve measured are called features

In classification the target variable takes on a nominal
value, and in the task of regression its value could be continuous. In a training set the
target variable is known.

#### 1.3: Key tasks of machine learning
Classification and regression are examples of supervised learning. This set of problems is known as supervised because we’re telling the algorithm what to predict.

The opposite of supervised learning is a set of tasks known as unsupervised learning. In unsupervised learning, there’s no label or target value given for the data. A task where we group similar items together is known as clustering. In unsupervised learning, we may also want to find statistical values that describe the data. This is known as density estimation. Another task of unsupervised learning may be reducing the data from many features to a small number so that we can properly visualize it in two or three dimensions. 

#### 1.4: How to choose the right algorithm
If you’re trying to predict or forecast a target value, then you need to look into supervised learning. If not, then unsupervised learning is the place you want to be. If you’ve chosen supervised learning, what’s your target value? Is it a discrete value like Yes/No, 1/2/3, A/B/C, or Red/Yellow/Black? If so, then you want to look into classification. If the target value can take on a number of values, say any value from 0.00 to 100.00, or -999 to 999, or + to -, then you need to look into regression.

If you’re not trying to predict a target value, then you need to look into unsupervised learning. Are you trying to fit your data into some discrete groups? If so and that’s all you need, you should look into clustering. Do you need to have some numerical estimate of how strong the fit is into each group? If you answer yes, then you probably should look into a density estimation algorithm.

#### 1.5: Steps in developing a machine learning application
1. Collect data
2. Prepare the input data
3. Analyze the input data
4. Train the algorithm
5. Test the algorithm
6. Use it

#### 1.8: Summary
Classification, one the popular and essential tasks of machine learning, is used to place an unknown piece of data into a known group.

### 2: Classifying with k-Nearest Neighbors
Training -> doesn’t apply to knn

#### 2.1: Classifying with distance measurements
> **k-Nearest Neighbors**
> 
> **Pros**: High accuracy, insensitive to outliers, no assumptions about data
> 
> **Cons**: Computationally expensive, requires a lot of memory
> 
> **Works with**: Numeric values, nominal values

##### 2.1.3: How to test a classifier
To test out a classifier, you start with some known data so you can hide the answer
from the classifier and ask the classifier for its best guess. You can add up the number
of times the classifier was wrong and divide it by the total number of tests you gave it.
This will give you the error rate, which is a common measure to gauge how good a classifier is doing on a dataset. An error rate of 0 means you have a perfect classifier, and
an error rate of 1.0 means the classifier is always wrong.

#### 2.2: Example: improving matches from a dating site with kNN
##### 2.2.3: Prepare: normalizing numeric values
When dealing with values that lie in different ranges, it’s common to normalize them. Common ranges to normalize them to are 0 to 1 or -1 to 1. To scale everything from 0 to 1, you need to apply the following formula:

![\Large newValue = \frac{oldValue-min}{max-min}](https://render.githubusercontent.com/render/math?math=%5CLarge%20newValue%20%3D%20%5Cfrac%7BoldValue-min%7D%7Bmax-min%7D)

##### 2.2.4: Test: Testing the classifier as a whole
One common task in machine learning is evaluating an algorithm’s accuracy. One way you can use the existing data is to take some portion, say 90%, to train the classifier. Then you’ll take the remaining 10% to test the classifier and see how accurate it is.

you can measure the performance of a classifier with the error rate. In classification, the error rate is the number of misclassified pieces of data divided by the total number of data points tested. An error rate of 0 means you have a perfect classifier, and an error rate of 1.0 means the classifier is always wrong.

*pseudo code k-Nearest Neighbors algorithm*
```python
for every point in our dataset:
  calculate the distance between test data and each row of training data with the help of any of the methods
  # Euclidean, Manhattan or Hamming distance. The most commonly used method to calculate distance is Euclidean
  sort the distances in increasing order
  take k items with lowest distances to test data
  find the majority class among these items
  return the majority class as our prediction for the class of inX
```

#### 2.4: Summary
The k-Nearest Neighbors algorithm is a simple and effective way to classify data.  kNN is an example of instance-based learning, where you need to have instances of data close at hand to perform the machine learning algorithm. The algorithm has to carry around the full dataset; for large datasets, this implies a large amount of storage. In addition, you need to calculate the distance measurement for every piece of data in the database, and this can be cumbersome.

An additional drawback is that kNN doesn’t give you any idea of the underlying structure of the data; you have no idea what an “average” or “exemplar” instance from each class looks like.

### 3: Decision trees 
Splitting datasets one feature at a time

Decision Tree algorithm belongs to the family of supervised learning algorithms. Unlike other supervised learning algorithms, the decision tree algorithm can be used for solving regression and classification problems too.

> **Decision trees**
> 
> **Pros**: Computationally cheap to use, easy for humans to understand learned results, missing values OK, can deal with irrelevant features
> 
> **Cons**: Prone to overfitting
> 
> **Works with**: Numeric values, nominal values

To build a decision tree, you need to make a first decision on the dataset to dictate which feature is used to split the data. To determine this, you try every feature and measure which split will give you the best results. After that, you’ll split the dataset into subsets. The subsets will then traverse down the branches of the first decision node. If the data on the branches is the same class, then you’ve properly classified it and don’t need to continue splitting it. If the data isn’t the same, then you need to repeat the splitting process on this subset. 

*Pseudo-code for a function called createBranch()*
```python
Check if every item in the dataset is in the same class:
  if so:
    return the class label
  else:
    find the best feature to split the data
    split the dataset
    create a branch node
      for each split
        call createBranch and add the result to the branch node
    return branch node
```

#### 3.1: Tree Construction

##### 3.1.1: Information gain
We choose to split our dataset in a way that makes our unorganized data more organized.
One way to organize this messiness is to measure the information. 

Using information theory, you can measure the information before and after the split. The change in information before and after the split is known as the information gain. you can split your data across every feature to see which split gives you the highest information gain. The split with the highest information gain is your best option

The measure of information of a set is known as the Shannon entropy, or just entropy for short. Its name comes from the father of information theory, Claude Shannon. Entropy is a measure of disorder or uncertainty and the goal of machine learning models and Data Scientists in general is to reduce uncertainty.

[Entropy: How Decision Trees Make Decisions](https://towardsdatascience.com/entropy-how-decision-trees-make-decisions-2946b9c18c8)

##### 3.1.2: Splitting the dataset

We will combine the Shannon entropy calculation and thesplitDataSet() function to cycle through the dataset and decide which feature is the best to split on

#### 3.3: Testing and storing the classifier

##### 3.3.2: Persisting the decision tree
Pickle, to serialize objects, as shown in the following listing. Serializing objects allows you to store them for later use. Serializing can be done with any object, and dictionaries work as well.

#### 3.5: Summary
Starting with a dataset, you can measure the inconsistency of a set or the entropy to find a way to split the set until all the data belongs to the same class. The ID3 algorithm can split nominal-valued datasets. Recursion is used in tree-building algorithms to turn a dataset into a decision tree. The tree is easily represented in a Python dictionary rather than a special data structure.

The Python Pickle module can be used for persisting our tree

This overfitting can be removed by pruning the decision tree, combining adjacent leaf nodes that don’t provide a large amount of information gain.

The algorithm we used in this chapter, ID3, is good but not the best. ID3 can’t handle numeric values. We could use continuous values by quantizing them into discrete bins, but ID3 suffers from other problems if we have too many splits

### 4: Naïve Bayes
Classifying with probability theory. We start out with the simplest probabilistic classifier and then make a few assumptions and learn the naïve Bayes classifier. It’s called naïve because the formulation makes some naïve assumptions. 

> **Naïve Bayes**
> 
> **Pros**: Works with a small amount of data, handles  multiple classes
> 
> **Cons**: Sensitive to how the input data is prepared
> 
> **Works with**: Nominal values

Naïve Bayes is a subset of Bayesian decision theory

That’s Bayesian decision theory in a nutshell: choosing the decision with the highest probability.

![\Large P(gray|bucketB) = \frac{P(gray &nbsp; And &nbsp; bucketB)}{P(bucketB)}](https://render.githubusercontent.com/render/math?math=%5CLarge%20P(gray%7CbucketB)%20%3D%20%5Cfrac%7BP(gray%20%26nbsp%3B%20And%20%26nbsp%3B%20bucketB)%7D%7BP(bucketB)%7D)

![\Large P(c|x) = \frac{P(x|c)}{P(x)}](https://render.githubusercontent.com/render/math?math=%5CLarge%20P(c%7Cx)%20%3D%20%5Cfrac%7BP(x%7Cc)%7D%7BP(x)%7D)

#### 4.4: Document classification with naïve Bayes

*Assumptions in naive bayes:*
1. Independence among the features, means statistical independence; one feature or word is just as likely by itself as it is next to other words. 
2. Every feature is equally important. We know that isn’t true either.

*pseudo code: conditional probabilities for each class*
```python
for every training document:
  for each class:
    if a token appears in the document:
      increment the count for that token
    increment the count for tokens
  for each class:
    for each token:
      divide the token count by the total token count to get conditional probabilities
  return conditional probabilities for each class
```
#### 4.5: Classifying text with Python

##### 4.5.3: Modifying the classifier for real-world conditions
When we attempt to classify a document, we multiply a lot of probabilities together to get the probability that a document belongs to a given class. This will look something like p(w0|1)p(w1|1)p(w2|1). If any of these numbers are 0, then when we multiply them together we get 0. To lessen the impact of this, we’ll initialize all of our occurrence counts to 1, and we’ll initialize the denominators to 2

Another problem is underflow: doing too many multiplications of small numbers. When we go to calculate the product p(w0|ci)p(w1|ci)p(w2|ci)...p(wN|ci) and many of these numbers are very small, we’ll get underflow, or an incorrect answer. (Try to multiply many small numbers in Python. Eventually it rounds off to 0.) One solution to this is to take the natural logarithm of this product. If you recall from algebra, ln(a*b) = ln(a)+ln(b). Doing this allows us to avoid the underflow or round-off error problem.

##### 4.5.4: The bag-of-words document model
Up until this point we’ve treated the presence or absence of a word as a feature. This could be described as a set-of-words model. If a word appears more than once in a document, that might convey some sort of information about the document over just the word occurring in the document or not. This approach is known as a bag-of-words model. A bag of words can have multiple occurrences of each word, whereas a set of words can have only one occurrence of each word. In bag-of-word, every time it encounters a word, it increments the word vector rather than setting the word vector to 1 for a given index

#### 4.8: Summary
Using probabilities can sometimes be more effective than using hard rules for classification. Bayesian probability and Bayes’ rule gives us a way to estimate unknown probabilities from known values. You can reduce the need for a lot of data by assuming conditional independence among the features in your data. The assumption we make is that the probability of one word doesn’t depend on any other words in the document. We know this assumption is a little simple. That’s why it’s known as naïve Bayes. Despite its incorrect assumptions, naïve Bayes is effective at classification. There are a number of practical considerations when implementing naïve Bayes in a modern programming language. Underflow is one problem that can be addressed by using the logarithm of probabilities in your calculations. The bag-of-words model is an improvement on the set-of-words model when approaching document classification. There are a number of other improvements, such as removing stop words, and you can spend a long time optimizing a tokenizer.

### 5: Logistic regression
This is the first chapter where we encounter optimization algorithms. Perhaps you’ve seen some data points and then someone fit a line called the best-fit line to these points; that’s regression.

What happens in logistic regression is we have a bunch of data, and with the data we try to build an equation to do classification for us. 

The regression aspects means that we try to find a best-fit set of parameters. Finding the best fit is similar to regression, and in this method it’s how we train our classifier. We’ll use optimization algorithms to find these best-fit parameters. This best-fit stuff is where the name regression comes from.

In our study of optimization algorithms, you’ll learn gradient ascent, and then we’ll look at a modified version called stochastic gradient ascent. These optimization algorithms will be used to train our classifier.

#### 5.1: Classification with logistic regression and the sigmoid function

> **Logistic regression**
> 
> **Pros**: Computationally inexpensive, easy to implement, knowledge representation easy to interpret
> 
> **Cons**: Prone to underfitting, may have low accuracy
> 
> **Works with**: Numeric values, nominal values

We will use sigmoid function here, it is given by

![\Large \sigma(z) = \frac{1}{1+e^{-z}}](https://render.githubusercontent.com/render/math?math=%5CLarge%20%5Csigma(z)%20%3D%20%5Cfrac%7B1%7D%7B1%2Be%5E%7B-z%7D%7D)

At 0 the value of the sigmoid is 0.5. For increasing values of x, the sigmoid will approach 1, and for decreasing values of x, the sigmoid will approach 0. On a large enough scale (the bottom frame of figure 5.1), the sigmoid looks like a step function.

For the logistic regression classifier we’ll take our features and multiply each one by a weight and then add them up. This result will be put into the sigmoid, and we’ll get a number between 0 and 1. Anything above 0.5 we’ll classify as a 1, and anything below 0.5 we’ll classify as a 0. You can also think of logistic regression as a probability estimate.

#### 5.2: Using optimization to find the best regression coefficients

The input to the sigmoid function described will be z, where z is given by the following:

![\Large z = w_0x_0+w_1x_1+w_2x_2+...+w_nx_n](https://render.githubusercontent.com/render/math?math=%5CLarge%20z%20%3D%20w_0x_0%2Bw_1x_1%2Bw_2x_2%2B...%2Bw_nx_n)

In vector notation we can write this as z=wTx. All that means is that we have two vectors of numbers and we’ll multiply each element and add them up to get one number. The vector x is our input data, and we want to find the best coefficients w.

##### 5.2.1: Gradient Ascent

Gradient ascent is based on the idea that if we want to find the maximum point on a function, then the best way to move is in the direction of the gradient.

![\Large \nabla f(x,y) = \Large \Big ( \frac { \frac {\partial f(x,y)}{\partial x}}{ \frac {\partial f(x,y)}{\partial y}} \Big )](https://render.githubusercontent.com/render/math?math=%5CLarge%20%5Cnabla%20f(x%2Cy)%20%3D%20%5CLarge%20%5CBig%20(%20%5Cfrac%20%7B%20%5Cfrac%20%7B%5Cpartial%20f(x%2Cy)%7D%7B%5Cpartial%20x%7D%7D%7B%20%5Cfrac%20%7B%5Cpartial%20f(x%2Cy)%7D%7B%5Cpartial%20y%7D%7D%20%5CBig%20))

The gradient operator will always point in the direction of the greatest increase. We’ve talked about direction, but I didn’t mention anything to do with magnitude of movement. The magnitude, or step size, we’ll take is given by the parameter.

**Ascent**: We’re trying to maximize some function.

![\Large w = w + \alpha \nabla_wf(w)](https://render.githubusercontent.com/render/math?math=%5CLarge%20w%20%3D%20w%20%2B%20%5Calpha%20%5Cnabla_wf(w))

**Descent** : We’re trying to minimize some function rather than maximize it.

![\Large w = w - \alpha \nabla_wf(w)](https://render.githubusercontent.com/render/math?math=%5CLarge%20w%20%3D%20w%20-%20%5Calpha%20%5Cnabla_wf(w))

Where alpha is the step size of the learning. This step is repeated until we reach a stopping condition: either a specified number of steps or the algorithm is within a certain tolerance margin.

##### 5.2.2: Using gradient ascent to find the best parameters

*pseudo code: for gradient ascent*
```python
Start with the weights all set to 1
for i in range(numOfCycles):
  calculate the gradient of the entire dataset
  update the weights vector by alpha*gradient
  return the weights vector
```

##### 5.2.4: Stochastic gradient ascent

An alternative to gradient ascent ( computaionally expensive ) algorith is Stochastic gradient ascent. Stochastic gradient ascent is an example of an online learning algorithm. This is known as online because we can incrementally update the classifier as new data comes in rather than all at once. The all-at-once method is known as batch processing.

*pseudo code: for Stochastic gradient ascent*
```python
Start with the weights all set to 1
for data in dataset:
  calculate the gradient of one piece of data
  update the weights vector by alpha*gradient
  return the weights vector
```

Performance of simple stochastic gradient ascent is even worse than gradient ascent. If you think about what’s happening, it should be obvious that there are pieces of data that don’t classify correctly and cause a large change in the weights. We’d like to see the algorithm converge to a single value rather than oscillate, and we’d like to see the weights converge more quickly.

We'll make some modification in simple stochastic ascent

1. In improved version, alpha changes on each iteration. This will improve the oscillations that occur in the dataset. Alpha decreases as the number of iterations increases, but it never reaches 0 because there’s a constant term.
2. In improved version, you’re randomly selecting each instance to use in updating the weights. This will reduce the periodic variations that you saw in simple stochastic ascent.

#### 5.3: Estimating horse fatalities from colic

We’ll first handle the problem of how to deal with missing values in a dataset, and then we’ll use logistic regression and stochastic gradient ascent to forecast whether a horse will live or die.

##### 5.3.1: Dealing with missing values in the data

Here are some options:
1. Use the feature’s mean value from all the available data.
2. Fill in the unknown with a special value like -1
3. Ignore the instance
4. Use a mean value from similar items
5. Use another machine learning algorithm to predict the value

#### 5.4: Summary 

Logistic regression is finding best-fit parameters to a nonlinear function called the sigmoid. Methods of optimization can be used to find the best-fit parameters. Among the optimization algorithms, one of the most common algorithms is gradient ascent.

Gradient ascent can be simplified with stochastic gradient ascent. Stochastic gradient ascent can do as well as gradient ascent using far fewer computing resources. In addition, stochastic gradient ascent is an online algorithm; it can update what it has learned as new data comes in rather than reloading all of the data as in batch processing.

One major problem in machine learning is how to deal with missing values in the data. There’s no blanket answer to this question. It really depends on what you’re doing with the data. There are a number of solutions, and each solution has its own advantages and disadvantages.

### 6: Support vector machines

Support vector machines are considered by some people to be the best stock classifier. By stock, I mean not modified. This means you can take the classifier in its basic form and run it on the data, and the results will have low error rates. Support vector machines make good decisions for data points that are outside the training set.

There are many implementations of support vector machines but we’ll focus on one of the most popular implementations: the sequential minimal optimization (SMO) algorithm. After that, you’ll see how to use something called kernels to extend SVMs to a larger number of datasets. 

You can use an SVM in almost any classification problem. One thing to note is that SVMs are binary classifiers. You’ll need to write a little more code to use an SVM on a problem with more than two classes.

#### 6.1: Separating data with the maximum margin

> **Support Vector Machine**
> 
> **Pros**: Low generalization error, computationally inexpensive, easy to interpret results
> 
> **Cons**: Sensitive to tuning parameters and kernel choice; natively only handles binary classification
> 
> **Works with**: Numeric values, nominal values

A dataset is said to be linearly separable if it is possible to draw a line that can separate the red and green points from each other.

The line used to separate the dataset is called a separating hyperplane. In our simple 2D plots, it’s just a line. But, if we have a dataset with three dimensions, we need a plane to separate the data; and if we have data with 1024 dimensions, we need something with 1023 dimensions to separate the data. What do you call something with 1023 dimensions? How about N-1 dimensions? It’s called a hyperplane. The hyperplane is our decision boundary. Everything on one side belongs to one class, and everything on the other side belongs to a different class.

We’d like to find the point closest to the separating hyperplane and make sure this is as far away from the separating line as possible. This is known as margin.

The points closest to the separating hyperplane are known as support vectors. Now that we know that we’re trying to maximize the distance from the separating line to the support vectors, we need to find a way to optimize this problem.

#### 6.2: Finding the maximum margin

Our separating hyperplane has the form
![w^tx+b](https://render.githubusercontent.com/render/math?math=w%5Etx%2Bb)

Distance of a point to separting plane is given by

![\Large \frac {|w^tx+b|}{\|w\|}](https://render.githubusercontent.com/render/math?math=%5CLarge%20%5Cfrac%20%7B%7Cw%5Etx%2Bb%7C%7D%7B%5C%7Cw%5C%7C%7D)

When we’re doing this and deciding where to place the separating line, this margin is calculated by
label*(wTx+b)

This is where the -1 and 1 class labels help out. If a point is far away from the separating plane on the positive side, then 
![w^tx+b](https://render.githubusercontent.com/render/math?math=w%5Etx%2Bb) 
will be a large positive number, and 
![label*(w^tx+b)](https://render.githubusercontent.com/render/math?math=label*(w%5Etx%2Bb)) 
will give us a large number. If it’s far from the negative side and has a negative label, 
![label*(w^tx+b)](https://render.githubusercontent.com/render/math?math=label*(w%5Etx%2Bb)) 
will also give us a large positive number.

#### 6.3: Efficient optimization with the SMO algorithm

##### 6.3.1: Platt’s SMO algorithm

The SMO algorithm works to find a set of alphas and b. Once we have a set of alphas, we can easily compute our weights w and get the separating hyperplane.

Here’s how the SMO algorithm works: it chooses two alphas to optimize on each cycle. Once a suitable pair of alphas is found, one is increased and one is decreased. To be suitable, a set of alphas must meet certain criteria. One criterion a pair must meet is that both of the alphas have to be outside their margin boundary. The second criterion is that the alphas aren’t already clamped or bounded.

##### 6.3.2: Solving small datasets with the simplified SMO

The simplification uses less code but takes longer at runtime. The outer loops of the Platt SMO algorithm determine the best alphas to optimize. We’ll skip that for this simplified version and select pairs of alphas by first going over every alpha in our dataset. Then, we’ll choose the second alpha randomly from the remaining alphas.

*pseudo code: for for our first version of the SMO algorithm*
```python
create an alphas vector filled with 0s
while the number of iterations is less than MaxIterations:
  for every data vector in the dataset:
    if the data vector can be optimized:
      select another data vector at random
      optimize the two vectors together
      if the vectors can’t be optimized ➞ break
  if no vectors were optimized ➞ increment the iteration count
```

#### 6.4: Speeding up optimization with the full Platt SMO

The optimization portion where we change alphas and do all the algebra stays the same. The only difference is how we select which alpha to use in the optimization. 