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

![newValue = \frac{oldValue-min}{max-min}](https://render.githubusercontent.com/render/math?math=newValue%20%3D%20%5Cfrac%7BoldValue-min%7D%7Bmax-min%7D)

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


