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
      - [2.2: Example: improving matches from a dating site with kNN](#22-example-improving-matches-from-a-dating-site-with-knn)
        - [2.2.3: Prepare: normalizing numeric values](#223-prepare-normalizing-numeric-values)
        - [2.2.4: Test: Testing the classifier as a whole](#224-test-testing-the-classifier-as-a-whole)
      - [2.4: Summary](#24-summary)

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
> k-Nearest Neighbors
> 
> Pros: High accuracy, insensitive to outliers, no assumptions about data
> 
> Cons: Computationally expensive, requires a lot of memory
> 
> Works with: Numeric values, nominal values

#### 2.2: Example: improving matches from a dating site with kNN
##### 2.2.3: Prepare: normalizing numeric values
When dealing with values that lie in different ranges, it’s common to normalize them. Common ranges to normalize them to are 0 to 1 or -1 to 1. To scale everything from 0 to 1, you need to apply the following formula:

$$ newValue = \frac{oldValue-min}{max-min} $$

##### 2.2.4: Test: Testing the classifier as a whole
One common task in machine learning is evaluating an algorithm’s accuracy. One way you can use the existing data is to take some portion, say 90%, to train the classifier. Then you’ll take the remaining 10% to test the classifier and see how accurate it is.

you can measure the performance of a classifier with the error rate. In classification, the error rate is the number of misclassified pieces of data divided by the total number of data points tested. An error rate of 0 means you have a perfect classifier, and an error rate of 1.0 means the classifier is always wrong.

> k-Nearest Neighbors algorithm
> 1. Calculate the distance between test data and each row of training data with the help of any of the methods namely: Euclidean, Manhattan or Hamming distance. The most commonly used method to calculate distance is Euclidean.
> 2. Now, based on the distance value, sort them in ascending order.
> 3. Next, it will choose the top K rows from the sorted array.
> 4. Now, it will assign a class to the test point based on most frequent class of these rows.

#### 2.4: Summary
The k-Nearest Neighbors algorithm is a simple and effective way to classify data.  kNN is an example of instance-based learning, where you need to have instances of data close at hand to perform the machine learning algorithm. The algorithm has to carry around the full dataset; for large datasets, this implies a large amount of storage. In addition, you need to calculate the distance measurement for every piece of data in the database, and this can be cumbersome.

An additional drawback is that kNN doesn’t give you any idea of the underlying structure of the data; you have no idea what an “average” or “exemplar” instance from each class looks like.