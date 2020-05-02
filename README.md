# ML Notes

## Document Outline
  - [Part 1: Classification](#part-1-classification)
    1. [Machine learning basics](#1-machine-learning-basics)

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

