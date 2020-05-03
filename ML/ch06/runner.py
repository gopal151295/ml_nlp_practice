import svmMLiA as SVM

dataArr,labelArr = SVM.loadDataSet('data/testSet.txt')

b,alphas = SVM.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
"""
output: 
  iteration number: 29
  j not moving enough
  iteration number: 30
  iter: 30 i:17, pairs changed 1
  j not moving enough
  iteration number: 0
  j not moving enough
  iteration number: 1

  b: matrix([[-3.8486163]])
  alphas[alphas>0] = matrix([[0.09313378, 0.27456007, 0.04445935, 0.3232345 ]])

"""

# To see which points of our dataset are support vectors
for i in range(100):
  if alphas[i]>0.0:
    print(dataArr[i],labelArr[i])

"""
output:
  [4.658191, 3.507396] -1.0
  [3.457096, -0.082216] -1.0
  [2.893743, -1.643468] -1.0
  [6.080573, 0.418886] 1.0
"""

""" optimised SMOP """
b,alphas = SVM.smoP(dataArr, labelArr, 0.6, 0.001, 40)