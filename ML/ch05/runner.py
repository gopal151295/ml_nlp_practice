import LogisticRegression as LR
import numpy as np

dataArr,labelMat=LR.loadDataSet()

# With gradient ascent 
weights = LR.gradAscent(dataArr,labelMat)
LR.plotBestFit(weights.getA())
"""
output:
  matrix([[ 4.12414349],
        [ 0.48007329],
        [-0.6168482 ]])
"""

# with stocGradAscent0 
weights = LR.stocGradAscent0(np.array(dataArr),labelMat)
LR.plotBestFit(weights)

# with stocGradAscent1 
weights = LR.stocGradAscent1(np.array(dataArr),labelMat)
LR.plotBestFit(weights)

# with different number of iterations
# weights = stocGradAscent1(np.array(dataArr),labelMat, 50)
# plotBestFit(weights)

# weights = stocGradAscent1(np.array(dataArr),labelMat,100)
# plotBestFit(weights)

# weights = stocGradAscent1(np.array(dataArr),labelMat,200)
# plotBestFit(weights)

# weights = stocGradAscent1(np.array(dataArr),labelMat,300)
# plotBestFit(weights)

# weights = stocGradAscent1(np.array(dataArr),labelMat,500)
# plotBestFit(weights)

# Test horse fatalities
LR.multiTest()
"""
output:
  the error rate of this test is: 0.298507
  the error rate of this test is: 0.417910
  the error rate of this test is: 0.298507
  the error rate of this test is: 0.268657
  the error rate of this test is: 0.373134
  the error rate of this test is: 0.343284
  the error rate of this test is: 0.268657
  the error rate of this test is: 0.507463
  the error rate of this test is: 0.417910
  the error rate of this test is: 0.313433
  after 10 iterations the average error rate is: 0.350746
"""