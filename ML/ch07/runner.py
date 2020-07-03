import adaboost
import numpy as np

datMat, classesLabels = adaboost.loadSimpData()

D = np.mat(np.ones((5,1))/5)
adaboost.buildStump(datMat,classLabels,D)