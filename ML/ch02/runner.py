import kNN as KNN

group, labels = KNN.createDataSet()
clas = KNN.classify0([0,0], group, labels, 3)
"""
output: B
"""

"""
this file has 3 columns
  ■ Number of frequent flyer miles earned per year
  ■ Percentage of time spent playing video games
  ■ Liters of ice cream consumed per week
"""
datingDataMat,datingLabels = KNN.file2matrix('data/datingTestSet.txt')
"""
output:
  >>> datingDataMat
    array([[ 7.29170000e+04, 7.10627300e+00, 2.23600000e-01],
    [ 1.42830000e+04, 2.44186700e+00, 1.90838000e-01],
    [ 7.34750000e+04, 8.31018900e+00, 8.52795000e-01],
    ...,
    [ 1.24290000e+04, 4.43233100e+00, 9.24649000e-01],
    [ 2.52880000e+04, 1.31899030e+01, 1.05013800e+00],
    [ 4.91800000e+03, 3.01112400e+00, 1.90663000e-01]])

  >>> datingLabels[0:20]
    ['didntLike', 'smallDoses', 'didntLike', 'largeDoses', 'smallDoses',
    'smallDoses', 'didntLike', 'smallDoses', 'didntLike', 'didntLike',
    'largeDoses', 'largeDose s', 'largeDoses', 'didntLike', 'didntLike',
    'smallDoses', 'smallDoses', 'didntLike', 'smallDoses', 'didntLike']
"""

normMat, ranges, minVals = KNN.autoNorm(datingDataMat)
"""
output:
  >>> normMat
    array([[ 0.33060119, 0.58918886, 0.69043973],
    [ 0.49199139, 0.50262471, 0.13468257],
    [ 0.34858782, 0.68886842, 0.59540619],
    ...,
    [ 0.93077422, 0.52696233, 0.58885466],
    [ 0.76626481, 0.44109859, 0.88192528],
    [ 0.0975718 , 0.02096883, 0.02443895]])
  >>> ranges
    array([ 8.78430000e+04, 2.02823930e+01, 1.69197100e+00])
  >>> minVals
    array([ 0. , 0. , 0.001818])
"""

KNN.datingClassTest()
"""
output:
the total error rate is: 0.080000
16.0
"""

KNN.classifyPerson()
"""
output:
  percentage of time spent playing video games?4
  frequent flier miles earned per year?5569
  liters of ice cream consumed per year?1.213192
  You will probably like this person: in small doses
"""

KNN.handwritingClassTest()
"""
output:
  the total number of errors is: 10
  the total error rate is: 0.010571
"""

