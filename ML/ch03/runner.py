import trees as DT

myDat, labels = DT.createDataSet()
entropy = DT.calcShannonEnt(myDat) # 0.9709505944546686

# Let’s make the data a little messier and see how the entropy changes
# myDat[0][-1]='maybe'
# entropy = DT.calcShannonEnt(myDat)
# """
# output:
# 1.3709505944546687
# """

splittedDat = DT.splitDataSet(myDat,0,1) # [[1, 'yes'], [1, 'yes'], [0, 'no']]

splittedDat = DT.splitDataSet(myDat,0,0) # [[1, 'no'], [1, 'no']]

bestFeature = DT.chooseBestFeatureToSplit(myDat) # 0

myTree = DT.createTree(myDat,labels) # {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}


import treePlotter as TP

# TP.createPlot()
myTree = TP.retrieveTree(0)  #{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
n = TP.getNumLeafs(myTree) # 3
d = TP.getTreeDepth(myTree) # 2

TP.createPlot(myTree)

# classify
myDat, labels = DT.createDataSet()
myTree = TP.retrieveTree(0) # {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
class1 = DT.classify(myTree,labels,[1,0]) # no
class2 = DT.classify(myTree,labels,[1,1]) # yes

# storing the tree pickeld form
DT.storeTree(myTree,'data/classifierStorage.txt')
grabedTree = DT.grabTree('data/classifierStorage.txt') # {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}

# lens tree
fr = open('data/lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels=['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = DT.createTree(lenses,lensesLabels)
"""
output:
{'tearRate': {'reduced': 'no lenses',
  'normal': {'astigmatic': {'yes': {'prescript': {'myope': 'hard',
      'hyper': {'age': {'young': 'hard',
        'presbyopic': 'no lenses',
        'pre': 'no lenses'}}}},
    'no': {'age': {'young': 'soft',
      'presbyopic': {'prescript': {'myope': 'no lenses', 'hyper': 'soft'}},
      'pre': 'soft'}}}}}}
"""
TP.createPlot(lensesTree)

DT.storeTree(lensesTree,'data/lenseTreePickle.txt')

