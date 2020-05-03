import NaiveBayes as NB

listOPosts,listClasses = NB.loadDataSet()
myVocabList = NB.createVocabList(listOPosts)

setOfWord0 = NB.setOfWords2Vec(myVocabList, listOPosts[0])
# [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

setOfWord3 = NB.setOfWords2Vec(myVocabList, listOPosts[3])
# [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]

trainMat = []
for postinDoc in listOPosts:
  trainMat.append(NB.setOfWords2Vec(myVocabList, postinDoc))
  p0V,p1V,pAb=bayes.trainNB0(trainMat,listClasses)

"""
output:
  p0V
      [-2.56494936, -2.56494936, -3.25809654, -2.56494936, -2.56494936,
       -3.25809654, -2.56494936, -3.25809654, -2.15948425, -2.56494936,
       -1.87180218, -3.25809654, -2.56494936, -2.56494936, -2.56494936,
       -2.56494936, -3.25809654, -3.25809654, -3.25809654, -2.56494936,
       -3.25809654, -2.56494936, -3.25809654, -2.56494936, -2.56494936,
       -2.56494936, -2.56494936, -3.25809654, -2.56494936, -3.25809654,
       -2.56494936, -2.56494936]
  p1V
    [-3.04452244, -3.04452244, -2.35137526, -3.04452244, -2.35137526,
       -2.35137526, -3.04452244, -1.94591015, -2.35137526, -3.04452244,
       -3.04452244, -2.35137526, -3.04452244, -3.04452244, -3.04452244,
       -1.94591015, -2.35137526, -2.35137526, -2.35137526, -3.04452244,
       -2.35137526, -3.04452244, -2.35137526, -3.04452244, -3.04452244,
       -3.04452244, -3.04452244, -2.35137526, -3.04452244, -1.65822808,
       -2.35137526, -3.04452244]
  pAb
    0.5
"""

NB.testingNB()
"""
output:
  ['love', 'my', 'dalmation'] classified as:  0
  ['stupid', 'garbage'] classified as:  1
"""

# email spam test
NB.spamTest()
"""
output:
  error rate = 0.0
  classification error
  ['home', 'based', 'business', 'opportunity',
  'knocking', 'your', 'door', 'don', 'rude', 'and', 'let', 'this', 'chance',
  'you', 'can', 'earn', 'great', 'income', 'and', 'find', 'your',
  'financial', 'life', 'transformed', 'learn', 'more', 'here', 'your',
  'success', 'work', 'from', 'home', 'finder', 'experts']
"""

# rss parser
newsUrl= "https://delhi.craigslist.org/search/vnn?format=rss"
generalUrl="https://delhi.craigslist.org/search/com?format=rss"

""" rss feed parser """
import feedparser

news=feedparser.parse(newsUrl)
general=feedparser.parse(generalUrl)

vocabList,pSF,pNY=NB.localWords(news,general) # the error rate is:  0.0

NB.getTopWords(news,general)
"""
output:
  the error rate is:  0.0
  SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**
  women jee years heaven piece join earth picture will app see providing practice into neet gapio online pearl production equipment one guys frames amp mother playing origin interested feel talking based family main well card little including bit vibhushan than put cast com meticulously during globe confidence iii make more click frame https lead price padma please act money www exhibition group association leading female promotes guy book time 2011 displayed fear start thing wrote movies displays unnecessary help titled nothing held fictional download appear thrones not centre tell test established organization education series profit across around artist groo elephant those chairman amazon mind getting programs renowned properly worldwide public business habitat them 6x4 outbreak craftsmen portal empowerment reddy speaking ice very patti shop event teen just been check sculpt artisans global meet mdf club know carefully chance totally friends hit neeraj physician photos year mall chatting lack b07x how about was break want news vacation kites jobs moulds also hosp yourself learning due interesting ladies website apollo wooden crew then play coaching non wood medical founded students easy sheconnects professional going used trending new television much inlaid meeting able can back middleman today everyone physicians there center team anyone offer prathap films need copper gupta handcrafted link matured contact royal happy all networking launched guide highways consult interventi their opened gates testing cricket people imports his abatement minister slowly sportsoverload made old billion abe renjen would now wake sakura subscribe mostly fighting prime lakhs from friendship indians companies especially lover dependence currently country these announced novel only term newly highly ncr passionate kits get leave first though devil other through fight experienced must period however march planning ootklojszbq stay stroke imported nine world long china trains staying neurologist many before example unprecedented email invisible read almost kamal like closely had she tokyo 375 priyamvada cinematic continue reduce south infrastructure con water watch countries korea corridors flying grant potent destin complete metro channel but sharma sought park loan bright sent positive toeing supply called projects music nationals japanese updates scoreboards founder living domestic senior youtu cases cure inauguration republic sports disorders hundreds livelihoods agreed built ramesh embassy detailed visit surge same 5000 pollution enemy community vowed stranded specialist together platform shinzo analysis management here system vijayvargia ramped nationwide formal designated authorities rail urban what number daughter blog appeal forest yen korean indo surging updated nervous battle
  NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**
  japanese sports testing sportsoverload billion abe subscribe country novel kits other however stroke neurologist like about park news nationals cases new specialist vijayvargia nationwide korean highways based including consult interventi their opened gates cricket during globe people imports more his abatement https minister slowly made old group renjen would now wake sakura mostly fighting prime lakhs from friendship indians companies especially lover dependence currently these announced only term newly highly ncr passionate get leave first though devil through fight experienced must period march planning ootklojszbq stay around imported nine world long china trains staying many before example unprecedented public email invisible read almost kamal closely had she tokyo 375 priyamvada cinematic continue reduce south infrastructure con water watch countries korea corridors flying grant potent destin year complete metro channel but sharma was want sought loan bright sent positive also toeing supply called projects music updates scoreboards founder living domestic senior then youtu production cure inauguration republic disorders hundreds livelihoods agreed built ramesh embassy detailed visit trending surge same 5000 pollution enemy community vowed stranded together platform back shinzo analysis management here one amp system ramped formal designated authorities rail urban what link number daughter blog appeal forest yen all indo surging updated nervous battle origin interested feel talking family main jee well card little bit vibhushan than put cast com years heaven meticulously confidence iii make click frame lead price padma please act money www exhibition association leading female promotes guy book time 2011 piece displayed fear join start earth thing wrote movies displays unnecessary help titled nothing held fictional download picture appear thrones not centre will tell test established app organization education series profit across artist groo elephant those chairman see amazon providing mind getting programs renowned properly worldwide business practice habitat them 6x4 outbreak craftsmen portal empowerment reddy speaking ice very patti shop event teen just been check into neet sculpt artisans gapio online global meet pearl mdf club know carefully chance totally friends hit neeraj physician photos mall chatting lack b07x how break women vacation kites jobs moulds hosp yourself learning due interesting ladies website apollo wooden crew play coaching non wood medical founded students easy sheconnects professional equipment going used television much inlaid meeting able can middleman today guys everyone physicians there frames center team anyone offer mother prathap films need copper gupta handcrafted matured contact playing royal happy networking launched guide
"""
