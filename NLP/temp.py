# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import nltk
#nltk.download('punkt')
sentence = "the little yellow dog barked at the cat"

grammar = ('''
    NP: {<DT>?<JJ>*<NN>} # NP
    ''')

chunkParser = nltk.RegexpParser(grammar)
tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
tagged
