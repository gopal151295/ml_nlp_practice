from nltk.tokenize.casual import TreebankWordTokenizer

sentence = """Moticello wasn't designated as UNESCO\
         World Heritage site unit 1987."""
tokenizer = TreebankWordTokenizer()
tokenizer.tokenize(sentence)
"""
Output: 

['Moticello',
 'was',
 "n't",
 'designated',
 'as',
 'UNESCO',
 'World',
 'Heritage',
 'site',
 'unit',
 '1987',
 '.']
"""