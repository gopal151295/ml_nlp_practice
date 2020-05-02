from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+|$[0-9.]+|\S+')

sentence = """Moticello wasn't designated as UNESCO\
         World Heritage site unit 1987."""
tokenizer.tokenize(sentence)

"""
Output: 

['Moticello',
 'wasn',
 "'t",
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