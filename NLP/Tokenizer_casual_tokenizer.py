from nltk.tokenize.casual import casual_tokenize

sentence = """RT @TJMonticello Best day everrrrrrrr at Monticello. 
Awesommmmmmmeeeeeee day :*)"""

casual_tokenize(sentence)
"""
Output: 

['RT',
 '@TJMonticello',
 'Best',
 'day',
 'everrrrrrrr',
 'at',
 'Monticello',
 '.',
 'Awesommmmmmmeeeeeee',
 'day',
 ':*)']
"""

casual_tokenize(sentence, reduce_len=True, strip_handles=True)

"""
Output: 
['RT',
 'Best',
 'day',
 'everrr',
 'at',
 'Monticello',
 '.',
 'Awesommmeee',
 'day',
 ':*)']
"""