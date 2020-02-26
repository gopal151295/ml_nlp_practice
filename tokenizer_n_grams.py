import re
sentence = "Tomas Jefferson began building Monticello at the age of 26."
pattern = re.compile(r'([-\s.,;:?])+')

tokens = pattern.split(sentence)
tokens = [x for x in tokens if x and x not in '- \t\n.,;:??']
print(tokens)
"""
output:
['Tomas', 'Jefferson', 'began', 'building', 'Monticello', 'at', 'the', 'age', 'of', '26']
"""
# n gram tokenizer - 2 gram

from nltk.util import ngrams
two_grams = list(ngrams(tokens, 2))

"""
Output:
[('Tomas', 'Jefferson'),
 ('Jefferson', 'began'),
 ('began', 'building'),
 ('building', 'Monticello'),
 ('Monticello', 'at'),
 ('at', 'the'),
 ('the', 'age'),
 ('age', 'of'),
 ('of', '26')]
"""

[" ".join(x) for x in two_grams]
"""
['Tomas Jefferson',
 'Jefferson began',
 'began building',
 'building Monticello',
 'Monticello at',
 'at the',
 'the age',
 'age of',
 'of 26']
"""

# n gram tokenizer - 3 gram
three_grams = list(ngrams(tokens, 3))
"""
[('Tomas', 'Jefferson', 'began'),
 ('Jefferson', 'began', 'building'),
 ('began', 'building', 'Monticello'),
 ('building', 'Monticello', 'at'),
 ('Monticello', 'at', 'the'),
 ('at', 'the', 'age'),
 ('the', 'age', 'of'),
 ('age', 'of', '26')]
"""


