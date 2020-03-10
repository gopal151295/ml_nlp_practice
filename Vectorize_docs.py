from nlpia.data.loaders import harry_docs as docs
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

doc_tokens = []
for doc in docs:
    doc_tokens += [sorted(tokenizer.tokenize(doc.lower()))]

len(doc_tokens)

all_doc_tokens = sum(doc_tokens, [])
len(all_doc_tokens)

lexicons = sorted(set(all_doc_tokens))
len(lexicons)
lexicons

# create zero vector for comparison
from collections import OrderedDict
zero_vec = OrderedDict((token, 0) for token in lexicons)
zero_vec

#now make copy of zero_vec and update the values for each doc
import copy
from collections import Counter
doc_vec = []
for doc in docs:
    vec = copy.copy(zero_vec)
    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)
    for token, count in token_counts.items():
        vec[token] = count/len(lexicons)
    doc_vec.append(vec)
    
doc_vec
    