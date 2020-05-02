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

# create zero vector for comparison
from collections import OrderedDict
zero_vec = OrderedDict((token, 0) for token in lexicons)

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

# calculating tfidf vectors
doc_tfidf_vector = []
for doc in docs:
    vec = copy.copy(zero_vec)
    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)
    
    for token, count in token_counts.items():
        docs_containing_token = 0
        for _doc in docs:
            if token in _doc.lower():
                docs_containing_token += 1
        tf = count/len(lexicons)
        if docs_containing_token:
            idf = len(docs)/docs_containing_token
        else:
            idf = 0
        vec[token] = tf*idf
    doc_tfidf_vector.append(vec)
























    