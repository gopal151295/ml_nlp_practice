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
    
import math
def consine_sim(vec1, vec2):
    vec1 = [val for val in vec1.values()]
    vec2 = [val for val in vec2.values()]

    dot_prod = 0
    
    for i, v in enumerate(vec1):
        dot_prod += v * vec2[i]
    
    mag1 = math.sqrt(sum([x**2 for x in vec1]))
    mag2 = math.sqrt(sum([x**2 for x in vec2]))
    
    return dot_prod/(mag1 * mag2)
    
# create query to get the string similarity
#query = "why i am so hairy as harry"
query = "How long does it take to get to the store?"
query_vec = copy.copy(zero_vec)

query_tokens = tokenizer.tokenize(query)
query_tokens_counts = Counter(query_tokens)

for token, count in query_tokens_counts.items():
    doc_containing_token = 0
    for _doc in docs:
        if token in _doc.lower():
            doc_containing_token += 1
    if doc_containing_token == 0:
        continue
    tf = count/len(lexicons)
    if(doc_containing_token):
        idf = len(docs)/doc_containing_token
    else:
        idf = 0
    query_vec[token] = tf*idf

# checking string similarity againes stored docs_tfidf_vectors
for tfidf in doc_tfidf_vector:
    print(consine_sim(query_vec, tfidf))





















    