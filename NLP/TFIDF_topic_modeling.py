#import data
from nlpia.data.loaders import kite_text, kite_history

# to lowercase
kite_intro = kite_text.lower()
kite_history = kite_history.lower()

#tokenizer initialization
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

# extract tokens
tokens_intro = tokenizer.tokenize(kite_intro)
tokens_history = tokenizer.tokenize(kite_history)

# tokens lengths
len_intro_tokens = len(tokens_intro)
len_history_tokens = len(tokens_history)

# tokens counters
from collections import Counter
counter_intro_tokens = Counter(tokens_intro)
counter_history_tokens = Counter(tokens_history)

tf_intro = {}
tf_history = {}

# terms frequencies of 'kite' tokens in intro and history
tf_intro['kite'] = counter_intro_tokens['kite'] / len_intro_tokens
tf_history['kite'] = counter_history_tokens['kite'] / len_history_tokens

# terms frequencies of 'and' tokens in intro and history
tf_intro['and'] = counter_intro_tokens['and'] / len_intro_tokens
tf_history['and'] = counter_history_tokens['and'] / len_history_tokens

# terms frequencies of 'china' tokens in intro and history
tf_intro['china'] = counter_intro_tokens['china'] / len_intro_tokens
tf_history['china'] = counter_history_tokens['china'] / len_history_tokens

# Calculate number of documents containing specific words
num_doc_containing_and = 0
num_doc_containing_kite = 0
num_doc_containing_china = 0

for doc in [tokens_intro, tokens_history]:
    if 'and' in doc:
        num_doc_containing_and += 1
    if 'kite' in doc:
        num_doc_containing_kite += 1
    if 'china' in doc:
        num_doc_containing_china += 1
        
# for calculating idf
num_doc = 2

idf_intro = {}
idf_history = {}

# inverse terms frequencies of 'kite' tokens in intro and history
idf_intro['kite'] = num_doc/num_doc_containing_kite
idf_history['kite'] = num_doc/num_doc_containing_kite

# inverse terms frequencies of 'and' tokens in intro and history
idf_intro['and'] = num_doc/num_doc_containing_and
idf_history['and'] = num_doc/num_doc_containing_and

# inverse terms frequencies of 'china' tokens in intro and history
idf_intro['china'] = num_doc/num_doc_containing_china
idf_history['china'] = num_doc/num_doc_containing_china

# calculating tfidf
tfidf_intro = {}
tfidf_history = {}

tfidf_intro['kite'] = tf_intro['kite'] * idf_intro['kite']
tfidf_intro['and'] = tf_intro['and'] * idf_intro['and']
tfidf_intro['china'] = tf_intro['china'] * idf_intro['china']

tfidf_history['kite'] = tf_history['kite'] * tf_history['kite']
tfidf_history['and'] = tf_history['and'] * tf_history['and']
tfidf_history['china'] = tf_history['china'] * tf_history['china']

