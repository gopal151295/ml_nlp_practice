from collections import Counter
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

from nlpia.data.loaders import kite_text
tokens = tokenizer.tokenize(kite_text.lower())

token_counts = Counter(tokens)
token_counts

# remove common stopwords
import nltk
nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words('english')
tokens = [x for x in tokens if x not in stopwords]
kite_counts = Counter(tokens)

kite_counts

kite_counts.most_common(10)
