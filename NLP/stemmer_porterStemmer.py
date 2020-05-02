from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
sentence = "dish washer's washed dishes"

joined = ' '.join([stemmer.stem(w).strip("'") for w in sentence.split()])
print(joined)
