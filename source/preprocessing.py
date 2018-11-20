import gensim
import pandas as pd
from gensim import models
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

filename = '../processedreview/ReviewData1.csv'
temp_file = "../model/LDAReplacementmodelTFIDF/"


def preprocess_data(doc_set):
    """
    Input  : docuemnt list
    Purpose: preprocess text (tokenize, removing stopwords, and stemming)
    Output : preprocessed text
    """
    # initialize regex tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    # create English stop words list
    en_stop = set(stopwords.words('english'))
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    # list for tokenized documents in loop
    texts = []
    # loop through document list
    for i in doc_set:
        # clean and tokenize document string
        if type(i) is not str:
            continue
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        # add tokens to list
        texts.append(stemmed_tokens)
    return texts


if __name__ == '__main__':

    documents = pd.read_csv(filename, delimiter='\n', header=None)[0]
    print('reading from file completed')
    processed_docs = preprocess_data(documents)

    print('pre processing completed')

    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=15, no_above=1.0, keep_n=100000)

    print('Gensim dictionary creation completed')

    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=50, workers=4)

    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    lda_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=50, id2word=dictionary, passes=50, workers=4)

    dictionary.save(temp_file + "dictionary")
    lda_model.save(temp_file + "model")

    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic))
