#https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')

import pandas as pd

from gensim import corpora, models

#from nltk.corpus import sentiwordnet as swn
#breakdown = swn.senti_synset('breakdown.n.03')

wordnet = WordNetLemmatizer()
stemmer = SnowballStemmer("english")

def lemmatize_stemming(text):
    return stemmer.stem(wordnet.lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    notFlag = False
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
            result.append(lemmatize_stemming(token))           
    return result

def replaceWords(text):
    positivedataDF = pd.read_csv('positive-words.csv', header=None)[0]
    negativedataDF = pd.read_csv('negative-words.csv', header=None)[0]

    positivedata = set(positivedataDF)
    negativedata = set(negativedataDF)
    notFlag=False
    result = ''
    for token in text.split():
        if token == 'not':
            notFlag=True
            #print('here0')
            continue

        if not notFlag and token in positivedata:
            result +=' POSITIVEREVIEW'
            notFlag = False
            #print('here1')
                
        elif notFlag and token in positivedata:
            result +=' NEGATIVEREVIEW'
            notFlag = False
            #print('here2')
                
        elif not notFlag and token in negativedata:
            result +=' NEGATIVEREVIEW'
            notFlag = False
            #print('here3')
                
        elif notFlag and token in negativedata:
            result +=' POSITIVEREVIEW'
            notFlag = False
            #print('here4')	
        else: 
            result +=' '+token
        
    return result

if __name__ == '__main__':

    documents = pd.read_csv('ReviewData2.csv', header=None)[0]

    processed_docs = documents.map(preprocess)
    processed_docs[:10]

    dictionary = gensim.corpora.Dictionary(processed_docs)

    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    print('preprocessinig completed')

    count = 0
    for k, v in dictionary.iteritems():
        print(k, v)
        count += 1
        if count > 10:
                break

    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

        # print("here 1")

        # if __name__ == '__main__':   
                # lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
                # for idx, topic in lda_model.print_topics(-1):
                        # print('Topic: {} \nWords: {}'.format(idx, topic))


        # print("here 3")




        # from gensim.test.utils import datapath
        # temp_file = datapath("model")
        # lda.save(temp_file)
        # lda = LdaModel.load(temp_file)


    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=50, id2word=dictionary, passes=50, workers=4)
    for idx, topic in lda_model_tfidf.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic))

    question = 'food is really good. but service is bad. expensive price. horrible ambinence'
    important_words = preprocess(replaceWords(question.lower()))

    ques_vec = []
    ques_vec = dictionary.doc2bow(important_words)

    topic_vec = []
    topic_vec = lda_model_tfidf[ques_vec]

    for doc in topic_vec:
        print(doc)
