import itertools
from collections import Counter

import gensim
import nltk
import numpy as np
import pandas as pd
from gensim import models, corpora
from nltk.stem import WordNetLemmatizer, SnowballStemmer

np.random.seed(2018)
nltk.download('wordnet')
nltk.download('stopwords')

wordnet = WordNetLemmatizer()
stemmer = SnowballStemmer("english")

temp_file = "../model/LDAReplacementmodelTFIDF/"


def lemmatize_stemming(text):
    return stemmer.stem(wordnet.lemmatize(text, pos='v'))


def replaceWords(text):
    positivedataDF = pd.read_csv('../pn-words/positive-words.csv', header=None)[0]
    negativedataDF = pd.read_csv('../pn-words/negative-words.csv', header=None)[0]

    positivedata = set(positivedataDF)
    negativedata = set(negativedataDF)
    notFlag = False
    result = ''
    for token in text.split():
        if token == 'not':
            notFlag = True
            # print('here0')
            continue

        if not notFlag and token in positivedata:
            result += ' POSITIVEREVIEW'
            notFlag = False
            # print('here1')

        elif notFlag and token in positivedata:
            result += ' NEGATIVEREVIEW'
            notFlag = False
            # print('here2')

        elif not notFlag and token in negativedata:
            result += ' NEGATIVEREVIEW'
            notFlag = False
            # print('here3')

        elif notFlag and token in negativedata:
            result += ' POSITIVEREVIEW'
            notFlag = False
            # print('here4')
        else:
            result += ' ' + token

    return result


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
            result.append(lemmatize_stemming(token))
    return result


def print_topic_terms(model, num_topics=10, num_words=10, unique=False):
    results = model.print_topics(num_topics=num_topics, num_words=num_words)
    if not unique:
        print('=============================== Terms Per Topic ===============================')
        for r in results:
            topic = r[0]
            term_list = r[1]

            term_list = term_list.split('"')[1::2]
            topic_terms = [term for term in term_list]
            print('{}\t{}'.format(topic, topic_terms))
    else:
        terms = [x[1] for x in results]
        term_lists = [x.split('"')[1::2] for x in terms]

        flatList = itertools.chain.from_iterable(term_lists)
        term_counts = Counter(flatList)

        # non_unique_terms = term_counts
        test = dict(term_counts)

        # extract terms that appear more than once
        non_unique_terms = [key for key, value in test.items() if value > 1]

        print('============================ Unique Terms Per Topic ===========================')
        for r in results:
            topic = r[0]
            term_list = r[1]

            term_list = term_list.split('"')[1::2]
            topic_terms = [term for term in term_list if term not in non_unique_terms]
            print('{}\t{}'.format(topic, topic_terms))


lda_model = models.LdaModel.load(temp_file + "model")

print_topic_terms(lda_model, num_topics=50, num_words=10, unique=False)

dictionary = corpora.Dictionary.load(temp_file + "dictionary")

question = 'service is bad'
important_words = preprocess(replaceWords(question.lower()))

ques_vec = dictionary.doc2bow(important_words)
topic_vec = lda_model[ques_vec]

for doc in topic_vec:
    print(doc)
