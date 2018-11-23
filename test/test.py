import csv
import itertools
from collections import Counter

from gensim import models, corpora
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

temp_file = "../finalModel/LDAReplacementmodelTFIDF/"
testreview = "../testreview/testreview.csv"


def preprocess_data(review):
    # initialize regex tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    # create English stop words list
    en_stop = set(stopwords.words('english'))
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    raw = review.lower()
    tokens = tokenizer.tokenize(raw)
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    return stemmed_tokens


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


def get_review_topics(model, cur_dict, cur_rev):
    ques_vec = cur_dict.doc2bow(preprocess_data(cur_rev))
    topic_vec = model[ques_vec]
    return topic_vec


dictionary = corpora.Dictionary.load(temp_file + "dictionary")
lda_model = models.LdaModel.load(temp_file + "model")

print_topic_terms(lda_model, num_topics=50, num_words=10, unique=False)

dict = {}
with open(testreview) as csvfile:
    readCSV = csv.reader(csvfile, delimiter='|')
    for row in readCSV:
        if len(row) == 0:
            continue
        key = row[0]
        val = row[1] + '|' + row[2]
        if key in dict:
            temp = dict[key]
            temp.append(val)
            dict[key] = temp
        else:
            temp = []
            temp.append(val)
            dict[key] = temp


s = set()
for key in dict:
    count = 0
    total_rating = 0
    value = dict[key]
    topic_dict = {}
    count_dict = {}
    del value[100:]

    for v in value:
        count += 1
        review, r = v.split('|')
        rating = int(r)
        total_rating += rating

        topic_vec = get_review_topics(lda_model, dictionary, review)
        for doc in topic_vec:
            if doc[1] > 0.2:
                if doc[0] in topic_dict:
                    topic_dict[doc[0]] += rating
                    count_dict[doc[0]] += 1
                else:
                    topic_dict[doc[0]] = rating
                    count_dict[doc[0]] = 1

    print("avg_rating  = " + str(total_rating / count))
    for k in topic_dict:
        if count_dict[k] > 30:
            print(str(k) + " " + str(topic_dict[k] / count_dict[k]))
            s.add(k)

print(sorted(s))
