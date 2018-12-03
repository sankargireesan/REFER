import csv
import itertools
from collections import Counter

from gensim import models, corpora
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

positive_model_file = "../model/LDAPositivemodel/"
negative_model_file = "../model/LDANegativemodel/"
testreview = "../testreview/testreview.csv"
mapfile = "model50category.csv"


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


# def getMapFromTopics(filename):
#     data=pd.read_csv(filename, header=None, sep='|')
#     dict = {}
#     for row in data:
#         subTopicSet = (row[3].split(','))
#         if(row[0] == 0):
#             bool_value = 'pos'
#         else:
#             bool_value = 'neg'
#         topicStr = row[1] + bool_value
#         dict[topicStr] = subTopicSet
#     return dict

def getMapsForTopics(filename):
    with open(filename) as csvfile:
        data = csv.reader(csvfile, delimiter='|')
        pos_dict = {}
        neg_dict = {}
        for datarow in data:
            subtopics = datarow[2].split(',')
            if datarow[0] == '0':
                # print('here')
                for subtopic in subtopics:
                    pos_dict[int(subtopic)] = datarow[1]
            else:
                # print('here')
                for subtopic in subtopics:
                    neg_dict[int(subtopic)] = datarow[1]

    return pos_dict, neg_dict


positive_dictionary = corpora.Dictionary.load(positive_model_file + "dictionary")
positive_lda_model = models.LdaModel.load(positive_model_file + "model")

negative_dictionary = corpora.Dictionary.load(negative_model_file + "dictionary")
negative_lda_model = models.LdaModel.load(negative_model_file + "model")

# print("Going to print positive model topics")
# print_topic_terms(positive_lda_model, num_topics=50, num_words=10, unique=False)
# print("Going to print negative model topics")
# print_topic_terms(negative_lda_model, num_topics=50, num_words=10, unique=False)


pos_dict, neg_dict = getMapsForTopics(mapfile)

# print(neg_dict)

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

review_data = open('result.csv', 'w', encoding='UTF8', newline='')
csvwriter = csv.writer(review_data)

ps = set()
ns = set()
for key in dict:
    count = 0
    total_rating = 0
    value = dict[key]
    del value[100:]
    topic_dict = {}
    count_dict = {}
    # neg_topic_dict = {}
    # neg_count_dict = {}

    for v in value:
        count += 1
        review, r = v.split('|')
        rating = int(r)
        total_rating += rating

        pos_topic_vec = get_review_topics(positive_lda_model, positive_dictionary, review)
        for doc in pos_topic_vec:
            pset = set()
            if doc[1] > 0.2:
                if doc[0] in pos_dict.keys():
                    if pos_dict[doc[0]] in pset:
                        continue
                    if pos_dict[doc[0]] in topic_dict.keys():
                        topic_dict[pos_dict[doc[0]]] += min(rating + 1, 5)
                        count_dict[pos_dict[doc[0]]] += 1
                    else:
                        topic_dict[pos_dict[doc[0]]] = min(rating + 1, 5)
                        count_dict[pos_dict[doc[0]]] = 1

                    pset.add(pos_dict[doc[0]])

        neg_topic_vec = get_review_topics(negative_lda_model, negative_dictionary, review)
        for doc in neg_topic_vec:
            nset = set()
            if doc[1] > 0.2:
                if doc[0] in neg_dict.keys():
                    if neg_dict[doc[0]] in nset:
                        continue
                    if neg_dict[doc[0]] in topic_dict.keys():
                        # print('here')
                        topic_dict[neg_dict[doc[0]]] += max(rating - 1, 1)
                        count_dict[neg_dict[doc[0]]] += 1
                    else:
                        topic_dict[neg_dict[doc[0]]] = max(rating - 1, 1)
                        count_dict[neg_dict[doc[0]]] = 1

                    nset.add(neg_dict[doc[0]])

    print("avg_rating=" + str(total_rating / count))
    result = 'avg rating=' + str(total_rating / count) + ','
    for k in topic_dict:
        if count_dict[k] > 5:
            print(str(k) + "=" + str(topic_dict[k] / count_dict[k]))
            ps.add(k)
            result += str(k) + "=" + str(topic_dict[k] / count_dict[k]) + ","

    csvwriter.writerow([result])
