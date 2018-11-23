import csv
import json

import pandas as pd

restaurantRequired = 10

with open('../data/yelp_academic_dataset_business.json', encoding='UTF8') as f:
    count = 0
    business_set = set()
    for line in f:
        js = json.loads(line)
        if js["review_count"] > 100 and js["categories"] and any(word in 'Restaurants' for word in js["categories"].split()):
            count += 1
            business_set.add(js["business_id"])

        if count > 10:
            break
print('#business :', count)

review_data = open('../testreview/testreview.csv', 'w', encoding='UTF8', newline='')
csvwriter = csv.writer(review_data)
with open('../data/yelp_academic_dataset_review.json', encoding='UTF8') as f:

    positivedataDF = pd.read_csv('../pn-words/positive-words.csv', header=None)[0]
    negativedataDF = pd.read_csv('../pn-words/negative-words.csv', header=None)[0]

    positivedata = set(positivedataDF)
    negativedata = set(negativedataDF)

    from nltk.tokenize import RegexpTokenizer

    tokenizer = RegexpTokenizer(r'\w+')

    for line in f:
        js = json.loads(line)

        if len(js["text"]) < 5:
            continue

        if js["business_id"] in business_set:
            result = ''
            notFlag = False

            for token in tokenizer.tokenize(js["text"].lower()):

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
            result = str(js["business_id"]) + '|' + result + '|' + str(js["stars"])
            csvwriter.writerow([result])
            count += 1

print(count)
