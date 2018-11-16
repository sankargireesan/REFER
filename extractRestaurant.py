import json
import re
import csv
import pandas as pd

with open('yelp_academic_dataset_business.json', encoding='UTF8') as f:
	count = 0
	business_set = set()
	for line in f:
		js = json.loads(line)
		if js["review_count"] > 100 and js["categories"] and any(word in 'Restaurants Food bar Bar food' for word in js["categories"]):
			count+=1
		#	print(js["categories"])

		# list1 = ["Restaurants", "Food", "bar"]
		# list2 = js["categories"]
		# if js["categories"] and any(elem in list1 for elem in list2):
			# count+=1
			#print(js["categories"])
			
			business_set.add(js["business_id"])
print(count)

review_data = open('ReviewData2.csv', 'w', encoding='UTF8')
csvwriter = csv.writer(review_data)
with open('yelp_academic_dataset_review.json', encoding='UTF8') as f:
	count = 0

	positivedataDF = pd.read_csv('positive-words.csv', header=None)[0]
	negativedataDF = pd.read_csv('negative-words.csv', header=None)[0]

	positivedata = set(positivedataDF)
	negativedata = set(negativedataDF)
	
	for line in f:
		js = json.loads(line)
		if(len(js["text"]) < 5) :
			continue
			
		if js["business_id"] in business_set:
			result = ''
			notFlag = False
			for token in js["text"].lower().split():
				
				
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

			csvwriter.writerow([result])
			# review_data.write(js["text"])
			# review_data.write('\n')
			count+=1
			#print(js["text"])
			
			if count > 10000:
				break
print(count)