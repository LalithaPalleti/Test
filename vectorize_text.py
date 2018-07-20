#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        #print path
        path = path.strip()
        path = path.replace('.','_')
        #print path
        #temp_counter += 1
        #if temp_counter < 200:
            #print path[:-1]
        path = os.path.join('..', path)
        #path ='C:/Users/Sravanthi/Documents/Udacity/ud120-projects-master/ud120-projects-master/text_learning/maildir/bailey-s/deleted-items/101_' 
        #print path
        #email = open(path, "r")
        email = open(path, "r")

        ### use parseOutText to extract the text from the opened email
        text = parseOutText(email)
        #print (type(text))

        ### use str.replace() to remove any instances of the words
        ### ["sara", "shackleton", "chris", "germani"]
        remove_these_words = ["sara", "shackleton", "chris", "germani"]
        for i in remove_these_words:
            text = text.replace(i,'')

            ### append the text to word_data
        word_data.append(text)
            #for i in word_data:
            #   text = text + i

            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
        if name == "sara":
            from_data.append(0)
        if name == "chris":
            from_data.append(1)


        email.close()

print "emails processed"
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )

print(len(word_data))
print(word_data[152])


### in Part 4, do TfIdf vectorization here
import numpy as np
#word_data = [[str(e)] for e in word_data]
print(type(word_data))
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words = "english")
#word_data = [3,2]
counts = vectorizer.fit_transform(word_data)
#counts = counts.toarray()
#print(len(np.unique(counts)))
pre_list = (vectorizer.get_feature_names())
set_list = set(pre_list)
unique_list = list(set_list)
print(len(unique_list))

