
# coding: utf-8

# In[11]:

#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
for key in data_dict.keys():
    for value in data_dict[key]:
        print value
    break


# In[12]:

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi']
for key in data_dict.keys():
    for value in data_dict[key]:
        if value in features_list:
            continue
        features_list.append(value)
    break

import pprint
pprint.pprint(features_list)
# You will need to use more features


# In[13]:

features_list.remove('email_address')
pprint.pprint(features_list)


# In[14]:

for feature in features_list:
    cnt=0
    for key in data_dict.keys():
        if data_dict[key][feature] == 'NaN':
            cnt+=1
    print feature + " -> " + str(cnt)


# In[15]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt

for feature in features_list:
    maxi = 0
    cnt=0
    for key in data_dict:
        cnt+=1
        point=data_dict[key][feature]
        if point>maxi and point != 'NaN':
            maxi = point
            name = key
        plt.scatter(point, cnt)
    plt.xlabel(feature)
    plt.show()
    print name
    print maxi
    print "\n ------------------------------------------------------------------------------------ \n"


# In[16]:

data_dict.pop('TOTAL')


# In[17]:

for feature in features_list:
    maxi = 0
    cnt=0
    for key in data_dict:
        cnt+=1
        point=data_dict[key][feature]
        if point>maxi and point != 'NaN':
            maxi = point
            name = key
        plt.scatter(point, cnt)
    plt.xlabel(feature)
    plt.show()
    print name
    print maxi
    print "\n ------------------------------------------------------------------------------------ \n"


# In[18]:

for key in data_dict.keys():
    print key


# In[19]:

data_dict.pop('THE TRAVEL AGENCY IN THE PARK')


# In[20]:

features_list.remove('restricted_stock_deferred')
features_list.remove('director_fees')
features_list.remove('loan_advances')


# In[21]:

for key in data_dict.keys():
    try:
        data_dict[key]['fraction_from_this_person_to_poi'] = float(data_dict[key]['from_this_person_to_poi']
                                                              )/data_dict[key]['from_messages']
    except:
        data_dict[key]['fraction_from_this_person_to_poi'] = 'NaN'
        
    try:
        data_dict[key]['fraction_from_poi_to_this_person'] = float(data_dict[key]['from_poi_to_this_person']
                                                              )/data_dict[key]['to_messages']
    except:
        data_dict[key]['fraction_from_poi_to_this_person'] = 'NaN'


# In[22]:

features_list.append('fraction_from_this_person_to_poi')
features_list.append('fraction_from_poi_to_this_person')
features_list.remove('from_this_person_to_poi')
features_list.remove('from_poi_to_this_person')
features_list.remove('from_messages')
features_list.remove('to_messages')


# In[23]:

pprint.pprint(features_list)


# In[24]:

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[25]:

print len(features)
print len(labels)


# In[26]:

print len(features[0])
print len(features[142])
print labels[0]


# In[27]:

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)


# In[28]:

from sklearn.feature_selection import SelectKBest
selection = SelectKBest(k=10)
features = selection.fit_transform(features, labels)
features_selected = selection.get_support(indices = True)
print selection.scores_


# In[29]:

new_flist = ['poi']

for index in features_selected:
    new_flist.append(features_list[index + 1])
    
features_list = new_flist
print features_list


# In[30]:

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)


# In[31]:

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

#clf = RandomForestClassifier(n_estimators=5, max_depth=10)
#clf = AdaBoostClassifier(algorithm='SAMME', n_estimators=5)
clf = DecisionTreeClassifier(criterion='entropy', max_depth = 2)
#clf = GaussianNB()
#clf = SVC(kernel='rbf', C=10)

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)


# In[32]:

from sklearn import metrics
print metrics.recall_score(labels_test, pred)
print metrics.accuracy_score(pred, labels_test)
print metrics.precision_score(labels_test, pred)


# In[33]:

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


# In[34]:

get_ipython().magic(u'run tester.py')

