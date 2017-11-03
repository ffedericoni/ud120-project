#!/usr/bin/python
'''
@author: Fabrizio Federiconi
@email:  ffedericoni@gmail.com

This is the final project for the ud120 course
My comments start with #ff

'''

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib

print __doc__

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi'] 
features_list += ['bonus', 'salary', 'deferral_payments', 'total_payments', 
'loan_advances', 'restricted_stock_deferred',
'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 
'other', 'long_term_incentive',
'restricted_stock', 'director_fees']
features_list += ['to_messages',  'from_poi_to_this_person', 
'from_messages', 'from_this_person_to_poi',
'shared_receipt_with_poi']
#'email_address'


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
#ff as we saw in Lesson 8.16, 'TOTAL' is a spreadsheet quirk and 
#ff it should be removed as outlier
data_dict.pop('TOTAL')

#ff I want to analyse the items with most NaNs
print("=============================================")
print("= These items have a lot of NaNs" )
print("=============================================")
keys = data_dict.keys()
for key in keys:
    countNaN=0
    for feature in features_list:
        value = data_dict[key][feature]
        if value=="NaN":
                countNaN+=1
    if countNaN > len(features_list) * 0.75:
        print key, countNaN, data_dict[key]['poi']
#ff the loop above prints 12 items and visually checking them, I see one of them is a Travel Agency.
#ff It is in the insiderpay PDF, even though not related to an employee, so I keep it.
#ff googling I found out that the sister of the Enron CEO owns 50% of the Travel Agency, 
#ff so the company could be part of the fraud
#ff All these items with an abundance of NaNs have poi=False, so I keep them
#ff because it seems the lack of information correlates with the poi feature 
print("=============================================")
print("= There are %d records in the Data Dictionary" % len(data_dict))
print("= after removing the outlier")
pois=0
for key in keys:
    if data_dict[key]['poi']:
        pois+=1
print("= Only %d records are POIs, so the dataset is quite unbalanced" % pois)
print("=============================================")
#ff Let's see if some features have >75% NaNs
print("=============================================")
print("= These are the features with > 75% NaNs")
print("=============================================")
for feature in features_list:
    countNaN=0
    for key in keys:
        value = data_dict[key][feature]
        if value=="NaN":
                countNaN+=1
    if countNaN > len(keys) * 0.75:
        print feature, countNaN
#ff The loan_advances, restricted_stock_deferred and director_fees have a lot of NaNs
#ff and it makes sense because only a small percentage of people is a director and 
#ff loan advances or deferred/restricted stocks are not commonly granted.
#ff For this reason I keep the features because the usage of these kinds 
#ff of payment could correlate with being a poi

#ff Let's plot some graphs to check for other outliers
data = featureFormat(data_dict, ['bonus', 'salary'], sort_keys = True)
labels, features = targetFeatureSplit(data)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
#TODO uncomment matplotlib.pyplot.show()
#ff Ther are points with a high stddev, 
#ff but that can be an interesting information for the model
data = featureFormat(data_dict, ['other', 'long_term_incentive'], sort_keys = True)
labels, features = targetFeatureSplit(data)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("other")
matplotlib.pyplot.ylabel("long_term_incentive")
#TODO uncomment matplotlib.pyplot.show()
#ff Also for the other and long_term_incentive features
#ff There are a few points with a high stddev, 
#ff but again that can be an interesting information for the model


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#ff Adding two new features, as in Lesson 12
for name in my_dataset:
    employee = my_dataset[name]
    if (employee['from_poi_to_this_person'] != 'NaN' and
                 employee['from_this_person_to_poi'] != 'NaN' and
                 employee['to_messages'] != 'NaN' and
                 employee['from_messages'] != 'NaN'
                 ):
        fraction_from_poi = float(employee["from_poi_to_this_person"]) / \
            float(employee["from_messages"])
        employee["fraction_from_poi"] = fraction_from_poi
        fraction_to_poi = float(employee["from_this_person_to_poi"]) / \
            float(employee["to_messages"])
        employee["fraction_to_poi"] = fraction_to_poi
    else:
        employee["fraction_from_poi"] = employee["fraction_to_poi"] = 0

my_features = features_list + ["fraction_from_poi", "fraction_to_poi"]

#ff Extract again with the two additional features
data = featureFormat(my_dataset, my_features, sort_keys = True)
labels, features = targetFeatureSplit(data)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clfgnb = GaussianNB()
clfgnb.fit(features, labels)
print("Gaussian NB Score=", clfgnb.score(features, labels))

# Out of the bos the precision is good but recall is very low, so I think the problem
# is the balancing of the data, as the positives are only 18 out of 144
# Playing with class weights improves the score, but I had to play also with random_state
# to achieve scores higher than 0.3
from sklearn import tree
class_weights = {0.0: 1, 1.0: 3.5}
#class_weights = "balanced"
clf = tree.DecisionTreeClassifier(class_weight=class_weights, criterion='gini', max_depth=4,
            max_features="auto", max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=4, min_weight_fraction_leaf=0.0,
            presort=False, random_state=51, splitter='random')
clf.fit(features, labels)
print("Decision Tree Score=", clf.score(features, labels))


import tester
tester.test_classifier(clf, my_dataset, my_features)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
print "This is the End"

#The features in the data fall into three major types, namely financial features, email features and POI labels.
#
#financial features: ['salary', 'deferral_payments', 'total_payments', 
#'loan_advances', 'bonus', 'restricted_stock_deferred',
#'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 
#'other', 'long_term_incentive',
#'restricted_stock', 'director_fees'] (all units are in US dollars)
#email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 
#'from_messages', 'from_this_person_to_poi',
#'shared_receipt_with_poi'] (units are generally number of emails messages; 
#notable exception is email_address, which is a text string)
#
#POI label: [poi] (boolean, represented as integer)
#You are encouraged to make, transform or rescale new features from the starter features. 
#If you do this, you should store the new feature to my_dataset, and if you use the new feature in the final algorithm, 
#you should also add the feature name to my_feature_list, so your coach can access it during testing. 
#For a concrete example of a new feature that you could add to the dataset, refer to the lesson on Feature Selection.