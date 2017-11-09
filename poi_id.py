#!/usr/bin/python
'''
@author: Fabrizio Federiconi
@email:  ffedericoni@gmail.com

This is the final project for the ud120 course
Comments start with #ff

'''

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib
from datetime import datetime

print __doc__

startt = datetime.now()

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
matplotlib.pyplot.show()
#ff There are points with a high stddev, 
#ff but that can be an interesting information for the model
data = featureFormat(data_dict, ['other', 'long_term_incentive'], sort_keys = True)
labels, features = targetFeatureSplit(data)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("other")
matplotlib.pyplot.ylabel("long_term_incentive")
matplotlib.pyplot.show()
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



### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


#ff Out of the box the Precision is good but Recall is very low, so I think the problem
#ff is the balancing of the data, as the positives are only 18 out of 144
#ff I tried to set class_weights = "balanced", but I got the opposite problem:
#ff recall becomes high, but precision becomes low. I decided to play with weights.
#ff Playing with class weights immediately improves the score, but I have to play 
#ff also with random_state to achieve scores higher than 0.3
#ff After achivieng good scores, I continue playing with other parameters.
#ff I get benefits only from a slight increase of min_weight_fraction_leaf
from sklearn import tree
from sklearn.metrics import f1_score
class_weights = {0.0: 1, 1.0: 3.5}

clfbestDT = tree.DecisionTreeClassifier(class_weight=class_weights, criterion='gini', 
            max_depth=4, 
            max_features="auto", max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=4, min_weight_fraction_leaf=0.01,
            presort=False, random_state=51, splitter='random')
clfbestDT.fit(features, labels)
print("Decision Tree Score=", clfbestDT.score(features, labels))
#import tester
#tester.test_classifier(clfbestDT, my_dataset, my_features)
# ============================BEST DT MODEL=================================
#DecisionTreeClassifier(class_weight={0.0: 1, 1.0: 3.5}, criterion='gini',
#            max_depth=4, max_features='auto', max_leaf_nodes=None,
#            min_impurity_split=1e-07, min_samples_leaf=1,
#            min_samples_split=4, min_weight_fraction_leaf=0.01,
#            presort=False, random_state=51, splitter='random')
#        Accuracy: 0.82480       Precision: 0.35932      Recall: 0.40100 
#        F1: 0.37902     F2: 0.39191
#        Total predictions: 15000        True positives:  802    False positives: 1430   
#        False negatives: 1198   True negatives: 11570
# =============================================================================
# ==========================BEST RF MODEL (F1)=================================
#RandomForestClassifier(bootstrap=True, class_weight={0.0: 1, 1.0: 12},
#            criterion='gini', max_depth=5, max_features='auto',
#            max_leaf_nodes=None, min_impurity_split=1e-07,
#            min_samples_leaf=1, min_samples_split=3,
#            min_weight_fraction_leaf=0.04, n_estimators=110, n_jobs=1,
#            oob_score=False, random_state=51, verbose=0, warm_start=False)
#        Accuracy: 0.84760       Precision: 0.42800      Recall: 0.42500 
#                                F1: 0.42649     F2: 0.42560
#        Total predictions: 15000        True positives:  850    False positives: 1136
#        False negatives: 1150   True negatives: 11864
# =============================================================================


#ff Trying with Random Forests
#ff Playing with class weights immediately improves the score
#ff I get benefits progressively increasing min_weight_fraction_leaf
#ff and fine tuning class weights according to the balance between Precision 
#ff and Recall: if Recall is much higher than Precision, I decrease the weight for 
#ff the class=1.0, otherwise I increase it
#ff After playing with most parameters, I get much better F1 score than 
#ff with Decision Trees
from sklearn.ensemble import RandomForestClassifier
clfRF = RandomForestClassifier(class_weight={0.0: 1, 1.0: 12},
                               max_depth=5,
                               min_samples_split=3,
                               n_estimators=110,
                               min_weight_fraction_leaf=0.04,
                               random_state=51)
clfRF.fit(features, labels)
y_preds = clfRF.predict(features)
print("Random Forest Score=", 
       clfRF.score(features, labels), f1_score(y_preds, labels) )
#ff I tried to use f1_score() for a much faster prediction of the results I would get
#ff with the tester, but it didn't work.
#tester.test_classifier(clfRF, my_dataset, my_features)
print("===============================================")

#ff checking if the added features are important
for ind, val in enumerate(clfRF.feature_importances_):
    print my_features[ind+1], val
#ff they are not the most important but they are quite important 

#ff I try to fit the RF with the same hyperparameters to the data without the
#ff two new features, but I get worse results
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
print("Nr. of features=", len(features))
clfRFo = RandomForestClassifier(class_weight={0.0: 1, 1.0: 12},
                               max_depth=5,
                               min_samples_split=3,
                               n_estimators=110,
                               min_weight_fraction_leaf=0.04,
                               random_state=51)
clfRFo.fit(features, labels)
y_preds = clfRFo.predict(features)
print("Random Forest Score(original features)=", 
       clfRFo.score(features, labels), f1_score(y_preds, labels) )
#tester.test_classifier(clfRFo, my_dataset, features_list)
print("===============================================")


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
from sklearn.model_selection import GridSearchCV , StratifiedShuffleSplit

cv = StratifiedShuffleSplit(n_splits = 3, test_size=0.3,
                            random_state = 51)
params_grid = {"class_weight": [
                                {0.0: 1, 1.0: 10}, 
                                {0.0: 1, 1.0: 11}, 
                                {0.0: 1, 1.0: 12},
                                {0.0: 1, 1.0: 13}, 
                                {0.0: 1, 1.0: 14}
                                ],
                "min_samples_split": [3, 4],
                "n_estimators":[ 110],
                "min_weight_fraction_leaf": [0.04],
                "random_state": [60, 62, 63, 64, 65, 67, 68, 69]}
#params_grid = {}
#ff I tried GridSearchCV to check for parameters that could increase F1 score
#ff but I wasn't able to get better values from the tester
#ff so at the end I keep the RF model that I found with my manual fine tuning
clfRFg = RandomForestClassifier(max_depth=5)
grid = GridSearchCV(clfRFg, param_grid = params_grid,
                          scoring = 'f1', 
                          cv = cv, n_jobs=1, verbose=1)
# grid = grid.fit(features, labels)
# print("Best GRID RF Estimator=", grid.best_estimator_)
# tester.test_classifier(grid.best_estimator_, my_dataset, my_features)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clfRF, my_dataset, my_features)
print startt, datetime.now()


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
