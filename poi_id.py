#!/usr/bin/python
import sys
import pickle
sys.path.append("../tools/")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from fancyimpute import KNN
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'exercised_stock_options', 'expenses', 'from_messages', 
               'from_poi_to_this_person', 'from_this_person_to_poi', 'other', 
               'restricted_stock', 'salary', 'shared_receipt_with_poi', 
               'to_messages', 'total_payments', 'total_stock_value' 
               ]

#Done later, after exporting to a dataframe and performing analysis. 
#Refer to modified_poi_id.py
#for more details. 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

data_dict.pop('TOTAL',0) 
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)     

enron = pd.DataFrame.from_records(list(data_dict.values()))
employees = pd.Series(list(data_dict.keys())) 
enron.set_index(employees, inplace=True) 

### Task 3: Create new feature(s)
enron.replace(to_replace='NaN', value=np.nan, inplace=True)

dropfeatures = ['loan_advances', 'director_fees', 'restricted_stock_deferred',
               'deferral_payments', 'deferred_income', 'long_term_incentive', 'bonus']

enron.drop(dropfeatures, axis=1, inplace=True)
enron.drop([u'email_address'] ,axis=1, inplace=True)
all_nans = enron[['from_poi_to_this_person','from_this_person_to_poi',
                  'from_messages','to_messages','shared_receipt_with_poi']].isnull().all(1)

enron.loc[all_nans, ['from_poi_to_this_person','from_this_person_to_poi',
                  'from_messages','to_messages','shared_receipt_with_poi']] = 0 


enron['ratio_from_poi'] = enron['from_poi_to_this_person']/enron['to_messages']
enron['ratio_from_poi'].fillna(0, inplace=True)



enron['ratio_to_poi'] = enron['from_this_person_to_poi']/enron['from_messages']
enron['ratio_to_poi'].fillna(0, inplace=True)
##Look at Enron.pdf for a clearer understanding of the two newly created features. 

enron_copy = pd.DataFrame(enron[['exercised_stock_options', 'expenses', 'from_messages', 
                                 'from_poi_to_this_person', 'from_this_person_to_poi', 
                                 'other', 'poi', 'restricted_stock', 'salary', 
                                 'shared_receipt_with_poi', 'to_messages', 
                                 'total_payments', 'total_stock_value', 
                                 'ratio_from_poi', 'ratio_to_poi']].copy())

col = [['exercised_stock_options', 'expenses', 'from_messages', 
        'from_poi_to_this_person', 'from_this_person_to_poi', 
        'other', 'poi', 'restricted_stock', 
        'salary', 'shared_receipt_with_poi', 'to_messages', 
        'total_payments', 'total_stock_value', 
        'ratio_from_poi', 'ratio_to_poi']]


enron = pd.DataFrame(KNN(k=2).complete(enron_copy))
##This is a built in function that imputes the missing values based on nearest neigbors. 

enron.columns = col
enron.index = enron_copy.index


### Store to my_dataset for easy export below.

enronml = pd.DataFrame(enron[['poi', 'exercised_stock_options', 'expenses', 'from_messages', 
               'from_poi_to_this_person', 'from_this_person_to_poi', 'other', 
               'restricted_stock', 'salary', 'shared_receipt_with_poi', 
               'to_messages', 'total_payments', 'total_stock_value' 
               ]].copy())




enronml = enronml.to_dict(orient="index")
my_dataset = enronml


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html



### Check the other classifiers tried in A,B,C,D,E,F,G and H files
### This file is meant to contain only the final working classifier that satisfies
### the Precision, Recall score requirement. 

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.naive_bayes import GaussianNB

pca = PCA()
gnbc = GaussianNB()
steps = [('scaler', MinMaxScaler()),
         ('best', SelectKBest()),
         ('pca', pca),
         ('gnbc', gnbc)]

pipeline = Pipeline(steps)

parameters = [    
{
'best__k':[3],
'pca__n_components': [1,2]
},
{
'best__k':[4],
'pca__n_components': [1,2,3]
},
{
'best__k':[5],
'pca__n_components': [1,2,3,4]
},
{
'best__k':[6],
'pca__n_components': [1,2,3,4,5]
},
{
'best__k':[7],
'pca__n_components': [1,2,3,4,5,6]
},
{
'best__k':[8],
'pca__n_components': [1,2,3,4,5,6,7]
},
{
'best__k':[9],
'pca__n_components': [1,2,3,4,5,6,7,8]
},
{
'best__k':[10],
'pca__n_components': [1,2,3,4,5,6,7,8,9]
},
{
'best__k':[11],
'pca__n_components': [1,2,3,4,5,6,7,8,9,10]
},
{
'best__k':[12],
'pca__n_components': [1,2,3,4,5,6,7,8,9,10,11]
}
]

cv = StratifiedShuffleSplit(test_size=0.2, random_state=42)
gnbforc = GridSearchCV(pipeline, param_grid = parameters, cv=cv, scoring="f1")
gnbforc.fit(features, labels)

means = gnbforc.cv_results_['mean_test_score']
stds = gnbforc.cv_results_['std_test_score']


for mean, std, params in zip(means, stds, gnbforc.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

print "\n"
print(gnbforc.best_estimator_ )
print "\n"
print(gnbforc.best_score_)
print "\n"
print(gnbforc.best_params_)

feature_step = gnbforc.best_estimator_.named_steps['best']

# Get SelectKBest scores, rounded to 2 decimal places, name them "feature_scores"
feature_scores = ['%.2f' % elem for elem in feature_step.scores_ ]
# Get SelectKBest pvalues, rounded to 3 decimal places, name them "feature_scores_pvalues"
feature_scores_pvalues = ['%.3f' % elem for elem in  feature_step.pvalues_ ]
# Get SelectKBest feature names, whose indices are stored in 'skb_step.get_support',
# create a tuple of feature names, scores and pvalues, name it "features_selected_tuple"
features_selected_tuple=[(features_list[i+1], feature_scores[i], feature_scores_pvalues[i]) for i in feature_step.get_support(indices=True)]

# Sort the tuple by score, in reverse order
features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)

# Print
print ' '
print 'Selected Features, Scores, P-Values'
print features_selected_tuple

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


dump_classifier_and_data(gnbforc.best_estimator_, my_dataset, features_list)