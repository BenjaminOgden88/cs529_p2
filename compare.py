#call this file when you want to test the sklearn libraries on the original dataset and PCA dataset created.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import time

# totoal predicitons correct/ total samples in validation data
def accuracy_using_validation(validation_data_frame:pd.DataFrame,features_list,target,ml):
    valid_x = validation_data_frame.loc[:, features_list].values
    valid_y = validation_data_frame.loc[:,[target]].values

    total = len(valid_y)
    #total_correct hold all correct predicvitons from validation
    total_correct = 0
    for i in range(len(validation_data_frame.index)):
        current_test_element = validation_data_frame.iloc[i].to_list()
        #remove last element as that is going to be the target
        current_test_element.pop()
        
        #predictions must be in form [[input1,input2,...]]
        current_test_element = [current_test_element]
        prediction = ml.predict(current_test_element)
        if prediction[0] == valid_y[i]:
            total_correct = total_correct + 1
    
    return total_correct/total



def compare(training_data_frame:pd.DataFrame,
            validation_data_frame:pd.DataFrame,
            testing_data_frame:pd.DataFrame, target : str,list_ignored_features : list.__str__):
    
    #list_ignored_features is to remove id and class from dataframes so they are not placed into each model.
    for i in list_ignored_features:
        #axis 0 rows, axis 1 is col
        del training_data_frame[i]
        del validation_data_frame[i]
        del testing_data_frame[i]
    features = training_data_frame.columns.to_list()
    if target != None: 
        features.remove(target)
    print(features)
    x = training_data_frame.loc[:, features].values
    y = training_data_frame.loc[:,[target]].values
    
    #need to flatten y, right now is [["class"],["class"],...] needs to be ["class","class",...]
    flat_list_y = []
    for row in y:
        flat_list_y.extend(row)
    

    #####################################################
    #random forests
    clf = RandomForestClassifier(max_depth=50, random_state=0,n_estimators=300)
    start = time.time()
    clf.fit(x,flat_list_y)

    #uses validaiton data
    
    print("accuracy random forest", accuracy_using_validation(validation_data_frame,features,target,clf))
    end = time.time()
    print("time for random forest to complete ", end - start)

    #Gaussian Naive Bayes

    start = time.time()
    gauss_naive_bayes = GaussianNB()
    gauss_naive_bayes.fit(x,flat_list_y)
    print("accuracy gauss_naive_bayes", accuracy_using_validation(validation_data_frame,features,target,gauss_naive_bayes))
    end = time.time()
    print("time for gauss_naive_bayes to complete ", end - start)
    #Gradient boosting Machines
    start = time.time()
    grad_boost_machine = GradientBoostingClassifier(n_estimators=100,
                                                     learning_rate =1.0,
                                                     max_depth = 5,
                                                     random_state=0)

    grad_boost_machine.fit(x,flat_list_y)
    print("grad score ",grad_boost_machine.score(x,flat_list_y))
    
    print("accuracy grad_boost_machine", accuracy_using_validation(validation_data_frame,features,target,grad_boost_machine))
    end = time.time()
    print("time for grad_boost_machine to complete ", end - start)
    

    #Support Vectors Machines
    start = time.time()
    support_vector_machines = make_pipeline(StandardScaler(), svm.SVC(gamma='auto',kernel = 'linear'))
    support_vector_machines.fit(x,flat_list_y)
    print("accuracy support_vector_machines", accuracy_using_validation(validation_data_frame,features,target,support_vector_machines))
    end = time.time()
    print("time for support_vector_machines to complete ", end - start)

target_att = "class"
all_training_data = pd.read_csv("data/training_data.csv")
validation_data = pd.read_csv("data/validation_data.csv")
testing_data_frame = pd.read_csv("data/testing_data.csv")
ignored_features = ['id']
print("comparing with original data")
compare(all_training_data,validation_data,testing_data_frame,target_att,list_ignored_features=ignored_features)

all_training_data = pd.read_csv("data/training_data_pca.csv")
validation_data = pd.read_csv("data/validation_data_pca.csv")
testing_data_frame = pd.read_csv("data/testing_data_pca.csv")
ignored_features = ['id']
print("comparing with PCA data")

compare(all_training_data,validation_data,testing_data_frame,target_att,list_ignored_features=ignored_features)

