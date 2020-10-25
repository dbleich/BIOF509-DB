import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree #For Decision Tree
#import pydotplus # To create our Decision Tree Graph
#from IPython.display import Image  # To Display a image of our graph

#%matplotlib inline

'''
Author: Deena Bleich
Class: BIOF509
The Therapeutical Applicable Research to Generate Effective Treatments (TARGET) program studies childhood cancers
and has produced datasets containing demographic and clinical data of these patients, including vital status (alive, dead). 
This pprogram explores whether that data can be used to create a model, using supervised learning, to predict vital status
given a set of demographic or clinical features.

'''

class project:
    # project loads, manipulates, and analyzes a given dataset
    def __init__(self, csv):
        ''' the __init__ function is automatically called when you create
        an instance of the class. Variables or attributes of a class are
        referred to using self.__ and are accessible by all the methods, and
        by the user outside of the class definition'''
   
        ''' Read in the TARGET Study data
        The Therapeutically Applicable Research to Generate Effective Treatments (TARGET) program studies childhood cancers. 
        This data was stored in the Google BigQuery project isb-cgc.TARGET_bioclin_v0.clinical_v1'''

        try:
          df = pd.read_csv(csv, low_memory=False)
          self.data = df
  
        except:
          print("The file {0} was not found or could not be read".format(csv))

'''Function to select our features'''
def select_features(df_clin, feature_list):
    new_df = df_clin[feature_list]
        
    return new_df

'''Split dataframe into features and labels'''
def create_features_and_labels(df):
    
    #drop nan frm vital_status, as vital_status will be the label
    new_df = df.dropna(subset=['vital_status'])
    new_df = new_df.reset_index(drop = True)
    
    #separate out the features from the labels.
    
    #Get just the vital_statuses, which will be the labels
    vital_statuses = new_df["vital_status"].to_list()
    #turn the vital status to binary values 0 and 1; alive = 0, dead = 1
    vital_status_codes, vital_status_factors = pd.factorize(vital_statuses)
   
    #Get just the features
    features_df = new_df.drop(columns=["vital_status"])
    
    return features_df, vital_status_codes

'''Function to encode features - feature set "demog" '''
def encode_categorical_features1(features_df):

    #hot encode each categorical feature
    df_disease = pd.get_dummies(features_df['disease_code'], prefix='disease')
    df_gender = pd.get_dummies(features_df['gender'], prefix='gender')
    df_race = pd.get_dummies(features_df['race'], prefix='race')
    df_ethnicity = pd.get_dummies(features_df['ethnicity'], prefix='ethnicity')

    #Add these new features to the features dataframe and drop the columns that they were derived from
    encoded_feature_df = pd.concat([features_df, df_disease, df_gender, df_race, df_ethnicity], axis =1)
    encoded_feature_df = encoded_feature_df.drop(columns=['disease_code', 'gender', 'race', 'ethnicity'])    
    
    return encoded_feature_df


'''Function to encode features - feature set "clin" '''
def encode_categorical_features2(features_df):

    #hot encode each categorical feature
    df_disease = pd.get_dummies(features_df['disease_code'], prefix='disease')
    df_INSS_stage = pd.get_dummies(features_df['INSS_stage'], prefix='INSS_stage')
    df_first_event = pd.get_dummies(features_df['first_event'], prefix='first_event')
    df_histology = pd.get_dummies(features_df['histology'], prefix='histology')
    df_MYCN = pd.get_dummies(features_df['MYCN_status'], prefix='MYCN')

    #Add these new features to the features dataframe and drop the columns that they were derived from
    encoded_feature_df = pd.concat([features_df, df_disease, df_INSS_stage, df_first_event, df_histology, df_MYCN], axis =1)
    encoded_feature_df = encoded_feature_df.drop(columns=['disease_code', 'INSS_stage', 'first_event','histology', 'MYCN_status'])
    
    return encoded_feature_df

'''Scale features'''
def scale_features(features_df):

    data_for_scaling = features_df.values
    scaler = MinMaxScaler()
    scaled_feature_data = scaler.fit_transform(data_for_scaling)
    
    return scaled_feature_data


'''Random Forest Classifer'''
def randomForest(num_of_feature_cols, feature_data, vital_status_codes):

    rf_model = RandomForestClassifier(n_estimators = 64)

    #split our dataset into 5 parts, and each part will get a turn being the test dataset
    skf = StratifiedKFold(n_splits=5)

    accuracy = 0

    #Each test set will have slightly different feature importances so sum the results for each feature
    all_feature_importances = np.zeros(num_of_feature_cols)

    #keep track of the labels from the random test sets in order to make a confusion matrix
    all_labels = []

    #keep track of the predictions from the random test sets in order to make a confusion matrix '''
    all_predictions = []

    for train_index, test_index in skf.split(feature_data, vital_status_codes):
    
        #training dataset
        X_train, y_train = feature_data[train_index], vital_status_codes[train_index]

        #testing dataset
        X_test, y_test = feature_data[test_index], vital_status_codes[test_index]

        rf_model.fit(X_train,y_train)
        
        #add the new feature importances to the current FI vlues
        all_feature_importances += rf_model.feature_importances_
        
        predictions = rf_model.predict(X_test)
        
        #add the new predictions onto the list of predictions
        all_predictions.extend(predictions)
        
        #add the new predictions onto the list of predictions
        all_labels.extend(y_test)
        
        accuracy += accuracy_score(predictions,y_test)
        
        return accuracy, all_predictions, all_labels, all_feature_importances
        
'''Decision Tree Classifer'''
def decisionTree(num_of_feature_cols, feature_data, vital_status_codes):

    clf_model = tree.DecisionTreeClassifier()

    #split our dataset into 5 parts, and each part will get a turn being the test dataset
    skf = StratifiedKFold(n_splits=5)

    accuracy = 0

    #Each test set will have slightly different feature importances so sum the results for each feature
    all_feature_importances = np.zeros(num_of_feature_cols)

    #keep track of the labels from the random test sets in order to make a confusion matrix
    all_labels = []

    #keep track of the predictions from the random test sets in order to make a confusion matrix '''
    all_predictions = []

    for train_index, test_index in skf.split(feature_data, vital_status_codes):
    
        #training dataset
        X_train, y_train = feature_data[train_index], vital_status_codes[train_index]

        #testing dataset
        X_test, y_test = feature_data[test_index], vital_status_codes[test_index]

        clf_model.fit(X_train,y_train)
        
        #add the new feature importances to the current FI vlues
        all_feature_importances += clf_model.feature_importances_
        
        predictions = clf_model.predict(X_test)
        
        #add the new predictions onto the list of predictions
        all_predictions.extend(predictions)
        
        #add the new predictions onto the list of predictions
        all_labels.extend(y_test)
        
        accuracy += accuracy_score(predictions,y_test)
        
        return accuracy, all_predictions, all_labels, clf_model
        

'''UMAP Dimensionality Reduction'''
def UMAPreduce(feature_data):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(scaled_feature_data)
    
    return embedding

'''
Parameter 1: TARGET Demographic and Clinical Data File
Parameter 2: Feature Set
                "demog" = "disease_code", "age_at_diagnosis", "gender", "race", "ethnicity", "vital_status"
                "clin" = "disease_code", "age_at_diagnosis", "INSS_stage", "first_event", "histology", "MYCN_status", "vital_status"
Parameter 3: Model Type
                "RF" = Random Forest
                "DT" = Decision Tree
'''
import sys
def modelTARGET(csv, feature_set, model_type):
   
    df_clinical = project(csv)
    df_clin = df_clinical.data 

    #Feature Set 1 is mainly demographic; Feature Set 2 is mainly clinical
    if feature_set == "demog":
        feature_list = ["disease_code", "age_at_diagnosis", "gender", "race", "ethnicity", "vital_status"]
    else:
        feature_list = ["disease_code", "age_at_diagnosis", "INSS_stage", "first_event", "histology", "MYCN_status", "vital_status"]
        
    #create new dataframe that only has the features which we are interested in
    new_df = select_features(df_clin, feature_list)
    
    #Separate the features from the labels (vital_status of alive, dead)
    features_df, vital_status_codes = create_features_and_labels(new_df)
    
    ##Hot Encode Categorical Data
    if feature_set == "demog":
       encoded_features_df = encode_categorical_features1(features_df)
    else:
       encoded_features_df = encode_categorical_features2(features_df)
    
    ##Scale features -- need to scale Age at Diagnosis
    scaled_feature_data = scale_features(encoded_features_df)
    
    ##Run the Supervised Learning Model of choice on the features of choice and print out accuracy and the confusion matrix
    
    if model_type == 'RF':
        #Perform Random Forest Classification
        accuracy, all_predictions, all_labels, all_feature_importances = randomForest(len(encoded_features_df.columns), scaled_feature_data, vital_status_codes)
        print()
        print('Success! Random Forest Classification has completed, with Accuracy:')
        print(accuracy)
        print()
        print('The Confusion Matrix shows the accuracy per label (Alive in top left, Dead in bottom right).')
        #Create a confusion matrix of all the labels against all the predictions
        cm = confusion_matrix(all_labels,all_predictions,normalize= 'true')
        #turn it into a dataframe so seaborn will label the graph using the columns/indicies of the dataframe
        df_cm = pd.DataFrame(cm,index=[0,1],columns=[0,1])
        #graph using seaborn heatmap function
        sns.heatmap(df_cm, annot=True)
        plt.show()
    else:
        #Perform Decision Tree Classification
        accuracy, all_predictions, all_labels, clf_model = decisionTree(len(encoded_features_df.columns), scaled_feature_data, vital_status_codes)
        print()
        print('Success! Decision Tree Classification has completed, with Accuracy:')
        print(accuracy)
        print()
        print('The Confusion Matrix shows the accuracy per label (Alive in top left, Dead in bottom right).')
        #Create a confusion matrix of all the labels against all the predictions
        cm = confusion_matrix(all_labels,all_predictions,normalize= 'true')
        #turn it into a dataframe so seaborn will label the graph using the columns/indicies of the dataframe
        df_cm = pd.DataFrame(cm,index=[0,1],columns=[0,1])
        #graph using seaborn heatmap function
        sns.heatmap(df_cm, annot=True)
        plt.show()
    
if __name__== "__main__":
    modelTARGET(sys.argv[1], sys.argv[2], sys.argv[3])