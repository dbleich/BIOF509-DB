----------------------------------------------------------
Predicting Survival of Childhood Cancers using TARGET data
----------------------------------------------------------

**Author: Deena Bleich**

*Class: BIOF509*

The Therapeutical Applicable Research to Generate Effective Treatments (TARGET) program studies childhood cancers
and has produced datasets containing demographic and clinical data of these patients, including vital status (alive, dead). 
This pprogram explores whether that data can be used to create a model, using supervised learning, to predict vital status
given a set of demographic or clinical features.

Python program -- to be run on command line
--------------------------------------------
Name
****

modelTARGET.py

Parameters
**********

- Parameter 1: TARGET Demographic and Clinical Data File. USe the file in this directory: 'TARGET-Clinical-All.csv'
- Parameter 2: Feature Set

               *  "demog" = "disease_code", "age_at_diagnosis", "gender", "race", "ethnicity", "vital_status"
               *  "clin" = "disease_code", "age_at_diagnosis", "INSS_stage", "first_event", "histology", "MYCN_status", "vital_status"
- Parameter 3: Model Type

               *  "RF" = Random Forest
               *  "DT" = Decision Tree
               
Example call
************

Copy the program and the data file to the same directory.

At the command prompt, if you change to that directory, type the following:
```
python modelTARGET.py "TARGET-Clinical-All.csv" "clin" "RF"
```

Prerequites
***********

The following packages need to be installed, if you don't already have them:
- pandas 
- numpy 
- seaborn
- matplotlib.pyplot
- umap (Note: install umap-learn!)
- sklearn
- pydotplus # To create our Decision Tree Graph
- IPython.display import Image  # To Display a image of our graph
