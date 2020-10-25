----------------------------------------------------------
Predicting Survival of Childhood Cancers using TARGET data
----------------------------------------------------------

**Author: Deena Bleich**

*Class: BIOF509*

The Therapeutical Applicable Research to Generate Effective Treatments (TARGET) program studies childhood cancers
and has produced datasets containing demographic and clinical data of these patients, including vital status (alive, dead). 
This program explores whether that data can be used to create a model, using supervised learning, to predict vital status
given a set of demographic or clinical features.

There are two versions of this program:
- modelTARGET.py - run the program on the command line
- TARGETanalysis.ipynb - run the program in a Jupyter notebook

Prerequisites
*************

The following packages need to be installed, if you don't already have them:

- pandas 
- numpy 
- seaborn
- matplotlib.pyplot
- sklearn

In addition, for the Jupyter notebook version:

This is only needed for UMAP dimensionality reduction, which is only included in the Jupyter notebook.

- umap (Note: install umap-learn!)

These are only needed if you want for creation and display of the Decision Tree graph, which is only included in the Jupyter notebook.

- pydotplus # To create our Decision Tree Graph
- IPython.display import Image  # To Display a image of the graph

Python program -- to be run on command line
--------------------------------------------

Name: modelTARGET.py
*******************

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

Jupyter Notebook
-----------------

You'll need to have Jupyter installed. If you don't have it yet, you can install Anaconda, which includes Jupyter notebook.
Go to https://www.anaconda.com/products/individual.

Name: TARGETanalysis.ipynb
**************************

1) The first 11 cells are the program and its functions. Run each of these cells.

2) The next cell runs the the program, calling function modelTARGETdata with parameters. See the parameter list above. It has the same functionality as the command line program, modelTARGET.py.

3) The next cells after that run through each function, one at a time, allowing you to see output from each step. 
    a) If you follow them in sequence, you will be able to run the whole program.
    b) It includes additional analysis steps, not included in the main program:
    
            - UMAP
            - feaure importance graphs
            - decision tree graphs

