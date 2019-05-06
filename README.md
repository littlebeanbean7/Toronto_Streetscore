# Toronto Streetscore: Predicting Perceived Street Safety Using CNN

## Overview
This project aims to build an explainable machine learning model that could predict people's perceived safety towards streets views.
We apply deep Convolutional Neural Network (CNN) techniques on street view images fetched using Google Street View Static API.


## code/    

### Data Exploration:
* PlacePulse1.0_explore.ipynb   
Explores the data structure of the original data - PlacePulse1.0 used by the Streetscore project. 

* Streetscore_create_target_label.ipynb  
Generates a binary 'safety' label.
 
### Data Preparation:
* Boston_Stratified_Sampling.ipynb  
Stratified sampling 20,000 samples from the Boston data predicted by the Streetscore project considering two factors at the same time: 1). Portion of zone classes of Toronto and 2). Portion of Target “safety” variable.

* boston_split_train_test.ipynb  
Splits the downsampled 20,000 boston samples into 80 : 20 as training and test sets, so as to use the trainings and test sets to fetch images.

* fetchimage.py  
Fetches images using Google Street View Static API

*  cropimage.py  
Crops out the Google Logo on each image

*  check_fetched_images_and_merge_with_target.ipynb  
Checks how many images were fetched and merges the fetched images with target variable. You should use this script if you are saving all Boston images into one folder. 

*  merge_fetched_train_test_image_with_target.ipynb  
Merges the fetched train / test images with target variable. You should use this script if you are saving Boston images into two  (train and test) seperate folders. 

*  Toronto_sample.ipynb  
Subsampled 2100 Toronto geolocations, with which fetched 2034 Toronto images. Those images would be used to deploy our final model for the predictions of Perceived Toronto Street View. 

### Model Building:

##### Transfer Learning:   
*  ResNeXt50ClassifierTemplate.py  
Generates a ResNeXt50 neural network with pretrained weights from Keras.

*  Classifier.py  
Trains the models using the pretrained weights, conducts 5-fold cross-validation, and save the 5 best models produced during cross-validation and 1 best model produced during the training on the full training data. 

*  ResNeXt_prediction.py  
Generates predictions by each of model on the test set. 

##### Building our own CNN: 
*  CNN_Tested_Explained_with_Toronto_images.ipynb  
Builds a CNN model by ourselves and explains it using LIME.

### Ensemble:
*  ensemble_boston_test.ipynb 
Conducts ensemble learning methods including: Averaging ensemble; Conditional ensemble; and Weighted ensemble; Calculates each model's confusion matrix and performance;  Finds one final model with relatively best prediction of both safety = 0 and safety =1.

*  ensemble_toronto.ipynb   
Applies the best ensemble strategy found in ensemble_boston_test.ipynb on Toronto images.
 
### Model explanation:
*  Model interpretation generated using Lime package.ipynb   
Interprets model predictions using the LIME package.

## data/  

### Boston Data  

##### Data produced by target column generation:   
*  Boston_ny_safety.csv   
Boston and New York City data with a newly created target column - safety.

*  Boston_safety.csv   
Boston data with a newly created target column - safety.

##### Data produced by stratified sampling:   
*  boston_safety_subsample.csv     
20000 downsampled Boston data samples, achieved by stratified sampling based on both the safety and the subdistrict columns.

##### Data produced by Train / Test split:   
*  boston_train_list.csv    
16000 Boston Train data samples (80% of the 20000) used to fetch Boston Train images.

*  boston_test_list.csv	    
4000 Boston Test data samples (20% of the 20000) used to fetch Boston Test images.

##### Data produced by image fetching:  
*  boston_train_fetched_with_target.csv	   
Boston Train images fetched using Google Street View Static API, merged with target column (safety).

*  boston_test_fetched_with_target.csv	   
Boston Test images fetched using Google Street View Static API, merged with target column (safety).

##### Data produced by ensembling:   
*  final_pred_boston_test_weighted_ensemble2.csv	   
Final ensembled predictions on Boston Test images. 

### Toronto Data

##### Data produced by downsampling:   
*  Toronto2100_sample.csv    
2100 downsampled Toronto data samples used to fetch Toronto street view images.

##### Data produced by ensembling:   
*  final_pred_toronto.csv	   
Final ensembled predictions on the Toronto street view images. 


