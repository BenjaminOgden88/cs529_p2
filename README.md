# CS_529_Project_2

## How To Run the Code
Run the main.py file to create a submission file.
The hyperparameters are in the top section of main.py:

- create_dataframes allows you to turn on or off (by setting to True or False) the creation of new dataframes from the audio files if you want to reuse data you've already made.
- validation_on must be turned on (set to True) during the creation of the data if you want to separate the validation data from the training data while creating the data files.  Otherwise ALL the training data will be put in the training_data.csv file and used to train the model.  This parameter also turns on or off the validation function that prints balanced accuracy for the model.
- tsne turns on or off the tsne analysis visualization of the data.
    - NOTE: REFER TO pca.py lines 73 and 74 on how to show tsne visuals based on original dataset or on PCA applied data
- run_logistic_reg_with_pca_data decides whether to train the model on the original data or the data that used pca.
- rootdir specifies a directory where the program will look for the /test and /train directories, and where it will create csv files for the data and submission.
- target_feat is the name of the column with the classes (musical genres, in this instance)
- sample_rate is the sample rate used for feature extraction from the audio files
- hop_length is the number audio samples between adjacent STFT columns for feature extraction
- number_mfcc is the number of mfcc coefficients created
- n_fft_stft and window_length are other hyperparameters for feature extraction
- validation_proportion is the amount of training data to split into validation data (it will be stratified by the class column)
- step_size is the starting step size for gradient descent
- regularization_strength is the strength of the regularization meant to reduce overfitting
- reps is the number of iterations of gradient descent.  We found it easier to hardcode this as a hyperparameter after trying other convergence method (which you can find commented out in the logReg.grad_desc function).  The first n/8 iterations will use the step size provided, the next n/8 will use step_size/2, then step_size/4, and so on.

## Specifying a Path for the Data Set
Use the rootdir hyperparameter described above.

Example if rootdir var in main.py = "data", then place test and train audio folders in there.


## File Manifest
- compare.py contains functions and a program for running other ml models on the data and printing validation metrics
- create_data.py contains functions for extracting features from the audio files
- graphvarianceexplained.py contains script for graphing our results of testing different variance percentages
- logReg.py contains functions for logistic regression, gradient descent, logistic regression validation, and putting predictions for testing data into a submission file
- main.py contains the main program for creating the data, training the model, and predicting the classes of the testing data
- pca.py contains functions for running pricipal component analysis on the data to reduce dimensions and noise


## Team
- Ben Oghden worked on feature extraction, PCA and TSNE analysis to imporve data, worked on comparing sklearn models(random forest, naive bayes, gradient boosting machines, and SVM) via validation data and wrote most of the report.
- Nathaniel Filer wrote the logReg.py and main.py files, wrote the README, and added a couple sections to the report.


## Final Kaggle Score
- Accuracy: 0.66
- Date: 04/10/2024
- Details:
    - sample_rate = 22050
    - hop_length = 50
    - number_mfcc = 13
    - n_fft_stft = 80
    - window_length = None
    - validation_proportion = 0.2 ( Tho validation was turned off at point of creating dataset so this value did not matter)
    - step_size = 0.01
    - regularization_strength = 2.5
    - reps = 200
