import time
from sklearn.model_selection import train_test_split
import create_data
import pca
import logReg

# Hyperparameters
create_dataframes = True
validation_on = False
#Whether or not t-SNE visual will display when doing PCA
tsne = True
run_logistic_reg_with_pca_data = False

rootdir = './data'
target_feat = 'class'

#sample rate of audio files
sample_rate = 22050
# Number audio samples between adjacent STFT columns
# smaller hop length => more columns will be made
# BASED ON librosa documentation 0.10 version
# default hop length is 512
hop_length = 50
number_mfcc = 13
#for STFTs.Note this wont change anything as STFT were removed from the final dataset creation
n_fft_stft = 80
window_length = None

#how much of the variance will be explained
variance_maintained = 0.9
#how much of the training data will we split out
validation_proportion = 0.2

step_size = 0.01
regularization_strength = 2.5
reps = 200

# Create Dataframes
if create_dataframes == True:
    creating_data_st = time.time()

    all_training_data_frame = create_data.create_data_frame(rootdir, 'train', sample_rate, hop_length, number_mfcc, n_fft_stft, window_length)
    all_training_data_frame.transform

    testing_data_frame = create_data.create_data_frame(rootdir, 'test', sample_rate, hop_length, number_mfcc, n_fft_stft, window_length)
    testing_data_frame.to_csv(rootdir+"/testing_data.csv",index=False)

    train_valid_split_data_frames = train_test_split(all_training_data_frame,
                                                    test_size = validation_proportion,
                                                    stratify=all_training_data_frame[target_feat])
    validation_data_frame = train_valid_split_data_frames[1].reset_index(drop=True)
    
    if validation_on == True:
        training_data_frame = train_valid_split_data_frames[0].reset_index(drop=True)
        validation_data_frame.to_csv(rootdir+"/validation_data.csv",index=False)
    else:
        training_data_frame = all_training_data_frame

    training_data_frame.to_csv(rootdir+"/training_data.csv",index=False)

    # Run PCA
    reduced_dims_data_frames = pca.reduced_dimensions_dataframes(target_feat, training_data_frame, validation_data_frame, testing_data_frame, tsne,variance_maintained)

    new_data_frame_train = reduced_dims_data_frames[0]
    new_data_frame_vaild = reduced_dims_data_frames[1]
    new_data_frame_test = reduced_dims_data_frames[2]

    new_data_frame_train.to_csv(rootdir+"/training_data_pca.csv",index=False)
    new_data_frame_vaild.to_csv(rootdir+"/validation_data_pca.csv",index=False)
    new_data_frame_test.to_csv(rootdir+"/testing_data_pca.csv",index=False)

    print("Created data in time: " + str(time.time() - creating_data_st))

# Run Gradient Descent
grad_desc_st = time.time()

if run_logistic_reg_with_pca_data == True:
    training_file = rootdir+'/training_data_pca.csv'
    validation_file = rootdir+'/validation_data_pca.csv'
    testing_file = rootdir+'/testing_data_pca.csv'
else:
    training_file = rootdir+'/training_data.csv'
    validation_file = rootdir+'/validation_data.csv'
    testing_file = rootdir+'/testing_data.csv'

w = logReg.grad_desc(training_file, target_feat, step_size, regularization_strength, reps)

print("Gradient descent completed in time: " + str(time.time() - grad_desc_st))

# Run Validation
if validation_on:
    valid_st = time.time()
    validation_metrics = logReg.log_reg_valid(validation_file, training_file, target_feat, w)

    confusion_matrix = validation_metrics[0]
    balanced_accuracy = validation_metrics[1]
    precision = validation_metrics[2]
    recall = validation_metrics[3]
    f1score = validation_metrics[4]

    print("Confusion Matrix:")
    print(confusion_matrix)
    print("---------------------------")

    print("Balanced Accuracy:")
    print(balanced_accuracy)
    print("---------------------------")

    print("Precision:")
    print(precision)
    print("---------------------------")

    print("F1 Score:")
    print(f1score)
    print("---------------------------")

    print("Recall:")
    print(recall)
    print("---------------------------")

    print("Validation completed in time: " + str(time.time() - valid_st))

# Create submission file with testing data
submission_data = logReg.log_reg_test(testing_file, training_file, target_feat, w)

submission_data.to_csv(rootdir+"/OgdenFilerSubmission.csv")
