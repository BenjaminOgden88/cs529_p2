from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import make_classification
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split


#given training, validation and testing dataframes, returns all 3 dataframes after pca applied to all of them. Also runs tsne on training data and displays visual
def reduced_dimensions_dataframes(target:str,training_dataframe :pd.DataFrame, validation_dataframe : pd.DataFrame, testing_dataframe : pd.DataFrame,run_tsne_analysis : bool,varaince_maintained):
    features = training_dataframe.columns.to_list()
    #if target is None then there is no target attr(test data)

    features.remove("id")
    if target != None: 
        features.remove(target)


    #standarize data, while removing traget attr from PCA
    x_train = training_dataframe.loc[:, features].values
    y_train = training_dataframe.loc[:,[target]].values

    id_training = training_dataframe.loc[:,["id"]].values
    id_valid = validation_dataframe.loc[:,["id"]].values
    id_testing = testing_dataframe.loc[:,["id"]].values

    x_valid = validation_dataframe.loc[:, features].values
    
    y_valid = validation_dataframe.loc[:,[target]].values

    x_test = testing_dataframe.loc[:, features].values

    #default n_components=2
    sc = StandardScaler()
    pca = PCA(n_components=varaince_maintained)

    #IMPORTANT. only call fit transform on training data, use transform for validation or testing data
    x_standarized_data_train = sc.fit_transform(x_train)
    transformed_x_from_pca_train = pca.fit_transform(x_standarized_data_train)

    x_standarized_data_valid = sc.transform(x_valid)
    transformed_x_from_pca_valid = pca.transform(x_standarized_data_valid)

    x_standarized_data_testing = sc.transform(x_test)
    transformed_x_from_pca_testing = pca.transform(x_standarized_data_testing)
    #explained variance ratio of all components kept
    print(pca.explained_variance_ratio_)

    #make new dataset
    new_data_frame_train = pd.DataFrame(data = transformed_x_from_pca_train)
    new_data_frame_vaild = pd.DataFrame(data = transformed_x_from_pca_valid)
    new_data_frame_test = pd.DataFrame(data = transformed_x_from_pca_testing)
    #shape[1] is the total cols in the dataframes
    #place targets and ids back onto the training and validation data

    new_data_frame_train.insert(new_data_frame_train.shape[1],"id",id_training.flatten())
    new_data_frame_vaild.insert(new_data_frame_vaild.shape[1],"id",id_valid.flatten())
    new_data_frame_test.insert(new_data_frame_test.shape[1],"id",id_testing.flatten())


    new_data_frame_train.insert(new_data_frame_train.shape[1],target,y_train.flatten())
    new_data_frame_vaild.insert(new_data_frame_vaild.shape[1],target,y_valid.flatten())


    if run_tsne_analysis == True:
            
        #TSNE ANALYSIS
        #This website helped give us a starting point for t-SNE visual
        #https://www.datacamp.com/tutorial/introduction-t-sne
        print("doing tsne analysis on training data")
        #UNCOMMENT LINE "X = x_train" IF YOU WANT TSNE TO GO OFF OF NON PCA APPLIED DATA.
        #UNCOMMENT LINE "X = transformed_x_from_pca_train" IF YOU WANT TSNE TO GO OFF PCA APPLIED DATA
        #use if want to just use normal data
        #X = x_train
        #use if account for pca
        #X = x_standarized_data_train
        #use if pca train data
        X = transformed_x_from_pca_train
        tsne = TSNE(n_components=2, random_state=None,perplexity=6)
        X_tsne = tsne.fit_transform(X)
        tsne.kl_divergence_
        #the scatter takes numbers, not strings for class names, hence colored_class_y is list of ints, where number is mapped
        #to given class
        dict_map_classes_to_numbers = {}
        count = 0
        #y is list of lists, each class is like [class] hence i[0]
        for i in y_train:
            #print(i)
            if i[0] not in dict_map_classes_to_numbers:
                dict_map_classes_to_numbers[i[0]] = count
                count = count + 1
        colored_class_y = []
        for i in y_train:
            colored_class_y.append(dict_map_classes_to_numbers[i[0]])

        fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=colored_class_y)
        fig.update_layout(
            title="t-SNE visualization of Custom Classification dataset",
            xaxis_title="First t-SNE",
            yaxis_title="Second t-SNE",
        )
        fig.show()
    return (new_data_frame_train,new_data_frame_vaild,new_data_frame_test)




if __name__ == '__main__':
#example for debugging purposes
    rootdir = "data"
    training_dataframe = pd.read_csv(rootdir+"/training_data.csv")
    target_att = "class"
    validation_proportion = 0.25
    validation_data = pd.read_csv(rootdir+"/validation_data.csv")
    testing_data_frame = pd.read_csv(rootdir+"/testing_data.csv")
    ignored_features = ['id']
    reduced_dims_data_frames = reduced_dimensions_dataframes(target_att,training_dataframe,validation_data,testing_data_frame,False,0.95)

    new_data_frame_train = reduced_dims_data_frames[0]
    new_data_frame_vaild = reduced_dims_data_frames[1]
    new_data_frame_test = reduced_dims_data_frames[2]

    new_data_frame_train.to_csv(rootdir+"/training_data_pca.csv",index=False)
    new_data_frame_vaild.to_csv(rootdir+"/validation_data_pca.csv",index=False)
    new_data_frame_test.to_csv(rootdir+"/testing_data_pca.csv",index=False)