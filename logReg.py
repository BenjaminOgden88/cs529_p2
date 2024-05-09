import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import pdb

#################
### Functions ###
#################

def classes_list(data, class_feature):
    classes = []
    for c in pd.unique(data[class_feature]):
        classes.append(c)

    return classes


def create_y(data, class_feature, classes):
    # Create y matrix (one row per example, one column per class)
    y_rows = []
    row = np.zeros(len(classes))
    for v in data[class_feature]:
        index = classes.index(v)
        new_row = np.copy(row)
        new_row[index] = 1
        y_rows.append(new_row)

    y = pd.DataFrame(y_rows, columns=classes)

    return y


def create_x(data, class_feature):
    # Create x matrix (one row per example, one column of ones + one column per feature)
    ones = pd.Series(np.ones(len(data)), name="Ones")
    columns = [ones]
    for c in data:
        if c != class_feature and c != "id":
            col = data[c]
            #ncol = (col-col.mean())/(col.std())

            columns.append(col)

    x = pd.concat(columns, axis=1)

    return x


def create_w(x, classes):
    # Create w matrix (one row of w0s + one row per feature, one column per class)
    w_rows = []
    for i in range(len(x.columns)):
        rw = np.random.rand(len(classes)-1) # Random weights
        nrw = ((rw-rw.mean())/rw.std()) * 0.0001
        new_row = np.append(nrw, 0)
        w_rows.append(new_row)

    w = pd.DataFrame(w_rows, columns=classes, index=x.columns)

    #print("w :")
    #print(w)

    return w


def create_p(x, w):
    # Calculate expected genre probabilities
    p = x.dot(w)

    #print("x.w :")
    #print(p)

    p = np.exp(p)
    p = p.div(p.sum(axis=1), axis=0)

    return p


def update_w(w, x, p, y, classes, step_size, regularization_strength):
    error_sum = 0
    for c in w:
        if c != classes[len(classes)-1]:
            #col_error_sum = 0
            for f in w.index:
                error = x[f] * (y[c] - p[c])
                esum = error.sum()
                #errorsq = error**2
                #esumsq = errorsq.sum()
                #col_error_sum += esumsq
                w.loc[f, c] = w[c][f] + (step_size * esum) - (step_size * regularization_strength * w[c][f])
            #error_sum += col_error_sum # Used for convergence
                
    #print("Sum of squared errors = " + str(error_sum))

    #print("w :")
    #print(w)

    return w, error_sum


def predictions(p, class_feature, classes):
    prediction_list = []

    for i in range(len(p)):
        row = p.iloc[i]
        rowlist = list(row)
        index = np.argmax(rowlist)
        prediction = classes[index]

        prediction_list.append(prediction)

    return pd.Series(prediction_list, name=class_feature)


def grad_desc(training_file, class_feature, step_size, regularization_strength, reps):
    training_data = pd.read_csv(training_file)
    ss = step_size

    classes = classes_list(training_data, class_feature)
    x = create_x(training_data, class_feature)
    w = create_w(x, classes)
    p = create_p(x, w)
    y = create_y(training_data, class_feature, classes)

    # This hard-coding method for convergence worked much faster than the method below, and it's good enough for this dataset
    for i in range(int(reps/8)):
        w = update_w(w, x, p, y, classes, ss, regularization_strength)[0]
        p = create_p(x, w)

    for i in range(int(reps/8)):
        w = update_w(w, x, p, y, classes, ss/2, regularization_strength)[0]
        p = create_p(x, w)

    for i in range(int(reps/8)):
        w = update_w(w, x, p, y, classes, ss/4, regularization_strength)[0]
        p = create_p(x, w)

    for i in range(int(reps/8)):
        w = update_w(w, x, p, y, classes, ss/8, regularization_strength)[0]
        p = create_p(x, w)

    for i in range(int(reps/8)):
        w = update_w(w, x, p, y, classes, ss/16, regularization_strength)[0]
        p = create_p(x, w)

    for i in range(int(reps/8)):
        w = update_w(w, x, p, y, classes, ss/32, regularization_strength)[0]
        p = create_p(x, w)

    for i in range(int(reps/8)):
        w = update_w(w, x, p, y, classes, ss/64, regularization_strength)[0]
        p = create_p(x, w)

    for i in range(int(reps/8)):
        w = update_w(w, x, p, y, classes, ss/128, regularization_strength)[0]
        p = create_p(x, w)

    # Alternative convergence method which will only update w if it imporves the error
    """update_w_info = update_w(w, x, p, y, classes, ss, regularization_strength)
    error_sum = update_w_info[1]

    w_potential = update_w_info[0]
    p_potential = create_p(x, w_potential)
    update_w_info_potential = update_w(w_potential, x, p_potential, y, classes, ss, regularization_strength)
    error_sum_potential = update_w_info_potential[1]

    counter = 1
    while error_sum > 15315: # allowed_error works best at 15315 for now
        print("iter " + str(counter))
        if error_sum_potential < error_sum:
            w = w_potential
            p = p_potential
            update_w_info = update_w_info_potential
            error_sum = error_sum_potential

            print("Updated W")

            w_potential = update_w_info[0]
            p_potential = create_p(x, w_potential)
            update_w_info_potential = update_w(w_potential, x, p_potential, y, classes, ss, regularization_strength)
            error_sum_potential = update_w_info_potential[1]
        else:
            ss = ss - (ss * (0.5 / (counter + 1)))
            update_w_info = update_w(w, x, p, y, classes, ss, regularization_strength)
            error_sum = update_w_info[1]

            print("Updated ss to " + str(ss))

            w_potential = update_w_info[0]
            p_potential = create_p(x, w_potential)
            update_w_info_potential = update_w(w_potential, x, p_potential, y, classes, ss, regularization_strength)
            error_sum_potential = update_w_info_potential[1]

        counter += 1"""
        
    return w


def log_reg_valid(validation_file, training_file, class_feature, weight_matrix):
    validation_data = pd.read_csv(validation_file)
    training_data = pd.read_csv(training_file)

    classes = classes_list(training_data, class_feature)
    x = create_x(validation_data, class_feature)
    w = weight_matrix
    p = create_p(x, w)

    true = validation_data[class_feature]
    predicted = predictions(p, class_feature, classes)

    cm = metrics.confusion_matrix(true, predicted)
    balanced_accuracy = metrics.balanced_accuracy_score(true, predicted)
    precision = metrics.precision_score(true, predicted, average='macro')
    recall = metrics.recall_score(true, predicted, average='macro')
    f1score = metrics.f1_score(true, predicted, average='macro')

    return cm, balanced_accuracy, precision, recall, f1score


def log_reg_test(testing_file, training_file, class_feature, weight_matrix):
    testing_data = pd.read_csv(testing_file)
    training_data = pd.read_csv(training_file)

    classes = classes_list(training_data, class_feature)
    x = create_x(testing_data, class_feature)
    w = weight_matrix
    p = create_p(x, w)

    id_col = testing_data["id"]
    prediction_col = predictions(p, class_feature, classes)

    submission_data = pd.concat([id_col, prediction_col], axis=1)
    submission_data = submission_data.set_index("id")

    return submission_data