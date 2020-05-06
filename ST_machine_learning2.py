"""
Research Workshop Machine Learning Part
"""

#All the packages
from sklearn.svm import LinearSVC
import mglearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.datasets import make_classification
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import mglearn 
from sklearn.datasets import load_digits
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import sklearn
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

#%%
#IMPORTING DATA SET AND GETTING THE 14 TARGET VARIABLES
path = 'C:/Users/Gebruiker/OneDrive/Documents/CSAI 2_2/Research Workshop'
df = pd.read_csv(path + '/features2.csv') 

X = df.drop(columns=['label','no_peaks'])
y = df['label'].values

new_y = []
y = list(y)

for value in y:
    value = str(value)
    if len(value) == 2:
        new_y.append(int(value[0]))
    else:
        new_y.append(int(value[:2]))
    
final_y = np.asarray(new_y, dtype=np.float32)
y = final_y
print(y)

#%%
#Feature scaling, Usefull for some models
scaler = preprocessing.StandardScaler().fit(X)
X_s = scaler.transform(X)
print(X_s)

#%%
#----------------------------------------------------------------------------------------------
#SVC
#Varying C, gamma, with kernel = "rbf"
#----------------------------------------------------------------------------------------------

# Number of random trials
NUM_TRIALS = 30

# Specifying dataset
X_svc = X_s
y = y

# Set up possible values of parameters to optimize over
p_grid = {"C": [.01, .25, 10, 100],
          "gamma": [.001, .01, .1, 1]}

# We will use a Support Vector Classifier with "rbf" kernel
svc = SVC(kernel="rbf")

# Arrays to store scores
svc_non_nested_scores = np.zeros(NUM_TRIALS)
svc_nested_scores = np.zeros(NUM_TRIALS)

svc_best_pars = [] #new


# Loop for each trial
for i in range(NUM_TRIALS):
    
    print("trial:{}".format(i+1))
    
    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
    svc_inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    svc_outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=svc, param_grid=p_grid, cv=svc_inner_cv)
    clf.fit(X_svc, y)
    svc_non_nested_scores[i] = clf.best_score_
    
    svc_best_pars.append((clf.best_params_['C'], clf.best_params_['gamma']))

    # Nested CV with parameter optimization
    nested_score = cross_val_score(clf, X=X_svc, y=y, cv=svc_outer_cv)
    svc_nested_scores[i] = nested_score.mean()
    

svc_score_difference = svc_non_nested_scores - svc_nested_scores

print("Average difference of {:6f} with std. dev. of {:6f}."
      .format(svc_score_difference.mean(), svc_score_difference.std()))

# Plot scores on each trial for nested and non-nested CV
plt.figure()
plt.subplot(211)
non_nested_scores_line, = plt.plot(svc_non_nested_scores, color='r')
nested_line, = plt.plot(svc_nested_scores, color='b')
plt.ylabel("score", fontsize="14")
plt.legend([non_nested_scores_line, nested_line],
           ["Non-Nested CV", "Nested CV"],
           bbox_to_anchor=(0, .4, .5, 0))
plt.title("Non-Nested and Nested Cross Validation on Touch Classification Dataset for SVC",
          x=.5, y=1.1, fontsize="15")

# Plot bar chart of the difference.
plt.subplot(212)
difference_plot = plt.bar(range(NUM_TRIALS), svc_score_difference)
plt.xlabel("Individual Trial #")
plt.legend([difference_plot],
           ["Non-Nested CV - Nested CV Score"],
           bbox_to_anchor=(0, 1, .8, 0))
plt.ylabel("score difference", fontsize="14")


print("SVC")
print("Number of Trials:{}".format(NUM_TRIALS))
print("Best parameters:{}\n".format(svc_best_pars))
print("Non Nested Scores:\n{}".format(svc_non_nested_scores))
print("Max score: {} (index {})\n".format(svc_non_nested_scores.max(), np.argmax(svc_non_nested_scores)))
print("Nested Scores:\n{}".format(svc_nested_scores))
print("Max score: {} (index {})\n".format(svc_nested_scores.max(), np.argmax(svc_nested_scores)))

#----------------------------------------------------------------------------------------------
#Confusion Matrix SVC

C = 10
gamma = .1
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X_svc, y, random_state=7)

svc_classifier = SVC(kernel="rbf", gamma=gamma, C=C).fit(X_train,y_train)

print("Support Vector Classifier Confusion Matrix using the best parameters")
print("Train score:{}".format(svc_classifier.score(X_train, y_train)))
print("Test score:{}\n".format(svc_classifier.score(X_test, y_test)))

class_names = ['grab','hit','massage','pat','pinch','poke','press','rub','scratch','slap','squeeze','stroke','tap','tickle']
np.set_printoptions(precision= 2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(svc_classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize, values_format='.2f', xticks_rotation='45')
    
    disp.ax_.set_title(title)
    

    print(title)
    print(disp.confusion_matrix)

plt.rcParams["figure.figsize"] = (10,10)

plt.show()




#%%
#----------------------------------------------------------------------------------------------
#KNEIGHBORS
#Varying number of neighbours : 1,5,10,15.
#----------------------------------------------------------------------------------------------
# Number of random trials
NUM_TRIALS = 30

# Load the dataset

X_knn = X_s
y = y

# Set up possible values of parameters to optimize over
p_grid = {"n_neighbors": [1, 5, 10, 15]}

# We will use a KNN classifier
knn = KNeighborsClassifier()

# Arrays to store scores
knn_non_nested_scores = np.zeros(NUM_TRIALS)
knn_nested_scores = np.zeros(NUM_TRIALS)

knn_best_pars = []


# Loop for each trial
for i in range(NUM_TRIALS):
    print("trial:{}".format(i+1))

    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
    knn_inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    knn_outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=knn, param_grid=p_grid, cv=knn_inner_cv)
    clf.fit(X_knn, y)
    knn_non_nested_scores[i] = clf.best_score_
    
    knn_best_pars.append(clf.best_params_['n_neighbors'])

    # Nested CV with parameter optimization
    nested_score = cross_val_score(clf, X=X_knn, y=y, cv=knn_outer_cv)
    knn_nested_scores[i] = nested_score.mean()

knn_score_difference = knn_non_nested_scores - knn_nested_scores

print("Average difference of {:6f} with std. dev. of {:6f}."
      .format(knn_score_difference.mean(), knn_score_difference.std()))

# Plot scores on each trial for nested and non-nested CV
plt.figure()
plt.subplot(211)
non_nested_scores_line, = plt.plot(knn_non_nested_scores, color='r')
nested_line, = plt.plot(knn_nested_scores, color='b')
plt.ylabel("score", fontsize="14")
plt.legend([non_nested_scores_line, nested_line],
           ["Non-Nested CV", "Nested CV"],
           bbox_to_anchor=(0, .4, .5, 0))
plt.title("Non-Nested and Nested Cross Validation on Touch Classification Data set KNN",
          x=.5, y=1.1, fontsize="15")

# Plot bar chart of the difference.
plt.subplot(212)
difference_plot = plt.bar(range(NUM_TRIALS), knn_score_difference)
plt.xlabel("Individual Trial #")
plt.legend([difference_plot],
           ["Non-Nested CV - Nested CV Score"],
           bbox_to_anchor=(0, 1, .8, 0))
plt.ylabel("score difference", fontsize="14")

print("\nKNN")
print("Number of Trials:{}".format(NUM_TRIALS))
print("Best parameters:{}\n".format(knn_best_pars))
print("Non Nested Scores:\n{}".format(knn_non_nested_scores))
print("Max score: {} (index {})\n".format(knn_non_nested_scores.max(), np.argmax(knn_non_nested_scores)))
print("Nested Scores:\n{}".format(knn_nested_scores))
print("Max score: {} (index {})\n".format(knn_nested_scores.max(), np.argmax(knn_nested_scores)))

#----------------------------------------------------------------------------------------------
#Confusion Matrix KNN

n_neighbors = 1 
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X_knn, y, random_state=1)

knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train,y_train)

print("KNN Confusion Matrix using the best parameters")
print("Train score:{}".format(knn_classifier.score(X_train, y_train)))
print("Test score:{}\n".format(knn_classifier.score(X_test, y_test)))

class_names = ['grab','hit','massage','pat','pinch','poke','press','rub','scratch','slap','squeeze','stroke','tap','tickle']
np.set_printoptions(precision= 2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(knn_classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize, values_format='.2f', xticks_rotation='45')
    
    disp.ax_.set_title(title)
    

    print(title)
    print(disp.confusion_matrix)

plt.rcParams["figure.figsize"] = (10,10)

plt.show()

#%%
#----------------------------------------------------------------------------------------------
#KNN with Nearest centroid.
#Varying the shrink threshold (0,0.25,0.5)
#----------------------------------------------------------------------------------------------

# Number of random trials
NUM_TRIALS = 30

# Load the dataset

X_knnc = X_s
y = y

# Set up possible values of parameters to optimize over
p_grid = {"shrink_threshold": [0, 0.25, 0.5]}

# We will use a Support Vector Classifier with "rbf" kernel
knnc = NearestCentroid()

# Arrays to store scores
knnc_non_nested_scores = np.zeros(NUM_TRIALS)
knnc_nested_scores = np.zeros(NUM_TRIALS)

knnc_best_pars = []

# Loop for each trial
for i in range(NUM_TRIALS):
    print("trial:{}".format(i+1))

    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
    knnc_inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    knnc_outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=knnc, param_grid=p_grid, cv=knnc_inner_cv)
    clf.fit(X_knnc, y)
    knnc_non_nested_scores[i] = clf.best_score_
    
    
    knnc_best_pars.append(clf.best_params_['shrink_threshold'])

    # Nested CV with parameter optimization
    nested_score = cross_val_score(clf, X=X_knnc, y=y, cv=knnc_outer_cv)
    knnc_nested_scores[i] = nested_score.mean()
    

score_difference = knnc_non_nested_scores - knnc_nested_scores

print("Average difference of {:6f} with std. dev. of {:6f}."
      .format(score_difference.mean(), score_difference.std()))

# Plot scores on each trial for nested and non-nested CV
plt.figure()
plt.subplot(211)
non_nested_scores_line, = plt.plot(knnc_non_nested_scores, color='r')
nested_line, = plt.plot(knnc_nested_scores, color='b')
plt.ylabel("score", fontsize="14")
plt.legend([non_nested_scores_line, nested_line],
           ["Non-Nested CV", "Nested CV"],
           bbox_to_anchor=(0, .4, .5, 0))
plt.title("Non-Nested and Nested Cross Validation on Touch Classification Data set KNN-centroid",
          x=.5, y=1.1, fontsize="15")

# Plot bar chart of the difference.
plt.subplot(212)
difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
plt.xlabel("Individual Trial #")
plt.legend([difference_plot],
           ["Non-Nested CV - Nested CV Score"],
           bbox_to_anchor=(0, 1, .8, 0))
plt.ylabel("score difference", fontsize="14")

print("\nKNN Nearest Centroid")
print("Number of Trials:{}".format(NUM_TRIALS))
print("Best parameters:{}\n".format(knnc_best_pars))
print("Non Nested Scores:\n{}".format(knnc_non_nested_scores))
print("Max score: {} (index {})\n".format(knnc_non_nested_scores.max(), np.argmax(knnc_non_nested_scores)))
print("Nested Scores:\n{}".format(knnc_nested_scores))
print("Max score: {} (index {})\n".format(knnc_nested_scores.max(), np.argmax(knnc_nested_scores)))

#----------------------------------------------------------------------------------------------
#Confusion Matrix KNN NC

shrink_threshold = 0
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X_knnc, y, random_state=3)

knnc_classifier = NearestCentroid(shrink_threshold=shrink_threshold).fit(X_train,y_train)

print("KNN Nearest Centroid Confusion Matrix using the best parameters")
print("Train score:{}".format(knnc_classifier.score(X_train, y_train)))
print("Test score:{}\n".format(knnc_classifier.score(X_test, y_test)))

class_names = ['grab','hit','massage','pat','pinch','poke','press','rub','scratch','slap','squeeze','stroke','tap','tickle']
np.set_printoptions(precision= 2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(knnc_classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize, values_format='.2f', xticks_rotation='45')
    
    disp.ax_.set_title(title)
    

    print(title)
    print(disp.confusion_matrix)

plt.rcParams["figure.figsize"] = (10,10)

plt.show()

#%%
#----------------------------------------------------------------------------------------------
#Logistic Regression
#Parameters: Varying C(0.01,1,100)
#Penalty L2
#Multi_class = "multinomial"
#----------------------------------------------------------------------------------------------

# Number of random trials
NUM_TRIALS = 30

# Load the dataset

X_lr = X_s
y = y

# Set up possible values of parameters to optimize over
p_grid = {"C": [0.01,1,100]}

# We will use a Support Vector Classifier with "rbf" kernel
lr = LogisticRegression(max_iter = 100000,penalty="l2",multi_class = 'multinomial')

# Arrays to store scores
lr_non_nested_scores = np.zeros(NUM_TRIALS)
lr_nested_scores = np.zeros(NUM_TRIALS)

lr_best_pars = []

# Loop for each trial
for i in range(NUM_TRIALS):
    print("trial:{}".format(i+1))

    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.


    lr_inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    lr_outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=lr, param_grid=p_grid, cv=lr_inner_cv)
    clf.fit(X_lr, y)
    lr_non_nested_scores[i] = clf.best_score_

    # Nested CV with parameter optimization
    lr_nested_score = cross_val_score(clf, X=X_lr, y=y, cv=lr_outer_cv)
    lr_nested_scores[i] = nested_score.mean()
    
    lr_best_pars.append(clf.best_params_['C'])

score_difference = lr_non_nested_scores - lr_nested_scores

print("Average difference of {:6f} with std. dev. of {:6f}."
      .format(score_difference.mean(), score_difference.std()))

# Plot scores on each trial for nested and non-nested CV
plt.figure()
plt.subplot(211)
non_nested_scores_line, = plt.plot(lr_non_nested_scores, color='r')
nested_line, = plt.plot(lr_nested_scores, color='b')
plt.ylabel("score", fontsize="14")
plt.legend([non_nested_scores_line, nested_line],
           ["Non-Nested CV", "Nested CV"],
           bbox_to_anchor=(0, .4, .5, 0))
plt.title("Non-Nested and Nested Cross Validation on Touch Classification Data set Logistic Regression",
          x=.5, y=1.1, fontsize="15")

# Plot bar chart of the difference.
plt.subplot(212)
difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
plt.xlabel("Individual Trial #")
plt.legend([difference_plot],
           ["Non-Nested CV - Nested CV Score"],
           bbox_to_anchor=(0, 1, .8, 0))
plt.ylabel("score difference", fontsize="14")

print("\nLogistic Regression")
print("Number of Trials:{}".format(NUM_TRIALS))
print("Best parameters:{}\n".format(lr_best_pars))
print("Non Nested Scores:\n{}".format(lr_non_nested_scores))
print("Max score: {} (index {})\n".format(lr_non_nested_scores.max(), np.argmax(lr_non_nested_scores)))
print("Nested Scores:\n{}".format(lr_nested_scores))
print("Max score: {} (index {})\n".format(lr_nested_scores.max(), np.argmax(lr_nested_scores)))

#----------------------------------------------------------------------------------------------
#Confusion Matrix Logistic Regression

C = 100
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X_lr, y, random_state=25)

lr_classifier = LogisticRegression(C=C, max_iter = 100000,penalty="l2",multi_class = 'multinomial').fit(X_train,y_train)

print("Logistic Regression Confusion Matrix using the best parameters")
print("Train score:{}".format(lr_classifier.score(X_train, y_train)))
print("Test score:{}\n".format(lr_classifier.score(X_test, y_test)))

class_names = ['grab','hit','massage','pat','pinch','poke','press','rub','scratch','slap','squeeze','stroke','tap','tickle']
np.set_printoptions(precision= 2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(lr_classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize, values_format='.2f', xticks_rotation='45')
    
    disp.ax_.set_title(title)
    

    print(title)
    print(disp.confusion_matrix)

plt.rcParams["figure.figsize"] = (10,10)

plt.show()

#%%
#----------------------------------------------------------------------------------------------
#TREES
#Varying max_depth to 3,4,5,6
#Setting splitting criterion as "entropy"
#----------------------------------------------------------------------------------------------

# Number of random trials
NUM_TRIALS = 30

# Load the dataset

X_tr = X_s
y = y

# Set up possible values of parameters to optimize over
p_grid = {"max_depth": [6,8,12]}

# We will use a Support Vector Classifier with "rbf" kernel

tr = DecisionTreeClassifier(criterion = "entropy")

# Arrays to store scores
tr_non_nested_scores = np.zeros(NUM_TRIALS)
tr_nested_scores = np.zeros(NUM_TRIALS)

tr_best_pars = [] #new

# Loop for each trial
for i in range(NUM_TRIALS):
    print("trial:{}".format(i+1))

    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
    tr_inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    tr_outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=tr, param_grid=p_grid, cv=tr_inner_cv)
    clf.fit(X_tr, y)
    tr_non_nested_scores[i] = clf.best_score_

    # Nested CV with parameter optimization
    nested_score = cross_val_score(clf, X=X_tr, y=y, cv=tr_outer_cv)
    tr_nested_scores[i] = nested_score.mean()
    
    tr_best_pars.append(clf.best_params_['max_depth'])

score_difference = tr_non_nested_scores - tr_nested_scores

print("Average difference of {:6f} with std. dev. of {:6f}."
      .format(score_difference.mean(), score_difference.std()))

# Plot scores on each trial for nested and non-nested CV
plt.figure()
plt.subplot(211)
non_nested_scores_line, = plt.plot(tr_non_nested_scores, color='r')
nested_line, = plt.plot(tr_nested_scores, color='b')
plt.ylabel("score", fontsize="14")
plt.legend([non_nested_scores_line, nested_line],
           ["Non-Nested CV", "Nested CV"],
           bbox_to_anchor=(0, .4, .5, 0))
plt.title("Non-Nested and Nested Cross Validation on Touch Classification Data set Trees",
          x=.5, y=1.1, fontsize="15")

# Plot bar chart of the difference.
plt.subplot(212)
difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
plt.xlabel("Individual Trial #")
plt.legend([difference_plot],
           ["Non-Nested CV - Nested CV Score"],
           bbox_to_anchor=(0, 1, .8, 0))
plt.ylabel("score difference", fontsize="14")

print("\nTrees")
print("Number of Trials:{}".format(NUM_TRIALS))
print("Best parameters:{}\n".format(tr_best_pars))
print("Non Nested Scores:\n{}".format(tr_non_nested_scores))
print("Max score: {} (index {})\n".format(tr_non_nested_scores.max(), np.argmax(tr_non_nested_scores)))
print("Nested Scores:\n{}".format(tr_nested_scores))
print("Max score: {} (index {})\n".format(tr_nested_scores.max(), np.argmax(tr_nested_scores)))

#----------------------------------------------------------------------------------------------
#Confusion Matrix Trees

max_depth = 8
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X_tr, y, random_state=26)

tr_classifier = DecisionTreeClassifier(max_depth=max_depth, criterion = "entropy").fit(X_train,y_train)

print("Trees Confusion Matrix using the best parameters")
print("Train score:{}".format(tr_classifier.score(X_train, y_train)))
print("Test score:{}\n".format(tr_classifier.score(X_test, y_test)))

class_names = ['grab','hit','massage','pat','pinch','poke','press','rub','scratch','slap','squeeze','stroke','tap','tickle']
np.set_printoptions(precision= 2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(tr_classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize, values_format='.2f', xticks_rotation='45')
    
    disp.ax_.set_title(title)
    

    print(title)
    print(disp.confusion_matrix)

plt.rcParams["figure.figsize"] = (10,10)

plt.show()

#%%
#----------------------------------------------------------------------------------------------
#MLP
#Varying alpha [0.0001, 0.01, 0.1, 1]
#Varying hidden_layer_size [(10,10), (20,20),(30,30)]
#Activation function "tanh"
#----------------------------------------------------------------------------------------------

# Number of random trials
NUM_TRIALS = 30

# Load the dataset

X_mlp = X_s
y = y

# Set up possible values of parameters to optimize over

p_grid = {"alpha": [0.01, 0.1, 1],
          'hidden_layer_sizes': [(40,40),(70,70),(100,100)]}

# We will use MLP with tanh as activation
mlp = MLPClassifier(max_iter=50, activation='tanh')


# Arrays to store scores
mlp_non_nested_scores = np.zeros(NUM_TRIALS)
mlp_nested_scores = np.zeros(NUM_TRIALS)

mlp_best_pars = [] #new

# Loop for each trial
for i in range(NUM_TRIALS):
    print("trial:{}".format(i+1))

    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
    mlp_inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    mlp_outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=mlp, param_grid=p_grid, cv=mlp_inner_cv)
    clf.fit(X_mlp, y)
    mlp_non_nested_scores[i] = clf.best_score_

    # Nested CV with parameter optimization
    nested_score = cross_val_score(clf, X=X_mlp, y=y, cv=mlp_outer_cv)
    mlp_nested_scores[i] = nested_score.mean()
    
    mlp_best_pars.append((clf.best_params_['alpha'], clf.best_params_['hidden_layer_sizes']))

score_difference = mlp_non_nested_scores - mlp_nested_scores

print("Average difference of {:6f} with std. dev. of {:6f}."
      .format(score_difference.mean(), score_difference.std()))

# Plot scores on each trial for nested and non-nested CV
plt.figure()
plt.subplot(211)
non_nested_scores_line, = plt.plot(mlp_non_nested_scores, color='r')
nested_line, = plt.plot(mlp_nested_scores, color='b')
plt.ylabel("score", fontsize="14")
plt.legend([non_nested_scores_line, nested_line],
           ["Non-Nested CV", "Nested CV"],
           bbox_to_anchor=(0, .4, .5, 0))
plt.title("Non-Nested and Nested Cross Validation on Touch Classification Data set Trees",
          x=.5, y=1.1, fontsize="15")

# Plot bar chart of the difference.
plt.subplot(212)
difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
plt.xlabel("Individual Trial #")
plt.legend([difference_plot],
           ["Non-Nested CV - Nested CV Score"],
           bbox_to_anchor=(0, 1, .8, 0))
plt.ylabel("score difference", fontsize="14")

print("\nMultilayered Perceptron")
print("Number of Trials:{}".format(NUM_TRIALS))
print("Best parameters:{}\n".format(mlp_best_pars))
print("Non Nested Scores:\n{}".format(mlp_non_nested_scores))
print("Max score: {} (index {})\n".format(mlp_non_nested_scores.max(), np.argmax(mlp_non_nested_scores)))
print("Nested Scores:\n{}".format(mlp_nested_scores))
print("Max score: {} (index {})\n".format(mlp_nested_scores.max(), np.argmax(mlp_nested_scores)))

#----------------------------------------------------------------------------------------------
#Confusion Matrix MLP

alpha = 0.01
hidden_layer_sizes = (100,100)
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X_mlp, y, random_state=16)

mlp_classifier = MLPClassifier(alpha = alpha, hidden_layer_sizes=hidden_layer_sizes,activation='tanh').fit(X_train,y_train)


print("Multilayered Perceptron Confusion Matrix using the best parameters")
print("Train score:{}".format(mlp_classifier.score(X_train, y_train)))
print("Test score:{}\n".format(mlp_classifier.score(X_test, y_test)))

class_names = ['grab','hit','massage','pat','pinch','poke','press','rub','scratch','slap','squeeze','stroke','tap','tickle']
np.set_printoptions(precision= 2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(mlp_classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize, values_format='.2f', xticks_rotation='45')
    
    disp.ax_.set_title(title)
    

    print(title)
    print(disp.confusion_matrix)

plt.rcParams["figure.figsize"] = (10,10)

plt.show()

