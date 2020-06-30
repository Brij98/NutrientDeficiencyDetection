import joblib
import pandas as pd
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def main():
    #  train_normal_leaf_model()
    #  train_npk_leaf_model()
    train_pk_leaf_model()

    # knn= KNeighborsClassifier(n_neighbors=7).fit(X_train, Y_train)
    # knn_prediction = knn.predict(X_test)
    #
    # accuracy = knn.score(X_test, Y_test)
    # print(accuracy)
    #
    # cm = confusion_matrix(Y_test, knn_prediction)
    #
    # print(cm)
    # print(classification_report(Y_test, knn_prediction))


def train_normal_leaf_model():
    normal_leaf_data = pd.read_csv("D:/leafimages/Normal_leaf_features.csv")

    normal_leaf_data.sample(frac=1).reset_index(drop=True) # shuffle data

    X = normal_leaf_data.drop('CLASS', axis=1)
    Y = normal_leaf_data['CLASS']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
    ns_probs = [0 for _ in range(len(Y_test))]
    svm_model_linear = SVC(kernel='linear', C=1, probability=True).fit(X_train, Y_train)  # polynomial kernel

    # load the saved model
    # load_model = joblib.load(filename=saved_mdl_path)

    svm_prediction = svm_model_linear.predict(X_test)

    accuracy = svm_model_linear.score(X_test, Y_test)
    print(accuracy)  # debug

    # creating a confusion matrix
    cm = confusion_matrix(Y_test, svm_prediction)

    print(cm)
    print(classification_report(Y_test, svm_prediction))

    # predict probabilities
    lr_probs = svm_model_linear.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(Y_test, ns_probs)
    lr_auc = roc_auc_score(Y_test, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(Y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(Y_test, lr_probs)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')

    # save the model
    saved_mdl_path = 'normal_leaf_model.sav'
    joblib.dump(svm_model_linear, saved_mdl_path)

    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()


def predict_normal_leaf_model(x_new):
    load_model = joblib.load(filename='normal_leaf_model.sav')
    y_new = load_model.predict(x_new)
    return y_new


def train_npk_leaf_model():
    npk_leaf_data = pd.read_csv("D:/leafimages/NPK_leaf_features.csv")

    npk_leaf_data.sample(frac=1).reset_index(drop=True)  # shuffle data

    X = npk_leaf_data.drop('CLASS', axis=1)
    Y = npk_leaf_data['CLASS']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
    ns_probs = [0 for _ in range(len(Y_test))]
    svm_model_linear = SVC(kernel='poly', C=1, probability=True).fit(X_train, Y_train)  # polynomial kernel

    # load the saved model
    # load_model = joblib.load(filename=saved_mdl_path)

    svm_prediction = svm_model_linear.predict(X_test)

    accuracy = svm_model_linear.score(X_test, Y_test)
    print(accuracy)  # debug

    # creating a confusion matrix
    cm = confusion_matrix(Y_test, svm_prediction)

    print(cm)
    print(classification_report(Y_test, svm_prediction))

    # predict probabilities
    lr_probs = svm_model_linear.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(Y_test, ns_probs)
    lr_auc = roc_auc_score(Y_test, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(Y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(Y_test, lr_probs)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')

    # save the model
    saved_mdl_path = 'npk_leaf_model.sav'
    joblib.dump(svm_model_linear, saved_mdl_path)

    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()


def predict_npk_leaf_model(x_new):
    load_model = joblib.load(filename='npk_leaf_model.sav')
    y_new = load_model.predict(x_new)
    return y_new


def train_pk_leaf_model():
    pk_leaf_data = pd.read_csv("D:/leafimages/PK_leaf_features.csv")


    pk_leaf_data.sample(frac=1).reset_index(drop=True)  # shuffle data

    X = pk_leaf_data.drop('CLASS', axis=1)
    Y = pk_leaf_data['CLASS']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
    ns_probs = [0 for _ in range(len(Y_test))]
    svm_model_linear = SVC(kernel='poly', C=1, probability=True).fit(X_train, Y_train)  # polynomial kernel

    # load the saved model
    # load_model = joblib.load(filename=saved_mdl_path)

    svm_prediction = svm_model_linear.predict(X_test)

    accuracy = svm_model_linear.score(X_test, Y_test)
    print(accuracy)  # debug

    # creating a confusion matrix
    cm = confusion_matrix(Y_test, svm_prediction)

    print(cm)
    print(classification_report(Y_test, svm_prediction))

    # predict probabilities
    lr_probs = svm_model_linear.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(Y_test, ns_probs)
    lr_auc = roc_auc_score(Y_test, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(Y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(Y_test, lr_probs)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')

    # save the model
    saved_mdl_path = 'pk_leaf_model.sav'
    joblib.dump(svm_model_linear, saved_mdl_path)

    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()


def predict_pk_leaf_model(x_new):
    load_model = joblib.load(filename='pk_leaf_model.sav')
    y_new = load_model.predict(x_new)
    return y_new


if __name__ == "__main__":
    # train_normal_leaf_model()
    main()

# reference:
# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
# https://www.geeksforgeeks.org/multiclass-classification-using-scikit-learn/
# https://machinelearningmastery.com/make-predictions-scikit-learn/
# https://machinelearningmastery.com/train-final-machine-learning-model/
# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
# https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
