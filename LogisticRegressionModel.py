import joblib
import pandas as pd
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


def main():
    train_pk_sheath_model()


def train_normal_sheath_model():
    normal_sheath_data = pd.read_csv("D:/leafimages/Normal_sheath_features.csv")

    normal_sheath_data.sample(frac=1).reset_index(drop=True)  # shuffle data

    X = normal_sheath_data.drop('CLASS', axis=1)
    Y = normal_sheath_data['CLASS']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
    ns_probs = [0 for _ in range(len(Y_test))]
    logisticreg_classifier = LogisticRegression(random_state=0).fit(X_train, Y_train)

    logistic_reg_prediction = logisticreg_classifier.predict(X_test)

    accuracy = logisticreg_classifier.score(X_test, Y_test)
    print(accuracy)  # debug

    # creating a confusion matrix
    cm = confusion_matrix(Y_test, logistic_reg_prediction)

    print(cm)
    print(classification_report(Y_test, logistic_reg_prediction))

    # predict probabilities
    lr_probs = logisticreg_classifier.predict_proba(X_test)
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
    saved_mdl_path = 'normal_sheath_model.sav'
    joblib.dump(logisticreg_classifier, saved_mdl_path)

    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()


def predict_normal_sheath_model(x_new):
    load_model = joblib.load(filename='normal_sheath_model.sav')
    y_new = load_model.predict(x_new)
    return y_new


def train_npk_sheath_model():
    npk_sheath_data = pd.read_csv("D:/leafimages/NPK_sheath_features.csv")

    npk_sheath_data.sample(frac=1).reset_index(drop=True)  # shuffle data

    X = npk_sheath_data.drop('CLASS', axis=1)
    Y = npk_sheath_data['CLASS']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
    ns_probs = [0 for _ in range(len(Y_test))]
    logisticreg_classifier = LogisticRegression(random_state=0).fit(X_train, Y_train)

    logistic_reg_prediction = logisticreg_classifier.predict(X_test)

    accuracy = logisticreg_classifier.score(X_test, Y_test)
    print(accuracy)  # debug

    # creating a confusion matrix
    cm = confusion_matrix(Y_test, logistic_reg_prediction)

    print(cm)
    print(classification_report(Y_test, logistic_reg_prediction))

    # predict probabilities
    lr_probs = logisticreg_classifier.predict_proba(X_test)
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
    saved_mdl_path = 'npk_sheath_model.sav'
    joblib.dump(logisticreg_classifier, saved_mdl_path)

    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()


def predict_npk_sheath_model(x_new):
    load_model = joblib.load(filename='npk_sheath_model.sav')
    y_new = load_model.predict(x_new)
    return y_new


def train_pk_sheath_model():
    pk_sheath_data = pd.read_csv("D:/leafimages/PK_sheath_features.csv")

    pk_sheath_data.sample(frac=1).reset_index(drop=True)  # shuffle data

    X = pk_sheath_data.drop('CLASS', axis=1)
    Y = pk_sheath_data['CLASS']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
    ns_probs = [0 for _ in range(len(Y_test))]
    logisticreg_classifier = LogisticRegression(random_state=0).fit(X_train, Y_train)

    logistic_reg_prediction = logisticreg_classifier.predict(X_test)

    accuracy = logisticreg_classifier.score(X_test, Y_test)
    print(accuracy)  # debug

    # creating a confusion matrix
    cm = confusion_matrix(Y_test, logistic_reg_prediction)

    print(cm)
    print(classification_report(Y_test, logistic_reg_prediction))

    # predict probabilities
    lr_probs = logisticreg_classifier.predict_proba(X_test)
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
    saved_mdl_path = 'pk_sheath_model.sav'
    joblib.dump(logisticreg_classifier, saved_mdl_path)

    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()


def predict_pk_sheath_model(x_new):
    load_model = joblib.load(filename='pk_sheath_model.sav')
    y_new = load_model.predict(x_new)
    return y_new


if __name__ == "__main__":
    main()
