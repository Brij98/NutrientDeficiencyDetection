import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def main():
    leaf_data = pd.read_csv("D:/leafimages/N_vs_PK_leaf_features.csv")

    #leaf_data.sample(frac=1) .reset_index(drop=True)  # shuffle data

    X = leaf_data.drop('CLASS', axis=1)
    Y = leaf_data['CLASS']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10)

    svm_model_linear = SVC(kernel='linear', C=1).fit(X_train, Y_train)
    svm_prediction = svm_model_linear.predict(X_test)

    accuracy = svm_model_linear.score(X_test, Y_test)
    print(accuracy)

    cm = confusion_matrix(Y_test, svm_prediction)

    print(cm)
    print(classification_report(Y_test, svm_prediction))


if __name__ == "__main__":
    main()
