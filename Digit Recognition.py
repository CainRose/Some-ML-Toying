import numpy as np
import os
from sklearn import svm

__author__ = 'Lukas'
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


def main():
    Xtrain, ytrain = read_svm_data(
        os.path.join(__location__, "Data\\train-01-images.dat"))
    XtrainW, ytrainW = read_svm_data(
        os.path.join(__location__, "Data\\train-01-images-W.dat"))
    Xtest, ytest = read_svm_data(
        os.path.join(__location__, "Data\\test-01-images.dat"))
    print("Data loaded")

    clf = svm.LinearSVC()
    clf.fit(Xtrain, ytrain)
    print("Training and testing error for correct data:")
    print(svm_test(clf, Xtrain, ytrain))
    print(svm_test(clf, Xtest, ytest))

    clfW = svm.LinearSVC(C=1e-8)
    clfW.fit(XtrainW, ytrainW)
    print("Training and testing error for mislabeled data:")
    print(svm_test(clfW, XtrainW, ytrainW))
    print(svm_test(clfW, Xtest, ytest))


# terribly inefficient - should probably be optimised
def read_svm_data(path):
    with open(path) as f:
        y, X = zip(*[(int(d[0]), [str.split(w[:], ':') for w in d[1:]])
                     for d in (str.split(l) for l in f)])
        X = [{int(d[0]): int(d[1]) for d in x} for x in X]
        nf = 750
        X = [[d.get(i, 0) for i in range(nf)] for d in X]

    return np.array(X), np.array(y)

def svm_test(clf, X_test, y_test):
    return np.sum(np.array(clf.predict(X_test)) == y_test) / len(y_test)

if __name__ == "__main__": main()
