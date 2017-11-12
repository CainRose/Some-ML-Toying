import numpy as np
import matplotlib.pyplot as plt
import os

__author__ = 'Lukas'
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


def main():
    # Data Import for Training Set A
    Xa = np.loadtxt(os.path.join(__location__, "Data\\p1_a_X.dat"))
    ya = np.loadtxt(os.path.join(__location__, "Data\\p1_a_y.dat"))
    (ta, ka) = perceptron_train(Xa, ya)
    # Angle from (1, 0), geometric margin, R
    aa = np.arccos(ta[0] / np.linalg.norm(ta))
    ga = np.min(np.abs(np.dot(Xa, ta) / np.linalg.norm(ta)))
    Ra = np.max(np.linalg.norm(Xa, axis=1))

    print("Training set A produces a classifer", np.round(ta, 2), "in", ka,
          "steps. \nThis has an angle of", round(aa, 3),
          "from the vector [1,0], a geometric margin of", round(ga, 3),
          ", and an R of", round(Ra, 3), ".")

    # Colour points depending on label
    ca = np.vectorize(lambda t: 'b' if t > 0 else 'r')(ya)
    plt.scatter(Xa[:, 0], Xa[:, 1], c=ca)

    # plot decision boundary
    ea = np.array([np.min(Xa[:, 0]), np.max(Xa[:, 0])])
    plt.plot(ea, -ea * ta[0] / ta[1], c='k')
    plt.show()

    # Data Import for Training Set B
    Xb = np.loadtxt(os.path.join(__location__, "Data\\p1_b_X.dat"))
    yb = np.loadtxt(os.path.join(__location__, "Data\\p1_b_y.dat"))
    (tb, kb) = perceptron_train(Xb, yb)
    # Angle from (1, 0), geometric margin, R
    ab = np.arccos(tb[0] / np.linalg.norm(tb))
    gb = np.min(np.abs(np.dot(Xb, tb) / np.linalg.norm(tb)))
    Rb = np.max(np.linalg.norm(Xb, axis=1))

    print("Training set B produces a classifer", np.round(tb, 2), "in", kb,
          "steps. \nThis has an angle of", round(ab, 3),
          "from the vector [1,0], a geometric margin of", round(gb, 3),
          ", and an R of", round(Rb, 3), ".")

    # Colour points depending on label
    cb = np.vectorize(lambda t: 'b' if t > 0 else 'r')(yb)
    plt.scatter(Xb[:, 0], Xb[:, 1], c=cb)

    # plot decision boundary
    eb = np.array([np.min(Xb[:, 0]), np.max(Xb[:, 0])])
    plt.plot(eb, -eb * tb[0] / tb[1], c='k')
    plt.show()


def perceptron_train(X, y):
    theta = np.zeros(len(X[0]))
    updated = True
    k = 0
    # ?' = ? + y_i * X_i, for arbitrary i, if y_i * ? . X_i <= 0
    while (updated):
        updated = False
        for i in range(len(X)):
            if y[i] * np.dot(theta, X[i]) <= 0:
                theta += y[i] * X[i]
                updated = True
                k += 1
    return (theta, k)


def perceptron_test(theta, X_test, y_test):
    return np.sum(y_test * np.dot(X_test, theta) > 0) / len(y)


if __name__ == "__main__": main()
