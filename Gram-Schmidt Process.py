__author__ = 'Lukas'

import numpy as np


def main():
    print(gram_schmidt_process(np.array([[0, 0, 0, 0, 0, 1],
                                         [1, 2, 3, 4, 5, 6],
                                         [1, 4, 9, 16, 25, 35],
                                         [1, 0, 0, 0, 0, 0]], dtype=float)).T)


# v is a list of vector, u is a basis of orthogonal unit vectors to v
def gram_schmidt_process(v):
    u = np.zeros_like(v)
    u[0] = v[0] / np.linalg.norm(v[0])
    for i in range(1, len(v)):
        # gram-schmidt process: vector i minus the projections of vector i
        # onto each of the previously constructed bases
        u[i] = v[i] - np.sum([np.dot(np.outer(u[j], v[i]), u[j])
                              for j in range(i)], axis=0)
        u[i] /= np.linalg.norm(u[i]) if u[i].any() else 1
    return u


if __name__ == "__main__": main()
