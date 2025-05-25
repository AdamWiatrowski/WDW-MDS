import argparse
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.metrics import pairwise_distances
from numpy.linalg import eigh
from scipy.sparse.linalg import eigsh

def parse_args():
    parser = argparse.ArgumentParser(
        description="MDS implementation for WDW (Wizualizacja Danych Wielowymiarowych)",
        epilog="Author: Adam Wiatrowski 148510 adam.wiatrowski@student.put.poznan.pl"
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="path to the input text file"
    )

    parser.add_argument(
        "-m", "--metric",
        dest="metric",
        choices=["euclidean", "manhattan", "chebyshev", "minkowski"],
        default="euclidean",
        help="distance metric to use when computing distance matrix"
    ) 

    parser.add_argument(
        "-d", "--dim",
        dest="dim",
        type=int,
        default=2,
        help="target dimension"
    )

    parser.add_argument(
        "-o", "--output_file",
        dest="output",
        type=str,
        default="result.png",
        help="output file of result"
    )



    return parser.parse_args()

def read_input(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    header = lines[0].split()
    mode = header[0].upper()
    col_ids = header[1:] # objor attr

    obj_labels = []
    numeric = []

    for line in lines[1:]:
        parts = line.split()
        obj_labels.append(parts[0])
        numeric.append([float(x) for x in parts[1:]])

    arr = np.array(numeric)
    if mode == 'X':
        return mode, obj_labels, arr
    elif mode == 'D':
        return mode, obj_labels, arr

def distance_matrix(X, metric='euclidean', p=3):
    if metric == 'minkowski':
        return cdist(X, X, metric=metric, p=p)
    else:
        return cdist(X, X, metric=metric)


#https://en.wikipedia.org/wiki/Stress_majorization#The_SMACOF_algorithm


# def compute_mds():
#     # todo
#     return


    # The SMACOF algorithm for metric MDS can be summarized by the following
    # steps:

    # 1. Set an initial start configuration, randomly or not.
    # 2. Compute the stress
    # 3. Compute the Guttman Transform
    # 4. Iterate 2 and 3 until convergence.
# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.smacof.html#sklearn.manifold.smacof

def main():
    args = parse_args()

    mode, labels, data = read_input(file_path=args.input_file)

    if mode == "X":
        D = distance_matrix(data, metric=args.metric)
    else:
        D = data
    
    # cords = compute_mds(D, dim=args.dim) #solver=
    # print(cords)
    #Q = compute_quality(D, coords)
    #plot_and_save()
    return


if __name__ == "__main__":
    main()
