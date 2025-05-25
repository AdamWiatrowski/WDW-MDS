import argparse
import numpy as np
from scipy.spatial.distance import cdist
from numpy.linalg import eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def compute_stress(X, W, D):
    dist = cdist(X, X)
    return np.sum(W * (dist - D) ** 2)

def pcoa_initialization(D, dim=2):
    # https://en.wikipedia.org/wiki/Multidimensional_scaling#Classical_multidimensional_scaling :-)

    # 1. Set up the squared proximity matrix
    D_squared = np.power(D, 2)

    # 2. Apply double centering
    n = D.shape[0]
    identity_matrix = np.identity(n)
    J_n = np.ones_like(identity_matrix)

    C = identity_matrix - (1/n) * J_n

    B = -0.5 * (C @ D_squared @ C)

    # 3. Determine dim- largest eigenvalues, eigenvectors.
    eigvals, eigvecs = eigh(B)

    top_idx = np.argsort(eigvals)[::-1][:dim]
    m_vals, m_vecs = eigvals[top_idx], eigvecs[:, top_idx]

    # 4. X
    Lambda_m_sqrt = np.diag(np.sqrt(np.maximum(m_vals, 0)))
    X = m_vecs @ Lambda_m_sqrt

    return X

def smacof(D, dim=2, max_iter=300, eps=1e-8):

    # https://en.wikipedia.org/wiki/Stress_majorization#The_SMACOF_algorithm
    # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.smacof.html#sklearn.manifold.smacof
    n = D.shape[0]

    W = np.zeros_like(D)
    mask = D > 0
    W[mask] = 1.0 / D[mask]

    V = np.diag(W.sum(axis=1))

    X = pcoa_initialization(D, dim)

    last_stress = compute_stress(X, W, D)

    for it in range(1, max_iter + 1):
        dist = cdist(X, X)
        dist[dist == 0] = eps
        B = -W * (D / dist)
        np.fill_diagonal(B, -B.sum(axis=1))
        X = np.linalg.solve(V, B.dot(X))
        stress = compute_stress(X, W, D)
        if (last_stress - stress) / last_stress < eps:
            break
        last_stress = stress

    return X, last_stress

def compute_mds(D, dim=2, max_iter=300, eps=1e-8):
    X, stress = smacof(D, dim=dim, max_iter=max_iter, eps=eps)
    return X, stress

def compute_quality(D_true, X):
    D_embed = cdist(X, X)
    return np.linalg.norm(D_true - D_embed, ord='fro')

def plot_and_save(X, labels, output_file):
    plt.figure(figsize=(6,6))
    xs, ys = X[:,0], X[:,1]
    plt.scatter(xs, ys)
    for label, x, y in zip(labels, xs, ys):
        plt.text(x, y, label)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    plt.close()

def plot_and_save_3d(X, labels, output_file):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = X[:,0], X[:,1], X[:,2]
    ax.scatter(xs, ys, zs)
    for label, x, y, z in zip(labels, xs, ys, zs):
        ax.text(x, y, z, label)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    plt.close()

def main():
    args = parse_args()

    mode, labels, data = read_input(file_path=args.input_file)

    if mode == "X":
        D = distance_matrix(data, metric=args.metric)
    else:
        D = data


    X, stress = compute_mds(D, dim=args.dim)
    quality = compute_quality(D, X)

    print(f"Stress: {stress:.6f}")
    print(f"Norma Frobeniusa: {quality:.6f}")
    if args.dim == 2:
        plot_and_save(X, labels, args.output)
    if args.dim == 3:
        plot_and_save_3d(X, labels, args.output)
        
    return


if __name__ == "__main__":
    main()
