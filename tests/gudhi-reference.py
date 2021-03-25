#!/usr/bin/env python3

import argparse
import gudhi.representations.kernel_methods
import numpy as np
import sys

import persistence_diagram

def main():
    parser = argparse.ArgumentParser(description="Validation of computations against GUDHI's.")
    parser.add_argument("--degree", action="store", type=int, required=True)
    parser.add_argument("-64", "--double", action="store_true")
    parser.add_argument("-s", "--sigma", action="store", type=float, required=True)
    parser.add_argument("-f", "--finitization", action="store", type=float, required=True)
    parser.add_argument("-o", "--output", action="store", type=str, default="-")
    parser.add_argument("file_1", metavar="file_1", type=str)
    parser.add_argument("file_2", metavar="file_2", type=str)

    args = parser.parse_args()

    if args.sigma <= 0 or not np.isfinite(args.sigma):
        print("Sigma must be positive and finite.", file=sys.stderr)
        exit(1)

    if args.degree < 0:
        print("Degree must be non-negative.", file=sys.stderr)
        exit(1)

    if not np.isfinite(args.finitization):
        print("Finitization must be finite.", file=sys.stderr)
        exit(1)

    fnames = [[], []]
    with open(args.file_1, "r") as f:
        fnames[0] = [line.rstrip() for line in f if line.rstrip() != ""]
    with open(args.file_2, "r") as f:
        fnames[1] = [line.rstrip() for line in f if line.rstrip() != ""]

    if args.file_1 == args.file_2:
        assert(fnames[0] == fnames[1])
        symmetric = True
    else:
        symmetric = False

    X = np.zeros((len(fnames[0]), len(fnames[1])))
    PSSK = gudhi.representations.kernel_methods.PersistenceScaleSpaceKernel(bandwidth=2*np.sqrt(args.sigma))

    pds = [None, None]
    prev_fnames = [None, None]
    for i in range(0, X.shape[0]):
        if fnames[0][i] != prev_fnames[0]:
            pds[0] = persistence_diagram.load(fnames[0][i], args.finitization)[args.degree]
            prev_fnames[0] = fnames[0][i]
        if symmetric:
            for j in range(i, X.shape[1]):
                if fnames[1][j] != prev_fnames[1]:
                    pds[1] = persistence_diagram.load(fnames[1][j], args.finitization)[args.degree]
                    prev_fnames[1] = fnames[1][j]
                X[i, j] = PSSK(pds[0], pds[1])/(2*np.sqrt(2*np.pi*args.sigma))
                X[j, i] = X[i, j]
        else:
            for j in range(0, X.shape[1]):
                if fnames[1][j] != prev_fnames[j]:
                    pds[1] = persistence_diagram.load(fnames[1][j])[args.degree]
                    prev_fnames[1] = fnames[1][j]
                X[i, j] = PSSK(pds[0], pds[1])/(2*np.sqrt(2*np.pi*args.sigma))

    f = sys.stdout if args.output == "-" else open(args.output, "w")
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            f.write("{:f} ".format(X[i, j]))
        f.write("\n")
    f.close()

if __name__ == "__main__":
    main()
