#!/usr/bin/env python3

import argparse
import numpy as np
import sys

import persistence_diagram

def main():
    parser = argparse.ArgumentParser(description="Generate random persistence diagrams.")
    parser.add_argument("--degree", action="store", type=int, default=0)
    parser.add_argument("-p", action="store", type=float, default=0.1)
    parser.add_argument("-N", action="store", type=int, required=True)
    parser.add_argument("output", metavar="output", type=str)

    args = parser.parse_args()

    if args.N <= 0:
        print("N must be positive.", file=sys.stderr)
        exit(1)

    if args.output is None:
        print("You must specify an output file.", file=sys.stderr)
        exit(1)

    pd = np.zeros((args.N, 2))
    pd[:, 0] = np.random.uniform(0, 1, args.N)
    pd[:, 1] = pd[:, 0] + np.random.uniform(0, 1, args.N)
    for i in range(0, args.N):
        if np.random.uniform(0, 1) < args.p:
            pd[i, 1] = np.inf

    x = []
    while len(x) <= args.degree:
        x.append([])
    x[args.degree] = pd
    
    persistence_diagram.save(args.output, x)
   

if __name__ == "__main__":
    main()
