# Really Fast Persistence Kernels On GPUs

There is an increasing appetite for using the persistent homology of
data as features for machine learning. For support vector machines in
particular, there have been numerous definitions of *persistence
kernels* that allow SVMs to be trained on, and then to classify,
persistence diagrams. One notable example is the so-called [*heat
kernel*](https://doi.org/10.1109/CVPR.2015.7299106) of Reininghaus et
al.

A persistence kernel K evaluated on persistence diagrams X and Y tends
to have a simple definition, but one that may involve computing some
numerical value for each pair of points in the cartesian product
X×Y. If one needs to compute a lot of kernel values K(Xᵢ, Xⱼ), for
example if one has a lot of data objects to classify, and each of the
Xⱼ's are relatively large, one is in a bind.

One approach is to exploit that the problem is data-parallel across
the Xᵢ's and for example distribute the computation of the kernel
values across tens or hundreds of CPUs (do ask me if you want me to
make public an OpenMPI implementation that has been tested succesfully
on hundreds of CPUs).

Computing K(X, Y) is also a task that lends itself nicely to GPU
computations. And I wanted to finally get my hands dirty with GPU
programming. Hence RFPKOG.

See [https://nonempty.org/software/rfpkog/](https://nonempty.org/software/rfpkog/)
for more information.

## About

The [heat kernel](https://doi.org/10.1109/CVPR.2015.7299106) is
currently the only supported one. Support for other kernels will come.

The code is licensed under the GNU Lesser General Public License,
version 3.0.

### Normalization and GUDHI compatibility

**Important:** the normalization prefactor and the `bandwidth`
parameter in [GUDHI's CPU
implementation](https://gudhi.inria.fr/python/latest/representations.html#gudhi.representations.kernel_methods.PersistenceScaleSpaceKernel)
uses a different convention than the `sigma` parameter in
RFPKOG. RFPKOG follows the
[paper](https://doi.org/10.1109/CVPR.2015.7299106)'s
convention. Specifically, if K'_σ denotes GUDHI's kernel with
bandwidth parameter σ, then RFPKOG with parameter σ is
```
K_σ(X, Y) = K'_{2*sqrt(σ)}/(2*sqrt(2πσ))
```

## Installation

RFPKOG needs only:

 * A C++ compiler that supports C++17. C++11 will suffice with very
   small modifications to the code. Debian/Ubuntu package name `g++`.
 
 * OpenCL 1.2 or above. Typically dependent on your GPU hardware.
 
 * Meson for building. Debian/Ubuntu package name `meson`.
   - CMake still works, but will be removed.
 

To build and install:

```
meson setup -Doptimization=2 -Dprefix=/where/you/want/rfpkog/to/go build
cd build
meson compile
meson install 
```

You can also build and run RFPKOG without installing it. The location
of the OpenCL code (confusingly also called a "kernel"… can we stop
naming more things kernels already?) is hardcoded based on the
installation directory, but you can override it by setting the
environment variable `RFPKOG_KERNEL_FILE_NAME`.

## Running

Run with `--help` to see a help text. An example invocation is
```
rfpkog -d 1 --devices 0 --platform 0 -f 1.0 -s 1.0 list_1.txt list_2.txt
```

See `rfpkog --help` for more information.

Multiple GPUs are supported. The set of pairs to compute is
parallelized over the GPUs.

## Benchmarks

RFPKOG was informally benchmarked on randomly generated persistence
diagrams of various sizes. Comparisons with a straight-forward C++
implementation for CPUs, and with [GUDHI's CPU
implementation](https://gudhi.inria.fr/python/latest/representations.html#gudhi.representations.kernel_methods.PersistenceScaleSpaceKernel)
were made.

The following table refers to crude benchmarks for a single pair of
persistence diagrams, each one of the given size.

| Backend \\ PD sizes                           | 10k       | 100k      | 500k      |
|-----------------------------------------------|----------:|----------:|----------:|
| *CPU reference*                               | 1.0 s     | 104 s     | —         |
| *CPU reference* (double precision)            | 1.4 s     | 144 s     | —         |
| *GUDHI CPU reference* (double precision)      | 8.5 s     | —         | —         |
| Intel UHD Graphics 630                        | 0.1 s     | 11.4 s    | —         |
| Intel UHD Graphics 630 (double precision)     | 0.4 s     | 37.6      | —         |
| Nvidia GeForce GTX 1650                       | 0.02 s    | 1.2 s     | 30 s      |
| Nvidia GeForce GTX 1650 (double precision)    | 0.2 s     | 13.8 s    | —         |
| Nvidia Tesla T4                               | —         | 0.9 s     | 19 s      |
| Nvidia Tesla T4 (double precision)            | —         | 7.0 s     | —         |

The CPU benchmarks, in italics above, were run on an Intel i7-9750H
CPU. As a rough estimate, the current version provides a speedup
factor of 100 from that CPU to a decent GPU.

## To do

 * Allow splitting really big persistence diagrams across GPUs and
   into parts.
   
 * More careful selection of local work sizes.
   
 * Turn the code into a library.
 
 * ASCII input, and binary output.
 
 * Tests.
 
 * Better error handling.
 
 * Support for more operating systems.

 * Support for more kernels, like [persistence weighted Gaussian
   kernels](http://proceedings.mlr.press/v48/kusano16.html).

 * More robust and standard options parsing.

### Maybe to do

 * Combine with the aforementioned MPI support to use *a lot* of GPUs.


