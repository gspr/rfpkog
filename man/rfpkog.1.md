% RFPKOG(1) | Really Fast Persistence Kernels On GPUs

NAME
====

**rfpkog** — Really Fast Persistence Kernels On GPUs

SYNOPSIS
========

| **rfpkog** \[**options**] <**file_1**> <**file_2**>
| **rfpkog** \[**options**] <**file**> <**file**>

DESCRIPTION
===========

The two input files, **file_1** and **file_2**, are text files that
list the input persistence diagrams. Each line should contain the file
name of a persistence diagram (currently the only supported type is
the DIPHA format). If the first file has M PDs and the second has N,
RFPKOG will compute an MxN matrix consisting of the kernel values of
each pair of diagrams. If the same list file is repeated twice,
i.e. the second invocation, then symmetry of the output is exploited.

RFPKOG will look for its OpenCL source code to be compiled for the GPU
device(s) in question in a location that was set during build. It can
be overridden by the environment variable *RFPKOG_KERNEL_FILE_NAME*,
for example in order to run the program without properly installing
it.

Options
-------

This part of the manpage is unfinished. Run **rfpkog --help** for
help, or see the *README.md* in the source tree.

ENVIRONMENT
===========

**RFPKOG_KERNEL_FILE_NAME**

:   Overrides location of OpenCL source.

BUGS
====

For now, report issues to <gspr@nonempty.org>.

AUTHOR
======

Gard Spreemann <gspr@nonempty.org>
