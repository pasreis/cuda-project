# CUDA Course Project: cuda-project
This project consists in a library named cuda-project that permits CUDA developers use functions that perform the following operations:
  - Addition of two vectors
  - Subtraction of two vectors
  - Multiplication of two matrices
  - Addition of two matrices
  - Subtration of two matrices
  - Dot Product between two vectors
  - Matrix Inversion
This project also includes a library that implements all the previous operations which is used for performance comparison. This library is called cuda-project-cpu and includes also the following operations which help the developers in case of bugs:
  - Print vectors
  - Print matrix

## How to compile the code?
A Make file is provided with the code. To compile
```
$ make
```

To remove all the files generated during compile time:
````
$ make clean
````

After compilation a file named cuda-project will be generated. This file is used for performance comparison between the functions implemented using CUDA and therefore the GPU and their implementation using CPU only. 
To run this code:
```
$ ./cuda-project <options> <arguments>
```
### Options
```-a```: vector addition: This takes **1 argument**: size of both vectors

```-b```: vector subtraction: This takes **1 argument**: size of both vectors

```-c```: matrix multiplication: This takes **4 arguments**: rows of matrix A, columns of matrix A, rows of matrix B, columns of matrix B

```-d```: matrix addition: This takes **2 arguments**: rows of both matrices and columns of both matrices

```-e```: matrix subtraction: This takes **2 arguments**: rows of both matrices and columns of both matrices

```-f```: dot product: This takes **1 argument**: size of both vectors

```-g```: matrix inversion: this takes **2 arguments**: rows of the matrix and columns of the matrix
