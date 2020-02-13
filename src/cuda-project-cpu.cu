
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

#include <cuda.h>

#include "cuda-project-cpu.h"

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 *
 * See cuda.h for error code descriptions.
 */
#define CHECK_CUDA_RESULT(N) {											\
	CUresult result = N;												\
	if (result != 0) {													\
		printf("CUDA call on line %d returned error %d\n", __LINE__,	\
			result);													\
		exit(1);														\
	} }

double timer() {
	struct timeval clock;
	gettimeofday(&clock, NULL);
	return ((double) clock.tv_sec + (double) clock.tv_usec * 1.e-6);
}

__host__
int addVectorCPU(float* a, float* b, float* c, int size) {
	if (a == NULL || b == NULL || c == NULL || size < 0) {
		printf("ERROR: addVectorCPU invalid values\n");
		return ERROR;
	}

	double begin = timer(), end;
	for (int i = 0; i < size; ++i) {
		c[i] = a[i] + b[i];
	}

	end = timer();
	printf("addVectorCPU() executed in %lf ms.\n", end - begin);

	return SUCCESS;
}

__host__
int subVectorCPU(float* a, float* b, float* c, int size) {
	if (a == NULL || b == NULL || c == NULL || size < 0) {
		printf("ERROR: subVectorCPU invalid values\n");
		return ERROR;
	}

	double begin = timer(), end;

	for (int i = 0; i < size; ++i) {
		c[i] = a[i] - b[i];
	}

	end = timer();
	printf("subVectorCPU() executed in %lf ms.\n", end - begin);

	return SUCCESS;
}

__host__
int addMatrixCPU(float* a, float* b, float* c, int rows, int cols) {
	if (a == NULL || b == NULL || c == NULL || rows < 0 || cols < 0) {
		printf("ERROR: addMAtrixCPU invalid values\n");
		return ERROR;
	}

	double begin = timer(), end;
	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			c[row * cols + col] = a[row * cols + col] + b[row * cols + col];
		}
	}

	end = timer();
	printf("addMatrixCPU() executed in %lf ms.\n", end - begin);

	return SUCCESS;
}

__host__
int subMatrixCPU(float* a, float* b, float* c, int rows, int cols) {
	if (a == NULL || b == NULL || c == NULL || rows < 0 || cols < 0) {
		printf("ERROR: addMAtrixCPU invalid values\n");
		return ERROR;
	}

	double begin = timer(), end;
	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			c[row * cols + col] = a[row * cols + col] - b[row * cols + col];
		}
	}

	end = timer();
	printf("subMatrixCPU() executed in %lf ms.\n", end -begin);

	return SUCCESS;
}

__host__
int mulMatrixCPU(float* a, float* b, float* c, int rowsA, int colsA, int rowsB, int colsB, int rowsC, int colsC) {
	if (a == NULL || b == NULL || c == NULL || rowsA < 0 || colsA < 0 || rowsB < 0 || colsB < 0 || rowsC < 0 || colsC < 0) {
		printf("ERROR: mulMatrixCPU invalid values\n");
		return ERROR;
	}

	if (colsA != rowsB) {
		printf("ERROR: mulMatrixCPU: is not possible to multiply the matrices because of their's dimensions\n");
		return ERROR;
	}

	if ((rowsA != rowsC) && (colsB != colsC)) {
		printf("ERROR: mulMatrixCPU: result matrix does not have the correct dimensions!\n");
		return ERROR;
	}

	double begin = timer(), end;
	for (int i = 0; i < rowsA; ++i) {
		for (int j = 0; j < colsB; ++j) {
			float tmp = 0.0f;
			for (int k = 0; k < colsA; ++k) {
				tmp += a[i * colsA + k] * b[k * colsB + j];
			}
			c[i * colsC + j] = tmp;
		}
	}

	end = timer();
	printf("mulMatrixCPU() executed in %lf ms.\n", end - begin);

	return SUCCESS;
}

__host__
int dotProductCPU(float* a, float* b, float* c, int size) {
	if (a == NULL || b == NULL || c == NULL || size < 0) {
		printf("ERROR: dotProductCPU invalid values\n");
		return ERROR;
	}

	float result = 0.0f;

	double begin = timer(), end;
	for (int i = 0; i < size; ++i) {
		result += a[i] * b[i];
	}

	c[0] = result;

	end = timer();
	printf("dotProductCPU() executed in %lf ms.\n", end - begin);

	return SUCCESS;
}

__host__
int inverseMatrixCPU(float* m, float* I, int rows, int cols) {
	if (m == NULL || I == NULL || rows < 0 || cols < 0) {
		printf("ERROR: inverseMatrixCPU invalid values\n");
		return ERROR;
	}

	double begin = timer(), end;
	for (int i = 0; i < rows; ++i) {
		for (int row = 0; row < rows; ++row) {
			for (int col = 0; col < cols; ++col) {
				if (row == i && row != col) {
					I[row * rows + col] /= m[i * rows + i];
					m[row * rows + col] /= m[i * rows + i];
				}
			}
		}

		for (int row = 0; row < rows; ++row) {
			for (int col = 0; col < cols; ++col) {
				if (row == col && row == i) {
					I[row * rows + col] = I[row * rows + col] / m[i * rows + i];
					m[row * rows + col] = m[row * rows + col] / m[i * row + i];
				}
			}
		}

		for (int row = 0; row < rows; ++row) {
			for (int col = 0; col < cols; ++col) {
				if (row != i) {
					I[row * rows + col] -= I[i * rows + col] * m[row * rows + i];

					if (col != i) {
						m[row * rows + col] -= m[i * rows + col] * m[row * rows + i];
					}
				}
			}
		}

		for (int row = 0; row < rows; ++row) {
			for (int col = 0; col < cols; ++col) {
				if (row != i && col == i) {
					m[row * rows + col] = 0;
				}
			}
		}
	}
	
	end = timer();
	printf("inverseMatrixCPU() executed in %lf ms.\n", end - begin);

	return SUCCESS;
}

