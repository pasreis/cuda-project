
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

__host__
int addVectorCPU(float* a, float* b, float* c, int size) {
	if (a == NULL || b == NULL || c == NULL || size < 0) {
		printf("ERROR: addVectorCPU invalid values\n");
		return ERROR;
	}

	for (int i = 0; i < size; ++i) {
		c[i] = a[i] + b[i];
	}

	return SUCCESS;
}

__host__
int subVectorCPU(float* a, float* b, float* c, int size) {
	if (a == NULL || b == NULL || c == NULL || size < 0) {
		printf("ERROR: subVectorCPU invalid values\n");
		return ERROR;
	}

	for (int i = 0; i < size; ++i) {
		c[i] = a[i] - b[i];
	}

	return SUCCESS;
}

__host__
int addMatrixCPU(float* a, float* b, float* c, int rows, int cols) {
	if (a == NULL || b == NULL || c == NULL || rows < 0 || cols < 0) {
		printf("ERROR: addMAtrixCPU invalid values\n");
		return ERROR;
	}

	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			c[row * cols + col] = a[row * cols + col] + b[row * cols + col];
		}
	}

	return SUCCESS;
}

__host__
int subMatrixCPU(float* a, float* b, float* c, int rows, int cols) {
	if (a == NULL || b == NULL || c == NULL || rows < 0 || cols < 0) {
		printf("ERROR: addMAtrixCPU invalid values\n");
		return ERROR;
	}

	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			c[row * cols + col] = a[row * cols + col] - b[row * cols + col];
		}
	}

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

	for (int i = 0; i < rowsA; ++i) {
		for (int j = 0; j < colsB; ++j) {
			float tmp = 0.0f;
			for (int k = 0; k < colsA; ++k) {
				tmp += a[i * colsA + k] * b[k * colsB + j];
			}
			c[i * colsC + j] = tmp;
		}
	}

	return SUCCESS;
}

__host__
int dotProductCPU(float* a, float* b, float* c, int size) {
	if (a == NULL || b == NULL || c == NULL || size < 0) {
		printf("ERROR: dotProductCPU invalid values\n");
		return ERROR;
	}

	float result = 0.0f;

	for (int i = 0; i < size; ++i) {
		result += a[i] * b[i];
	}

	c[0] = result;

	return SUCCESS;
}
