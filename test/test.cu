
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
#include <unistd.h>

#include <sys/time.h>

#include <cuda.h>

#include "../src/cuda-project.h"
#include "../src/cuda-project-cpu.h"

#define PASSED 1
#define FAILED 0

#define CPU_TEST_VECTOR_SIZE 3
#define CPU_TEST_MATRIX_NUM_ROWS 2
#define CPU_TEST_MATRIX_NUM_COLS 2
#define CPU_TEST_MATRIX_MUL_M 3
#define CPU_TEST_MATRIX_MUL_N 2
#define CPU_TEST_MATRIX_MUL_P 4
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

	
void fillVectorRandomly(float* v, int size) {
	for (int i = 0; i < size; ++i) {
		v[i] = rand() / (float) (RAND_MAX / 3);
	}
}

int testAddVectorCPU() {
	printf("Testing addVectorCPU with values:\n");

	float a[CPU_TEST_VECTOR_SIZE] = { 1.0f, 2.0f, 3.0f };
	float b[CPU_TEST_VECTOR_SIZE] = { 4.0f, 5.0f, 6.0f };
	float c[CPU_TEST_VECTOR_SIZE];
	float c_expected[CPU_TEST_VECTOR_SIZE] = { 5.0f, 7.0f, 9.0f };

	printf("A:\n");
	printVector(a, CPU_TEST_VECTOR_SIZE);
	printf("\nB:\n");
	printVector(b, CPU_TEST_VECTOR_SIZE);
	printf("\nSize: %d\n", CPU_TEST_VECTOR_SIZE);

	if (addVectorCPU(a, b, c, CPU_TEST_VECTOR_SIZE) != SUCCESS) return FAILED;

	for (int i = 0; i < CPU_TEST_VECTOR_SIZE; ++i) {
		if (c[i] != c_expected[i]) {
			printf("ERROR: addVector CPU, expected %f but got %f instead!\n", c_expected[i], c[i]);
			return FAILED;
		}
	}

	printf("Testing addVectorCPU with NULL vector A\n");
	if (addVectorCPU(NULL, b, c, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	printf("Testing addVectorCPU with NULL vector B\n");
	if (addVectorCPU(a, NULL, c, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	printf("Testing addVectorCPU with NULL vector C\n");
	if (addVectorCPU(a, b, NULL, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	printf("Testing addVectorCPU with negative size\n");
	if (addVectorCPU(a, b, c, -1) != ERROR) return FAILED;

	return PASSED;
}

int testSubVectorCPU() {
	printf("Testing subVectorCPU with values:\n");

	float a[CPU_TEST_VECTOR_SIZE] = { 1.0f, 2.0f, 3.0f };
	float b[CPU_TEST_VECTOR_SIZE] = { 4.0f, 5.0f, 6.0f };
	float c[CPU_TEST_VECTOR_SIZE];
	float c_expected[CPU_TEST_VECTOR_SIZE] = { -3.0f, -3.0f, -3.0f };

	printf("A:\n");
	printVector(a, CPU_TEST_VECTOR_SIZE);
	printf("\nB:\n");
	printVector(b, CPU_TEST_VECTOR_SIZE);
	printf("\nSize: %d\n", CPU_TEST_VECTOR_SIZE);

	if (subVectorCPU(a, b, c, CPU_TEST_VECTOR_SIZE) != SUCCESS) return FAILED;

	for (int i = 0; i < CPU_TEST_VECTOR_SIZE; ++i) {
		if (c[i] != c_expected[i]) {
			printf("ERROR: subVectorCPU, expected %f but got %f instead!\n", c_expected[i], c[i]);
			return FAILED;
		}
	}

	printf("Testing subVectorCPU with NULL vector A\n");
	if (subVectorCPU(NULL, b, c, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	printf("Testing subVectorCPU with NULL vector B\n");
	if (subVectorCPU(a, NULL, c, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	printf("Testing subVectorCPU with NULL vector C\n");
	if (subVectorCPU(a, b, NULL, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	printf("Testing subVectorCPU with negative size\n");
	if (subVectorCPU(a, b, c, -1) != ERROR) return FAILED;

	return PASSED;
}

int testAddMatrixCPU() {
	printf("Testing addMatrixCPU with values:\n");

	float a[CPU_TEST_MATRIX_NUM_ROWS * CPU_TEST_MATRIX_NUM_COLS] = { 1.0f, 2.0f, 3.0f, 4.0f };
	float b[CPU_TEST_MATRIX_NUM_ROWS * CPU_TEST_MATRIX_NUM_COLS] = { 5.0f, 6.0f, 7.0f, 8.0f };
	float c[CPU_TEST_MATRIX_NUM_ROWS * CPU_TEST_MATRIX_NUM_COLS];
	float c_expected[CPU_TEST_MATRIX_NUM_ROWS * CPU_TEST_MATRIX_NUM_COLS] = { 6.0f, 8.0f, 10.0f, 12.0f };

	printf("A:\n");
	printMatrix(a, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS);
	printf("B:\n");
	printMatrix(b, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS);
	printf("Number of rows: %d\n", CPU_TEST_MATRIX_NUM_ROWS);
	printf("Number of columns: %d\n", CPU_TEST_MATRIX_NUM_COLS);

	if (addMatrixCPU(a, b, c, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != SUCCESS) return FAILED;

	for (int i = 0; i < CPU_TEST_MATRIX_NUM_ROWS; ++i) {
		for (int j = 0; j < CPU_TEST_MATRIX_NUM_COLS; ++j) {
			if (c[i * CPU_TEST_MATRIX_NUM_COLS + j] != c_expected[i * CPU_TEST_MATRIX_NUM_COLS + j]) {
				printf("ERROR: addMatrixCPU, expected %f but got %f instead!\n", c_expected[i * CPU_TEST_MATRIX_NUM_COLS + j], c[i * CPU_TEST_MATRIX_NUM_COLS + j]);
				return FAILED;
			}
		}
	}


	printf("Testing subVectorCPU with NULL vector A\n");
	if (addMatrixCPU(NULL, b, c, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != ERROR) return FAILED;

	printf("Testing subVectorCPU with NULL vector B\n");
	if (addMatrixCPU(a, NULL, c, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != ERROR) return FAILED;

	printf("Testing subVectorCPU with NULL vector C\n");
	if (addMatrixCPU(a, b, NULL, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != ERROR) return FAILED;

	printf("Testing addMatrixCPU with negative number of rows\n");
	if (addMatrixCPU(a, b, c, -1, CPU_TEST_MATRIX_NUM_COLS) != ERROR) return FAILED;

	printf("Testing addMatrixCPU with negative number of columns\n");
	if (addMatrixCPU(a, b, c, CPU_TEST_MATRIX_NUM_ROWS, -1) != ERROR) return FAILED;

	return PASSED;
}

int testSubMatrixCPU() {
	printf("Testing subMatrixCPU with values:\n");

	float a[CPU_TEST_MATRIX_NUM_ROWS * CPU_TEST_MATRIX_NUM_COLS] = { 1.0f, 2.0f, 3.0f, 4.0f };
	float b[CPU_TEST_MATRIX_NUM_ROWS * CPU_TEST_MATRIX_NUM_COLS] = { 5.0f, 6.0f, 7.0f, 8.0f };
	float c[CPU_TEST_MATRIX_NUM_ROWS * CPU_TEST_MATRIX_NUM_COLS];
	float c_expected[CPU_TEST_MATRIX_NUM_ROWS * CPU_TEST_MATRIX_NUM_COLS] = { -4.0f, -4.0f, -4.0f, -4.0f };

	printf("A:\n");
	printMatrix(a, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS);
	printf("B:\n");
	printMatrix(b, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS);
	printf("Number of rows: %d\n", CPU_TEST_MATRIX_NUM_ROWS);
	printf("Number of columns: %d\n", CPU_TEST_MATRIX_NUM_COLS);

	if (subMatrixCPU(a, b, c, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != SUCCESS) return FAILED;

	for (int i = 0; i < CPU_TEST_MATRIX_NUM_ROWS; ++i) {
		for (int j = 0; j < CPU_TEST_MATRIX_NUM_COLS; ++j) {
			if (c[i * CPU_TEST_MATRIX_NUM_COLS + j] != c_expected[i * CPU_TEST_MATRIX_NUM_COLS + j]) {
				printf("ERROR: addMatrixCPU, expected %f but got %f instead!\n", c_expected[i * CPU_TEST_MATRIX_NUM_COLS + j], c[i * CPU_TEST_MATRIX_NUM_COLS + j]);
				return FAILED;
			}
		}
	}


	printf("Testing subVectorCPU with NULL vector A\n");
	if (subMatrixCPU(NULL, b, c, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != ERROR) return FAILED;

	printf("Testing subVectorCPU with NULL vector B\n");
	if (subMatrixCPU(a, NULL, c, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != ERROR) return FAILED;

	printf("Testing subVectorCPU with NULL vector C\n");
	if (subMatrixCPU(a, b, NULL, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != ERROR) return FAILED;

	printf("Testing addMatrixCPU with negative number of rows\n");
	if (subMatrixCPU(a, b, c, -1, CPU_TEST_MATRIX_NUM_COLS) != ERROR) return FAILED;

	printf("Testing addMatrixCPU with negative number of columns\n");
	if (subMatrixCPU(a, b, c, CPU_TEST_MATRIX_NUM_ROWS, -1) != ERROR) return FAILED;

	return PASSED;
}

int testMulMatrixCPU() {
	printf("Testing mulMatrixCPU with values:\n");

	float a[CPU_TEST_MATRIX_NUM_ROWS * CPU_TEST_MATRIX_NUM_COLS] = { 1.0f, 2.0f, 3.0f, 4.0f };
	float b[CPU_TEST_MATRIX_NUM_ROWS * CPU_TEST_MATRIX_NUM_COLS] = { 5.0f, 6.0f, 7.0f, 8.0f };
	float c[CPU_TEST_MATRIX_NUM_ROWS * CPU_TEST_MATRIX_NUM_COLS];
	float c_expected[CPU_TEST_MATRIX_NUM_ROWS * CPU_TEST_MATRIX_NUM_COLS] = { 19.0f, 22.0f, 43.0f, 50.0f };

	printf("A:\n");
	printMatrix(a, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS);
	printf("B:\n");
	printMatrix(b, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS);
	printf("Number of rows: %d\n", CPU_TEST_MATRIX_NUM_ROWS);
	printf("Number of columns: %d\n", CPU_TEST_MATRIX_NUM_COLS);

	if (mulMatrixCPU(a, b, c, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS,CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != SUCCESS)
		return FAILED;

	for (int i = 0; i < CPU_TEST_MATRIX_NUM_ROWS; ++i) {
		for (int j = 0; j < CPU_TEST_MATRIX_NUM_COLS; ++j) {
			if(c[i * CPU_TEST_MATRIX_NUM_COLS + j] != c_expected[i * CPU_TEST_MATRIX_NUM_COLS + j]) {
				printf("ERROR: mulMatrixCPU, expected %f but got %f instead!\n", c_expected[i * CPU_TEST_MATRIX_NUM_ROWS + j], c[i * CPU_TEST_MATRIX_NUM_COLS + j]);
				return FAILED;
			}
		}
	}
	printf("Testing mulMatrixCPU with NULL matrix A\n");
	if (mulMatrixCPU(NULL, b, c, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS,CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != ERROR)
		return FAILED;

	printf("Testing mulMatrixCPU with NULL matrix B\n");
	if (mulMatrixCPU(a, NULL, c, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS,CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != ERROR)
			return FAILED;

	printf("Testing mulMatrixCPU with NULL matrix C\n");
	if (mulMatrixCPU(a, b, NULL, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS,CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != ERROR)
			return FAILED;

	printf("Testing mulMatrixCPU with negative number of rows for matrix A\n");
	if (mulMatrixCPU(a, b, c, -1, CPU_TEST_MATRIX_NUM_COLS, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS,CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != ERROR)
			return FAILED;

	printf("Testing mulMatrixCPU with negative number of columns for matrix A\n");
	if (mulMatrixCPU(a, b, c, CPU_TEST_MATRIX_NUM_ROWS, -1, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS,CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != ERROR)
			return FAILED;

	printf("Testing mulMatrixCPU with negative number of rows for matrix B\n");
	if (mulMatrixCPU(a, b, c, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS, -1, CPU_TEST_MATRIX_NUM_COLS,CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != ERROR)
			return FAILED;

	printf("Testing mulMatrixCPU with negative number of columns for matrix B\n");
	if (mulMatrixCPU(a, b, c, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS, CPU_TEST_MATRIX_NUM_ROWS, -1,CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != ERROR)
			return FAILED;

	printf("Testing mulMatrixCPU with negative number of rows for matrix C\n");
	if (mulMatrixCPU(NULL, b, c, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS,-1, CPU_TEST_MATRIX_NUM_COLS) != ERROR)
			return FAILED;

	printf("Testing mulMatrixCPU with negative number of columns for matrix C\n");
	if (mulMatrixCPU(NULL, b, c, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS,CPU_TEST_MATRIX_NUM_ROWS, -1) != ERROR)
			return FAILED;

	printf("Testing mulMatrixCPU with incompatible matrices\n");
	if (mulMatrixCPU(NULL, b, c, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS, 3, CPU_TEST_MATRIX_NUM_COLS,CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != ERROR)
			return FAILED;

	printf("Testing mulMatrixCPU with values:\n");

	float m1[CPU_TEST_MATRIX_MUL_M * CPU_TEST_MATRIX_MUL_N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	float m2[CPU_TEST_MATRIX_MUL_N * CPU_TEST_MATRIX_MUL_P] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f};
	float res[CPU_TEST_MATRIX_MUL_M * CPU_TEST_MATRIX_MUL_P];
	float expected_res[CPU_TEST_MATRIX_MUL_M * CPU_TEST_MATRIX_MUL_P] = {29.0f, 32.0f, 35.0f, 38.0f, 65.0f, 72.0f, 79.0f, 86.0f, 101.0f, 112.0f, 123.0f, 134.0f};

	printf("A:\n");
	printMatrix(m1, CPU_TEST_MATRIX_MUL_M, CPU_TEST_MATRIX_MUL_N);
	printf("B:\n");
	printMatrix(m2, CPU_TEST_MATRIX_MUL_N, CPU_TEST_MATRIX_MUL_P);
	printf("Number of rows of matrix A: %d\n", CPU_TEST_MATRIX_MUL_M);
	printf("Number of columns of matrix A: %d\n", CPU_TEST_MATRIX_MUL_N);
	printf("Number of rows of matrix B: %d\n", CPU_TEST_MATRIX_MUL_N);
	printf("Number of columns of matrix B: %d\n", CPU_TEST_MATRIX_MUL_P);

	if (mulMatrixCPU(m1, m2, res, CPU_TEST_MATRIX_MUL_M, CPU_TEST_MATRIX_MUL_N, CPU_TEST_MATRIX_MUL_N, CPU_TEST_MATRIX_MUL_P, CPU_TEST_MATRIX_MUL_M, CPU_TEST_MATRIX_MUL_P) != SUCCESS)
		return FAILED;

	for (int i = 0; i < CPU_TEST_MATRIX_MUL_M; ++i) {
		for (int j = 0; j < CPU_TEST_MATRIX_MUL_P; ++j) {
			if (expected_res[i * CPU_TEST_MATRIX_MUL_M + j] != res[i * CPU_TEST_MATRIX_MUL_M + j]) {
				printf("ERROR: mulMatrixCPU, expected %f but got %f instead!\n");
				return FAILED;
			}
		}
	}

	return PASSED;
}

int testDotProductCPU() {
	printf("Testing dotProductCPU with values:\n");

	float a[CPU_TEST_VECTOR_SIZE] = {1.0f, 2.0f, 3.0f};
	float b[CPU_TEST_VECTOR_SIZE] = {4.0f, 5.0f, 6.0f};
	float c[1];
	float c_expected = 32;

	printf("A:\n");
	printVector(a, CPU_TEST_VECTOR_SIZE);
	printf("B:\n");
	printVector(b, CPU_TEST_VECTOR_SIZE);
	printf("Vector size: %d\n", CPU_TEST_VECTOR_SIZE);

	if (dotProductCPU(a, b, c, CPU_TEST_VECTOR_SIZE) != SUCCESS) return FAILED;

	if (c[0] != c_expected) {
		printf("ERROR: dotProductCPU, expected %f, but got %f instead!\n", c_expected, c[0]);
		return FAILED;
	}

	printf("Testing dotProductCPU with NULL vector A\n");
	if (dotProductCPU(NULL, b, c, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	printf("Testing dotProductCPU with NULL vector B\n");
	if (dotProductCPU(a, NULL, c, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	printf("Testing dotProductCPU with NULL vector C\n");
	if (dotProductCPU(a, b, NULL, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	printf("Testing dotProductCPU with negative size\n");
	if (dotProductCPU(a, b, c, -1) != ERROR) return FAILED;

	return PASSED;
}

int testInverseMatrixCPU() {
	printf("Testing inverseMatrixCPU with values:\n");

	float m[4] = {1.0f, 1.0f, 1.0f, 3.0f};
	float I[4] = {1.0f, 0.0f, 0.0f, 1.0f};
	float expected[4] = { 1.5f, -0.5f, -0.5f, 0.5f};

	printf("M:\n");
	printMatrix(m, 2, 2);

	printf("I:\n");
	printMatrix(I, 2, 2);
	if (inverseMatrixCPU(m, I, 2, 2) != SUCCESS) return FAILED;

	printf("M^-1\n");
	printMatrix(I, 2, 2);

	for (int i  = 0; i < 4; ++i)  {
		if (I[i] != expected[i]) {
			printf("ERROR: inverseMatrixCPU, expected %f, but got %f instead!\n", expected[i], I[i]);
			return FAILED;
		}
	}

	printf("Testing inverseMatrixCPU with NULL vector A\n");
	if (inverseMatrixCPU(NULL, I, 2, 2) != ERROR) return FAILED;

	printf("Testing inverseMatrixCPU with NULL vector B\n");
	if (inverseMatrixCPU(m, NULL, 2, 2) != ERROR) return FAILED;

	printf("Testing inverseMatrixCPU with NULL vector C\n");
	if (inverseMatrixCPU(m, I, -2, 2) != ERROR) return FAILED;

	printf("Testing inverseMatrixCPU with negative size\n");
	if (inverseMatrixCPU(m, I, 2, -1) != ERROR) return FAILED;

	return PASSED;
}

int testAddVector(int numberOfElements) {
	printf("Testing addVector with values:\n");

	int size = numberOfElements * sizeof(float);

	float* a = (float*) malloc(size);
	float* b = (float*) malloc(size);
	float* c = (float*) malloc(size);
	float* c_expected = (float*) malloc(size * sizeof(float));

	fillVectorRandomly(a, numberOfElements);
	fillVectorRandomly(b, numberOfElements);

	addVectorCPU(a, b, c_expected, numberOfElements);

	addVector(a, b, c, numberOfElements);

	for (int i = 0; i < numberOfElements; ++i) {
		if (fabs(c_expected[i] - c[i]) > 1e-5) {
			printf("ERROR: addVector, expected %f, but got %f instead!\n", c_expected[i], c[i]);
			return FAILED;
		}
	}
	if (addVector(NULL, b, c, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	if (addVector(a, NULL, c, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	if (addVector(a, b, NULL, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	if (addVector(a, b, c, -1) != ERROR) return FAILED;


	return PASSED;
}

int testSubVector(int numberOfElements) {
	printf("Testing subVector with values:\n");

	int size = numberOfElements * sizeof(float);

	float* a = (float*) malloc(size);
	float* b = (float*) malloc(size);
	float* c = (float*) malloc(size);
	float* c_expected = (float*) malloc(size);

	fillVectorRandomly(a, numberOfElements);
	fillVectorRandomly(b, numberOfElements);

	subVectorCPU(a, b, c_expected, numberOfElements);

	subVector(a, b, c, numberOfElements);
	for (int i = 0; i < numberOfElements; ++i) {
		if (fabs(c_expected[i] - c[i]) > 1e-5) {
			return FAILED;
		}
	}

	if (subVector(NULL, b, c, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	if (subVector(a, NULL, c, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	if (subVector(a, b, NULL, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	if (subVector(a, b, c, -1) != ERROR) return FAILED;

	return PASSED;
}

int testMulMatrix(int rowsA, int colsA, int rowsB, int colsB) {
	printf("Testing mulMatrix with values:\n");
	
	int rowsC = rowsA;
	int colsC = colsB;

	int totalDimA = rowsA * colsA;
	int totalDimB = rowsB * colsB;
	int totalDimC = rowsC * colsC;

	float* a = (float*) malloc( totalDimA *  sizeof(float));
	float* b = (float*) malloc( totalDimB *  sizeof(float));
	float* c = (float*) malloc( totalDimC *  sizeof(float));
	float* c_expected = (float*) malloc( totalDimC *  sizeof(float));

	fillVectorRandomly(a, totalDimA);
	fillVectorRandomly(b, totalDimB);

	mulMatrixCPU(a, b, c_expected, rowsA, colsA, rowsB, colsB, rowsC, colsC);

	if (mulMatrix(a, b, c, rowsA, colsA, rowsB, colsB, rowsC, colsC) != SUCCESS) return FAILED;

	for (int i = 0; i < totalDimC; ++i) {
		if (fabs(c_expected[i] - c[i]) > 1.e-3) {
			printf("ERROR: mulMatrix, expected %f but got %f instead!\n", c_expected[i], c[i]);
			return FAILED;
		}
	}

	if (mulMatrix(NULL, b, c, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS,CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != ERROR)
		return FAILED;

	if (mulMatrix(a, NULL, c, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS,CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != ERROR)
			return FAILED;

	if (mulMatrix(a, b, NULL, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS,CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != ERROR)
			return FAILED;

	if (mulMatrix(a, b, c, -1, CPU_TEST_MATRIX_NUM_COLS, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS,CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != ERROR)
			return FAILED;

	if (mulMatrix(a, b, c, CPU_TEST_MATRIX_NUM_ROWS, -1, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS,CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != ERROR)
			return FAILED;

	if (mulMatrix(a, b, c, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS, -1, CPU_TEST_MATRIX_NUM_COLS,CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != ERROR)
			return FAILED;

	if (mulMatrix(a, b, c, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS, CPU_TEST_MATRIX_NUM_ROWS, -1,CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != ERROR)
			return FAILED;

	if (mulMatrix(NULL, b, c, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS,-1, CPU_TEST_MATRIX_NUM_COLS) != ERROR)
			return FAILED;

	if (mulMatrix(NULL, b, c, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS,CPU_TEST_MATRIX_NUM_ROWS, -1) != ERROR)
			return FAILED;

	if (mulMatrix(NULL, b, c, CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS, 3, CPU_TEST_MATRIX_NUM_COLS,CPU_TEST_MATRIX_NUM_ROWS, CPU_TEST_MATRIX_NUM_COLS) != ERROR)
			return FAILED;

	return PASSED;
}

int testAddMatrix(int rows, int cols) {
	printf("Testing addMatrix with values:\n");

	int totalDim = rows * cols;

	float* a = (float*) malloc( totalDim *  sizeof(float));
	float* b = (float*) malloc( totalDim *  sizeof(float));
	float* c = (float*) malloc( totalDim *  sizeof(float));
	float* c_expected = (float*) malloc( totalDim *  sizeof(float));

	fillVectorRandomly(a, totalDim);
	fillVectorRandomly(b, totalDim);

	addMatrixCPU(a, b, c_expected, rows, cols);

	if (addMatrix(a, b, c, rows, cols) != SUCCESS) return FAILED;

	for (int i = 0; i < totalDim; ++i) {
		if (fabs(c_expected[i] - c[i]) > 1e-5) {
			printf("ERROR: addMatrix, expected %f but got %f instead!\n", c_expected[i], c[i]);
			return FAILED;
		}
	}

	if (addMatrix(NULL, b, c, rows, cols) != ERROR)
		return FAILED;

	if (addMatrix(a, NULL, c, rows, cols) != ERROR)
			return FAILED;

	if (addMatrix(a, b, NULL, rows, cols) != ERROR)
			return FAILED;

	if (addMatrix(a, b, c, -1, cols) != ERROR)
			return FAILED;

	if (addMatrix(a, b, c, rows, -1) != ERROR)
			return FAILED;
	return PASSED;
}

int testSubMatrix(int rows, int cols) {
	printf("Testing subMatrix with values:\n");

	int totalDim = rows * cols;

	float* a = (float*) malloc( totalDim *  sizeof(float));
	float* b = (float*) malloc( totalDim *  sizeof(float));
	float* c = (float*) malloc( totalDim *  sizeof(float));
	float* c_expected = (float*) malloc( totalDim *  sizeof(float));

	fillVectorRandomly(a, totalDim);
	fillVectorRandomly(b, totalDim);

	subMatrixCPU(a, b, c_expected, rows, cols);
	if (subMatrix(a, b, c, rows, cols) != SUCCESS) return FAILED;

	for (int i = 0; i < totalDim; ++i) {
		if (fabs(c_expected[i] - c[i]) > 1e-5) {
			printf("ERROR: subMatrix, expected %f but got %f instead!\n", c_expected[i], c[i]);
			return FAILED;
		}
	}

	if (subMatrix(NULL, b, c, rows, cols) != ERROR) return FAILED;
	if (subMatrix(NULL, b, c, rows, cols) != ERROR) return FAILED;
	if (subMatrix(a, NULL, c, rows, cols) != ERROR)
			return FAILED;
	if (subMatrix(a, b, NULL, rows, cols) != ERROR)
			return FAILED;
	if (subMatrix(a, b, c, -1, cols) != ERROR)
			return FAILED;
	if (subMatrix(a, b, c, rows, -1) != ERROR)
			return FAILED;
	return PASSED;
}

int testDotProduct(int numberOfElements) {
	printf("Testing dotProduct with values:\n");

	float* a;
	float* b;
	float* c;
	float* c_expected;

	int size = numberOfElements * sizeof(float);

	a = (float*) malloc(size);
	b = (float*) malloc(size);
	c = (float*) malloc(size);
	c_expected = (float*) malloc(size);

	fillVectorRandomly(a, numberOfElements);
	fillVectorRandomly(b, numberOfElements);

	dotProductCPU(a, b, c_expected, numberOfElements);
	if (dotProduct(a, b, c, numberOfElements) != SUCCESS) return FAILED;

	for (int i = 0; i < numberOfElements; ++i) {
		if (fabs(c_expected[i] - c[i]) > 1e-5) {
			printf("ERROR: dotProduct, expected %f but got %f instead!\n", c_expected[i], c[i]);
			return FAILED;
		}
	}

	if (dotProduct(NULL, b, c, size) != ERROR)
		return FAILED;
	if (dotProduct(a, NULL, c, size) != ERROR)
			return FAILED;

	if (dotProduct(a, b, NULL, size) != ERROR)
			return FAILED;

	if (dotProduct(a, b, c, -1) != ERROR)
			return FAILED;

	return PASSED;
}

int testInverseMatrix(int matrixDimension) {
	printf("Testing inverse Matrix with matrix:\n");

	float m[4] = {1, 1, 1, 3};

	printMatrix(m, 2, 2);

	float inverse[4];
	float inverse_expected[4] = { 1.5f, -0.5f, -0.5f, 0.5f };

	if (inverseMatrix(m, inverse, 2,2 ) != SUCCESS) return FAILED;

	for (int i = 0; i < 4; ++i) {
		if (inverse_expected[i] != inverse[i]) {
			printf("ERROR: inverseMatrix, expected %f but got %f instead!\n", inverse_expected[i], inverse[i]);
			return FAILED;
		}
	}

	printf("Testing inverseMatrix with NULL matrix A\n");
	if (inverseMatrix(NULL, inverse, 2, 2) != ERROR)
		return FAILED;

	printf("Testing inverseMatrix with NULL inverse\n");
	if (inverseMatrix(m, NULL, 2, 2) != ERROR)
			return FAILED;

	printf("Testing inverseMatrix with negative number of columns\n");
	if (inverseMatrix(m, inverse, 2, -1) != ERROR)
			return FAILED;

	printf("Testing inverseMatrix with negative number of rows\n");
	if (inverseMatrix(m, inverse, -1 ,2) != ERROR)
			return FAILED;

	return PASSED;
}

/**
 * This main function is responsible for running the necessary tests
 * to ensure the correctness of the solution
 */
int main(int argc, char **argv)
{
	char option = getopt(argc, argv, "abcdefg");

	switch (option) {
		case 'a':
			if (argc != 3 ) {
				printf("ERROR: invalid number of arguments!\n");
				exit(EXIT_FAILURE);
			}

			if (testAddVector(atoi(argv[2])) == PASSED) printf("TEST PASSED!\n"); else printf("TEST FAILED!\n");

			break;
		case 'b':
			if (argc != 3) {
				printf("ERROR: invalid number of arguments!\n");
				exit(EXIT_FAILURE);
			}
			
			if (testSubVector(atoi(argv[2])) == PASSED) printf("TEST PASSED!\n"); else printf("TEST FAILED!\n");

			break;
		case 'c':
			if (argc != 6) {
				printf("ERROR: invalid number of arguments!\n");
				exit(EXIT_FAILURE);
			}

			if (testMulMatrix(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5])) == PASSED) printf("TEST PASSED!\n"); else printf("TEST FAILED!\n");

			break;
		case 'd':
			if (argc != 4) {
				printf("ERROR: invalid number of arguments!\n");
				exit(EXIT_FAILURE);
			}

			if (testAddMatrix(atoi(argv[2]), atoi(argv[3])) == PASSED) printf("TEST PASSED!\n"); else printf("TEST FAILED!\n");

			break;
		case 'e':
			if (argc != 4) {
				printf("ERROR: invalid number of arguments!\n");
				exit(EXIT_FAILURE);
			}

			if (testSubMatrix(atoi(argv[2]), atoi(argv[3])) == PASSED) printf("TEST PASSED!\n"); else printf("TEST FAILED!\n");

			break;
		case 'f':
			if (argc != 3) {
				printf("ERROR: invalid number of arguments!\n");
				exit(EXIT_FAILURE);
			}
	
			if (testDotProduct(atoi(argv[2])) == PASSED) printf("TEST PASSED!\n"); else printf("TEST FAILED!\n");
			
			break;
		case 'g':
			if (argc != 3) {
				printf("ERROR: invalid number of arguments!\n");
				exit(EXIT_FAILURE);
			}

			if (testInverseMatrix(2) == PASSED) printf("TEST PASSED!\n"); else printf("TEST FAILED!\n");

			break;
	}

	return 0;
}

