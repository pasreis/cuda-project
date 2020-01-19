
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

float* fillVectorRandomly(float* v, int size) {
	for (int i = 0; i < size; ++i) {
		v[i] = rand() / (float) RAND_MAX;
	}

	return v;
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

int testAddVector() {
	printf("Testing addVector with values:\n");

	int size = 2<<10;

	float* a = (float*) malloc(size * sizeof(float));
	float* b = (float*) malloc(size * sizeof(float));
	float* c = (float*) malloc(size * sizeof(float));
	float* c_expected = (float*) malloc(size * sizeof(float));

	a = fillVectorRandomly(a, size);
	b = fillVectorRandomly(b, size);

	addVectorCPU(a, b, c_expected, size);

	printf("A:\n");
	printVector(a, size);
	printf("B:\n");
	printVector(b, size);

	printf("Vectors size: %d\n", size);

	addVector(a, b, c, size);

	for (int i = 0; i < size; ++i) {
		if (fabs(c_expected[i] - c[i]) > 1e-5) {
			printf("ERROR: addVector, expected %f, but got %f instead!\n", c_expected[i], c[i]);
			return FAILED;
		}
	}

	printf("Testing addVector with NULL vector A\n");
	if (addVector(NULL, b, c, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	printf("Testing addVector with NULL vector B\n");
	if (addVector(a, NULL, c, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	printf("Testing addVector with NULL vector C\n");
	if (addVector(a, b, NULL, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	printf("Testing addVector with negative size\n");
	if (addVector(a, b, c, -1) != ERROR) return FAILED;


	return PASSED;
}

int testSubVector() {
	printf("Testing subVector with values:\n");

	int size = 2<<10;

	float* a = (float*) malloc(size * sizeof(float));
	float* b = (float*) malloc(size * sizeof(float));
	float* c = (float*) malloc(size * sizeof(float));
	float* c_expected = (float*) malloc(size * sizeof(float));

	a = fillVectorRandomly(a, size);
	b = fillVectorRandomly(b, size);

	subVectorCPU(a, b, c_expected, size);

	printf("A:\n");
	//printVector(a, size);
	printf("B:\n");
	//printVector(b, size);

	printf("Vectors size: %d\n", size);

	subVector(a, b, c, size);

	for (int i = 0; i < size; ++i) {
		if (fabs(c_expected[i] - c[i]) > 1e-5) {
			printf("ERROR: subVector, expected %f, but got %f instead!\n", c_expected[i], c[i]);
			return FAILED;
		}
	}

	printf("Testing subVector with NULL vector A\n");
	if (subVector(NULL, b, c, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	printf("Testing subVector with NULL vector B\n");
	if (subVector(a, NULL, c, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	printf("Testing subVector with NULL vector C\n");
	if (subVector(a, b, NULL, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	printf("Testing subVector with negative size\n");
	if (subVector(a, b, c, -1) != ERROR) return FAILED;


	return PASSED;
}
/**
 * This main function is responsible for running the necessary tests
 * to ensure the correctness of the solution
 */
int main(int argc, char **argv)
{
	if (testAddVectorCPU() == PASSED) printf("TEST PASSED!\n"); else printf("TEST FAILED!\n");
	if (testSubVectorCPU() == PASSED) printf("TEST PASSED!\n"); else printf("TEST FAILED!\n");
	if (testAddMatrixCPU() == PASSED) printf("TEST PASSED!\n"); else printf("TEST FAILED!\n");
	if (testSubMatrixCPU() == PASSED) printf("TEST PASSED!\n"); else printf("TEST FAILED!\n");
	if (testMulMatrixCPU() == PASSED) printf("TEST PASSED!\n"); else printf("TEST FAILED!\n");
	if (testDotProductCPU() == PASSED) printf("TEST PASSED!\n"); else printf("TEST FAILED!\n");

	if (testAddVector() == PASSED) printf("TEST PASSED!\n"); else printf("TEST FAILED!\n");
	if (testSubVector() == PASSED) printf("TEST PASSED!\n"); else printf("TEST FAILED!\n");
	return 0;
}
