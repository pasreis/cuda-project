/*
 * cuda-project.h
 *
 *  Created on: Jan 12, 2020
 *      Author: cuda-s10
 */

#ifndef CUDA_PROJECT_H_
#define CUDA_PROJECT_H_

int addVector(float* a, float* b, float* c, int size);
int subVector(float* a, float* b, float* c, int size);
int addMatrix(float* a, float* b, float* c, int rows, int cols);
int subMatrix(float* a, float* b, float* c, int rows, int cols);
int mulMatrix(float* a, float* b, float* c, int rowsA, int colsA, int rowsB, int colsB, int rowsC, int colsC);
int dotProduct(float* a, float* b, float* c, int size);
int inverseMatrix(float* a, float* inverse, int rows, int cols);
int printVector(float* v, int size);
int printMatrix(float* m, int rows, int cols);

#endif /* CUDA_PROJECT_H_ */
