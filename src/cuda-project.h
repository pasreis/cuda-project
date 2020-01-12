/*
 * cuda-project.h
 *
 *  Created on: Jan 12, 2020
 *      Author: cuda-s10
 */

#ifndef CUDA_PROJECT_H_
#define CUDA_PROJECT_H_

void addVector(float* a, float* b, float* c, int size);
void subVector(float* a, float* b, float* c, int size);
void addMatrix(float* a, float* b, float* c, int rows, int cols);
void subMatrix(float* a, float* b, float* c, int rows, int cols);
void mulMatrix(float* a, float* b, float* c, int rowsA, int colsA, int rowsB, int colsB, int rowsC, int colsC);
void dotProduct(float* a, float* b, float* c, int size);
void inverseMatrix(float* a, float* inverse, int rows, int cols);

#endif /* CUDA_PROJECT_H_ */
