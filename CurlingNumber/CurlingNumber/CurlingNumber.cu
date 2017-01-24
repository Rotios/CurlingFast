////(c) 2017 John Freeman and Jose Rivas
//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include <time.h>
//#include <stdlib.h>
//#include <string.h>
//
//#define INITIAL_CAPACITY 1024
//#define X(r, c) ((r * r) + r) * 0.5 + c 
//
///******************** Find the min value **************************/
//__global__ void minCompare(char *table, char *temps, bool *comparisons, int *size) {
//
//    int colIdx = threadIdx.x;
//
//    int rowIdx = blockIdx.x * blockDim.x;
//    int rowIdy = blockIdx.y * blockDim.y;
//
//    int seqLength = *size;
//
//    // Work backwards on the table rows
//    int tabRowIdx = (seqLength) - 1 - rowIdx;
//    int tabRowIdy = (seqLength) - 1 - rowIdy;
//
//    // Get the index of the value we are looking at
//    int tabIdx = X(tabRowIdx, colIdx);
//    int tabIdy = X(tabRowIdy, colIdx);
//
//    int boolId = seqLength * colIdx + rowIdx;
//
//    int xval = ((colIdx + tabRowIdx + 1) >= seqLength) * (table[tabIdx] - '0');
//    int yval = ((colIdx + tabRowIdx + 1) >= seqLength) * (table[tabIdy] - '0');
//
//    if (yval == 0 || yval == xval) {}
//    else if (xval == 0 || xval > yval) {
//        comparisons[boolId] = false;
//    }
//}
//
//__global__ void cudaMin(char *table, char *temps, bool *comparisons, int *size) {
//    int idxx = *size * blockIdx.y + blockIdx.x;
//    int idx = X(*size - 1 - blockIdx.x, blockIdx.y);
//
//    if (comparisons[idxx]) {
//        temps[idxx] = table[idx];
//    }
//
//    comparisons[idxx] = true;
//}
//
///************************* Find the max value **********************/
//__global__ void maxCompare(char *a, bool *check, int *size) {
//    int idx = threadIdx.x + blockIdx.x * blockDim.x;
//    int idy = threadIdx.y + blockIdx.y * blockDim.y;
//
//    if (idx == idy) { return; }
//
//    int xval = a[idx];
//    int yval = a[idy];
//
//    if (xval < yval) {
//        check[idx] = false;
//    }
//}
//
//__global__ void cudaMax(char *a, bool *check, int* size, int *max) {
//    int idx = blockIdx.x;
//
//    if (check[idx]) {
//        max[0] = a[idx];
//    }
//    size++;
//}
//
///*********************** Helper Methods ********************************************/
//// Magic if it works
//__global__ void fillRow(char *table, char *sequence, int *index) {
//    int column = threadIdx.x + blockIdx.x * blockDim.x;
//    int row = *index;
//    int pastSequencePos = row - (column + 1);
//    int position = X(row, column);
//    int position2 = (pastSequencePos > column) * X(pastSequencePos, column);
//
//    table[position] += '1' + (sequence[pastSequencePos] == sequence[row]) * (((pastSequencePos > column) * (table[position2] - '0')) + (pastSequencePos <= column));
//}
//
//void initializeTable(char *table, char *sequence, int seqLength) {
//
//    int *index;
//    cudaMalloc((void **)&index, sizeof(int));
//
//    for (int i(0); i < seqLength; ++i) {
//        cudaMemcpy(index, (void *)&i, sizeof(int), cudaMemcpyHostToDevice);
//        fillRow << < dim3(i + 1, 1), 1 >> > (table, sequence, index);
//    }
//
//    cudaFree(index);
//}
//
//void printBoolArray(bool *comparisons, int size){
//    
//    bool *comp;
//    comp = (bool *)malloc(size * sizeof(bool));
//
//    cudaMemcpy(comp, comparisons, size * sizeof(bool), cudaMemcpyDeviceToHost);
//
//    while(comp){
//        printf("%d ", *comp++);
//    }
//
//    printf("\n");
//}
//
//void printTable(char *table, int length) {
//    char *CPUTable;
//    CPUTable = (char *)malloc(length * sizeof(char));
//    cudaMemcpy(CPUTable, table, length * sizeof(char), cudaMemcpyDeviceToHost);
//    length = strlen(CPUTable);
//    for (int i(0); i < length; ++i) {
//        printf("%c ", CPUTable[i]);
//    }
//    
//        printf("\n");
//    free(CPUTable);
//}
//
///********************* Find the Curl *****************************************/
//void findCurl(char *table, char *sequence, char *temps, bool *comparisons, int *size, int *curl, int seqLength) {
//
//    int numBlocks = seqLength >> 1;
//    minCompare << < dim3(numBlocks, numBlocks), numBlocks >> > (table, temps, comparisons, size);
//    //printBoolArray(comparisons, 16);
//    cudaMin << < dim3(numBlocks, numBlocks), 1 >> > (table, temps, comparisons, size);
//
//    numBlocks = (numBlocks * numBlocks) * 0.5f;
//    maxCompare << < dim3(numBlocks, numBlocks), numBlocks >> > (temps, comparisons, size);
//    cudaMax << < dim3(numBlocks, numBlocks), 1 >> > (sequence, comparisons, size, curl);
//
//    if (seqLength < INITIAL_CAPACITY) {
//        int len = *size++;
//        sequence[len] = *curl;
//    }
//}
//
//int main() {
//    char *table;
//    char *sequence;
//    char *temps;
//    bool *comparisons;
//    int *size;
//    int *cuda_capacity;
//    int *curl;
//    int capacity = INITIAL_CAPACITY;
//
//    // ((capacity + 1) * capacity) / 2 for table
//    int table_size = (((capacity + 1) * capacity) / 2);
//
//    // size needed for bool arrays in min and max functions
//    int compare_size = (capacity / 2) * (capacity / 2);
//
//    cudaMalloc((void**)&table, table_size * sizeof(char));
//    cudaMalloc((void**)&sequence, capacity * sizeof(char));
//    cudaMalloc((void**)&temps, capacity * sizeof(char));
//    cudaMalloc((void**)&comparisons, compare_size * sizeof(bool));
//    cudaMalloc((void**)&size, sizeof(int));
//    cudaMalloc((void**)&cuda_capacity, sizeof(int));
//    cudaMalloc((void**)&curl, 2 * sizeof(int));
//    cudaMemcpy(cuda_capacity, (int*)&capacity, sizeof(int), cudaMemcpyHostToDevice);
//
//    
//    bool *allTrues = (bool *) malloc(compare_size * sizeof(bool));
//    memset(allTrues, true, compare_size * sizeof(bool));
//    cudaMemcpy(comparisons, allTrues, compare_size * sizeof(bool), cudaMemcpyHostToDevice);
//
//    while (1) {
//
//        cudaMemset(table, '0', table_size * sizeof(char));
//
//        char buffer[INITIAL_CAPACITY];
//        printf("Input a sequence to curl:\n");
//        scanf("%s", buffer);
//
//        int seqLength = strlen(buffer);
//        cudaMemcpy(sequence, buffer, seqLength * sizeof(char), cudaMemcpyHostToDevice);
//        cudaMemcpy(size, (int*)&seqLength, sizeof(int), cudaMemcpyHostToDevice);
//
//        initializeTable(table, sequence, seqLength);
//        printTable(table, ((seqLength + 1) * seqLength)/2);
//        clock_t start = clock();
//
//        for (int i(seqLength); i < capacity; i++) {
//            findCurl(table, sequence, temps, comparisons, size, curl, seqLength);
//            fillRow << < dim3(seqLength++, 1), 1 >> > (table, sequence, size);
//        }
//
//        clock_t stop = clock();
//        double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
//        printf("Elapsed time: %.3fs\n", elapsed);
///*
//        char *seq = (char *) malloc(INITIAL_CAPACITY*sizeof(char));
//        
//        cudaMemcpy(seq, (int*)&sequence, seqLength * sizeof(char), cudaMemcpyDeviceToHost);
//        for (int i = 0; i < seqLength; ++i) {
//            printf("%c ", seq[i]);
//        }*/
//        printf("\n\n");
//    }
//    
//
//        cudaFree(table);
//        cudaFree(sequence);
//        cudaFree(temps);
//        cudaFree(comparisons);
//        cudaFree(size);
//        cudaFree(cuda_capacity);
//        cudaFree(curl);
//    return 0;
//}
//
///*
//__global__ void minCompare(int *a, int *set, bool *check, int *capacity) {
//    int cap = capacity[0];
//    int offset = set[0];
//
//    int idx = threadIdx.x + blockIdx.x * blockDim.x;
//    int idy = threadIdx.y + blockIdx.y * blockDim.y;
//    int tabx = idx + cap + offset;
//    int taby = idy + cap + offset;
//
//    if (idx == idy) { return; }
//
//    int xval = a[tabx];
//    int yval = a[taby];
//
//    if (yval == 0) {}
//    else if (xval == 0) {
//        check[idx] = false;
//    }
//    else if (xval > yval) {
//        check[idx] = false;
//    }
//}*/
//
/////********************** Min and Max Functions ******************************************/
////void findMin(int *arr, const int length, const int offset, int *minimum, int *capacity) {
////    //length - 1 = row, offset = location of first element
////
////    bool *check;
////    int *set;
////    int *row = (int*) malloc(sizeof(int));
////    const int intSize = sizeof(int);
////    const int bsize = length * sizeof(bool);
////
////    cudaMalloc((void**)&check, bsize);
////    cudaBoolFill<<< dim3(length, 1), 1 >>>(check, length);
////
////    cudaMalloc((void**)&set, intSize);
////    cudaMemcpy(set, (int*)&offset, intSize, cudaMemcpyHostToDevice);
////
////    cudaMemcpy(row, capacity, intSize, cudaMemcpyDeviceToHost);
////    row[0] = row[0] * (length - 1);
////   
////    printf("offset = %d    length = %d     row = %d\n", offset, length, row[0]);
////
////    int *row2;
////    cudaMalloc((void**) &row2, intSize);
////    cudaMemcpy(row2, row, intSize, cudaMemcpyHostToDevice);
////
////    minCompare<<< dim3(length, length), 1 >>>(arr, set, check, row2);
////    cudaMin<<< dim3(length, 1), 1 >>>(arr,  check, minimum, row2);
////
////    cudaFree(check);
////}
////
////int findMax(char *arr, const int length) {
////    bool *check;
////    int *max;
////
////    const int intSize = sizeof(int);
////    const int bsize = length * sizeof(bool);
////
////    cudaMalloc((void**)&check, bsize);
////    cudaBoolFill << < dim3(length, 1), 1 >> > (check, length);
////
////    cudaMalloc((void**)&max, intSize);
////
////    maxCompare << < dim3(length, length), 1 >> > (arr, check);
////    cudaMax << < dim3(length, 1), 1 >> > (arr, check, max);
////
////    int maxhost[1];
////    cudaMemcpy(maxhost, max, intSize, cudaMemcpyDeviceToHost);
////
////    cudaFree(max);
////    cudaFree(check);
////
////    return maxhost[0];
////}