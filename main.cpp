#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
// some headers for timing and random number generation

// ! Native Implementation
template <int rows, int columns, int inners> // template parameters, compile time constants
inline void matmulImplNative(const float *left, const float *right, float *result)
// left, right, and result point to the first element of matricies A, B, and C, respectively
// the const keyword means that the function cannot modify the values of the pointers for mats A and B
{
    for (int row = 0; row < rows; row++) // for each row of the matrix A
    {
        for (int col = 0; col < columns; col++) // for each column of the matrix B
        {
            for (int inner = 0; inner < inners; inner++) // for each element of the row/column
            {
                result[row * columns + col] += left[row * columns + inner] * right[inner * columns + col];
                // Result (ij) = A (ik) * B (kj) for each element of the row/column
                // the matrix is flattened into A 1D array
            }
        }
    }
}

// ! Native Implementation with Register Accumulation
// improved by performing the inner dot product in a register, rather than in memory
template <int rows, int columns, int inners>
inline void matmulImplNativeRegisterAcc(const float *left, const float *right, float *result)
{
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            float acc = 0.0 for (int inner = 0; inner < inners; inner++)
            {
                acc += left[row * columns + inner] * right[inner * columns + col];
            }
            result[row * columns + col] = acc;
        }
    }
}

// ! Cache Aware Loop Reorder
// Multidimensional matrices are represented in memory using a strided representation
// This means that the elements of the matrix are not stored contiguously in memory
// This can cause cache misses when looping over the matrix
// This implementation reorders the loops to improve cache performance

// "the inner loops now iterate through B & C in a memory sequential manner"
template <int rows, int columns, int inners>
inline void matmulImplLoopOrder(const float *left, const float *right, float *result)
{
    for (int row = 0; row < rows; row++)
    {
        for (int inner = 0; inner < inners, inner++)
        {
            for (int col = 0; col < columns; col++)
            {
                result[row * columns + col] += left[row * columns + inner] * right[inner * columns + col];
            }
        }
    }
}

// ! Loop Reorder Plus L1 Tiling on I
template <int rows, int columns, int inners, int tileSize>
inline void matmulImplTiling(const float *left, const float *right, float *result)
{
    for (int innerTile = 0; innerTile < inners; innerTile += tileSize)
    {
        for (int row = 0; row < rows; row++)
        {
            int innerTileEnd = std::min(inners, innerTile + tileSize);
            for (int inner = innerTile; inner < innerTileEnd; inner++)
            {
                for (int column = 0; column < columns, column++)
                {
                    result[row * columns + column] += left[row * inners + inner] * right[inner * columns + column];
                }
            }
        }
    }
}

// ! Multithreaded Implementation
// "We want to avoid having to do partial summing between threads, which would either require atomics or locking."
// By partitioning the result matrix into blocks, we can have each thread compute a block of the result matrix
template <int rows, int columns, int inners, int tileSize = ROW_COL_PARALLEL_INNER_TILING_SIZE>
inline void matmulImplRowColParallelInnerTiling(const float *left, const float *right, float *result)
{
    #pragma omp parallel for shared(result, left, right) default(none) \ // this is an OpenMP pragma, which tells the compiler to parallelize the following loop
        collapse(2) num_threads(8) // collapse(2) means that the two outer loops are collapsed into one loop, which is then parallelized
    for (int rowTile = 0; rowTile < rows; rowTile += 256)
    {
        for (int columnTile = 0; columnTile < columns; columnTile += 256)
        {
            for (int innerTile = 0; innerTile < inners; innerTile += tileSize)
            {
                for (int row = rowTile; row < rowTile + 256; row++)
                {
                    int innerTileEnd = std::min(inners, innerTile + tileSize);
                    for (int inner = innerTile; inner < innerTileEnd; inner++)
                    {
                        for (int col = columnTile; col < columnTile + 256; col++)
                        {
                            result[row * columns + col] += left[row * inners + inner] * right[inner * columns + col];
                        }
                    }
                }
            }
        }
    }
}

int main()
{
    // TODO: get some matricies and time each implementation
    return 0;
}