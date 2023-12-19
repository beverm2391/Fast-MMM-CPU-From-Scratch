


template <int rows, int columns, int inners>
inline void matmulImplNative(const float *left, const float *right, float *result)
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