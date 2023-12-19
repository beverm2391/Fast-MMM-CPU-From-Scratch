
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
            float acc = 0.0
            for (int inner = 0; inner < inners; inner++)
            {
                acc += left[row * columns + inner] * right[inner * columns + col];
            }
            result[row * columns + col] = acc;
        }
    }
}


int main() {
    // TODO: get some matricies and time each implementation
    return 0; 
}