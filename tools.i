// "tools.i"
%module tools


// Must include these two lib files before .cpp so that .cpp can see
// the types
%include "stdint.i" // Included to use uint64_t
%include <std_string.i>  // Included to use std::string


/* Initialize the swig and c++ interface: can only be done once */
%{
  #define SWIG_FILE_WITH_INIT
  #include "../example/test_gofmm.cpp"
%}


/* Use numpy array: this section must be placed before .cpp include */
%include "numpy.i"
%init %{
  import_array();
%}

// Typemap: the order and names of the parameteres must match exactly
// the function args in the cpp file. `\` is the line separator
%apply (float* IN_ARRAY2, int DIM1, int DIM2 ) \  // IN_ARRAY2: Input 2D
      {(float* numpy_matrix, int num_of_rows, int num_of_cols)}

// load_matrix_from_console
%apply (float* IN_ARRAY2, int DIM1, int DIM2 ) \  // IN_ARRAY2: Input 2D
      {(float* numpy_matrix, int num_of_rows, int num_of_cols)}

// mul_numpy is actually a 2D array flattened row-wise into a 1D array
// Why not use 2D? bc typemap (double* ARGOUT_ARRAY2, int DIM1, int DIM2)
// is not available in numpy.i
%apply (double* ARGOUT_ARRAY1, int DIM1) \  // ARGOUT: output array
       {(double* product_matrix, int len_mul_numpy)}

// inv_numpy is actually a 2D array flattened row-wise into a 1D array
// Why not use 2D? bc typemap (double* ARGOUT_ARRAY2, int DIM1, int DIM2)
// is not available in numpy.i
%apply (double* ARGOUT_ARRAY1, int DIM1) \  // ARGOUT: output array
       {(double* inverse_matrix, int len_inv_numpy)}

/* Typically in the end */
%include "../example/test_gofmm.h"
