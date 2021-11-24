#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/* Use a binary tree */
#define N_CHILDREN 2

namespace {
typedef float T;
typedef SPDMatrix<float> SPDMATRIX_DENSE;
typedef Data<float> DATA_s;
typedef gofmm::CommandLineHelper CommandLineHelper;

DATA_s loaded_matrix;
SPDMATRIX_DENSE spdmatrix;

/* Create infrastructure for storing matrix in a tree */
/** Use the geometric-oblivious splitter from the metric ball tree. */
typedef gofmm::centersplit<SPDMatrix<float>, N_CHILDREN, T> SPLITTER;

/** Use the geometric-oblivious splitter from the randomized tree. */
typedef gofmm::randomsplit<SPDMatrix<float>, N_CHILDREN, T> RKDTSPLITTER;

/** Create configuration for all user-define arguments. */
typedef gofmm::Configuration<T> CONFIGURATION;

/** (Optional) provide neighbors, leave uninitialized otherwise. */
typedef Data<pair<T, size_t>> DATA_PAIR;

} // namespace

/**
 * @brief This data structure gofmmTree enables storage of configuration
 * parameters needed to execute mainly the following two operations:
 * 1. Mat-vec multiplication
 * 2. SPD Inverse
 */
class GofmmTree {
private:
  SPDMATRIX_DENSE K_;
  std::vector<const char *> argv;

  /* Configuration parameters */
  std::string executable_;
  int problem_size_;
  int max_leaf_node_size_;
  int num_of_neighbors_;
  int max_off_diagonal_ranks_;
  int num_rhs_;
  T user_tolerance_;
  T computation_budget_;
  std::string distance_type_; //  geometry, kernel, angle
  std::string matrix_type_;   // testsuit, dense, ooc, kernel, userdefine
  std::string kernel_type_;   // gaussian, laplace

public:
  GofmmTree();
  GofmmTree(std::string executable, int problem_size, int max_leaf_node_size,
            int num_of_neighbors, int max_off_diagonal_ranks, int num_rhs,
            T user_tolerance, T computation_budget, std::string distance_type,
            std::string matrix_type, std::string kernel_type,
            SPDMATRIX_DENSE K);

  void ConvertToVector();
  void MultiplyDenseSpdMatrix(DATA_s data, double *product_matrix,
                              int len_mul_numpy);
  void InverseOfDenseSpdMatrix(T lambda, double *inverse_matrix,
                               int matrix_length);
}; // GofmmTree

/**
 *  @brief A container to store argvs in vector and use destructor to
 * automatically free the memory.
 */
class FileToArgv {
private:
  std::vector<const char *> argv;

public:
  FileToArgv();
  explicit FileToArgv(const char *filename);
  ~FileToArgv();

  void PrintArgv();
  std::vector<const char *> ReturnArgv();
}; // FileToArgv

hmlpError_t CallLaunchHelper(const char *filename);

hmlpError_t CallLaunchHelper(SPDMatrix<float> &K, const char *filename);

SPDMATRIX_DENSE LoadDenseSpdMatrix(uint64_t height, uint64_t width,
                                   std::string const &filename);

SPDMATRIX_DENSE& LoadDenseSpdMatrixFromConsole(float *numpy_matrix,
                                               int num_of_rows,
                                               int num_of_cols);

DATA_s& LoadNumpyMatrixFromConsole(float *numpy_matrix, int num_of_rows,
                                   int num_of_cols);
