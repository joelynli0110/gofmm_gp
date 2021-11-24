/***********************************************************************************
 *  HMLP (High-Performance Machine Learning Primitives)                            *
 *                                                                                 *
 *  Copyright (C) 2014-2018, The University of Texas at Austin                     *
 *                                                                                 *
 *  This program is free software: you can redistribute it and/or modify           *
 *  it under the terms of the GNU General Public License as published by           *
 *  the Free Software Foundation, either version 3 of the License, or              *
 *  (at your option) any later version.                                            *
 *                                                                                 *
 *  This program is distributed in the hope that it will be useful,                *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of                 *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the                   *
 *  GNU General Public License for more details.                                   *
 *                                                                                 *
 *  You should have received a copy of the GNU General Public License              *
 *  along with this program. If not, see the LICENSE file.                         *
 *                                                                                 *
 ***********************************************************************************/

/** Use GOFMM templates. */
#include <gofmm.hpp>
/** Use dense SPD matrices. */
#include <containers/SPDMatrix.hpp>
/** Use implicit kernel matrices (only coordinates are stored). */
#include <containers/KernelMatrix.hpp>
/** Use implicit matrices. */
#include <containers/VirtualMatrix.hpp>
/** Use implicit Gauss-Newton (multilevel perceptron) matrices. */
#include <containers/MLPGaussNewton.hpp>
/** Use OOC covariance matrices. */
#include <containers/OOCCovMatrix.hpp>
/** Use Gauss Hessian matrices provided by Chao. */
#include <containers/GNHessian.hpp>
/** Use STL and HMLP namespaces. */
#include "test_gofmm.h"
using namespace std;

GofmmTree::GofmmTree() {} // size 0 vector initialization

GofmmTree::GofmmTree(std::string executable, int problem_size,
                     int max_leaf_node_size, int num_of_neighbors,
                     int max_off_diagonal_ranks, int num_rhs, T user_tolerance,
                     T computation_budget, std::string distance_type,
                     std::string matrix_type, std::string kernel_type,
                     SPDMATRIX_DENSE K)
    : executable_(executable), problem_size_(problem_size),
      max_leaf_node_size_(max_leaf_node_size),
      num_of_neighbors_(num_of_neighbors),
      max_off_diagonal_ranks_(max_off_diagonal_ranks), num_rhs_(num_rhs),
      user_tolerance_(user_tolerance), computation_budget_(computation_budget),
      distance_type_(distance_type), matrix_type_(matrix_type),
      kernel_type_(kernel_type) {
  K_ = K;            // Initialize SPD matrix
  ConvertToVector(); // Initialize argv
}

/**
 * @brief Converts the vector of strings to vector of char*
 */
void GofmmTree::ConvertToVector() {
  std::vector<std::string> parameters;

  parameters.push_back(executable_);
  parameters.push_back(std::to_string(problem_size_));
  parameters.push_back(std::to_string(max_leaf_node_size_));
  parameters.push_back(std::to_string(num_of_neighbors_));
  parameters.push_back(std::to_string(max_off_diagonal_ranks_));
  parameters.push_back(std::to_string(num_rhs_));
  parameters.push_back(std::to_string(user_tolerance_));
  parameters.push_back(std::to_string(computation_budget_));
  parameters.push_back(distance_type_);
  parameters.push_back(matrix_type_);
  parameters.push_back(kernel_type_);

  for (int i = 0; i < parameters.size(); i++) {
    argv.push_back(parameters[i].c_str());
  }

  parameters.clear();
}

/**
 * @brief Computes the product of a SPD dense matrix K and a vector or matrix
 * that is stored in DATA_s
 * @param data The matrix or vector which is one of the operands in the
 * multiplication.
 * @param product_matrix The result of the multiplication K * data.
 * @param len_mul_numpy Length of the 1D array version of the product_matrix.
 */
void GofmmTree::MultiplyDenseSpdMatrix(DATA_s data, double *product_matrix,
                                       int len_mul_numpy) {
  gofmm::CommandLineHelper cmd(argv.size(), argv);
  size_t spdmatrix_size = cmd.n;
  size_t num_columns_in_data = cmd.nrhs;

  if ((K_.row() != spdmatrix_size) || (data.row() != spdmatrix_size) ||
      (data.col() != num_columns_in_data)) {
    std::cerr << "Please check the shape of K and w must match the ones"
              << " specified in the parameter file \n";
    exit(0);
  }

  HANDLE_ERROR(hmlp_init());

  /* Compress K into our gof tree                                */
  /* 1st step: Argument preparation for fitting K into a tree    */
  /* GOFMM metric ball tree splitter (for the matrix partition). */
  SPLITTER splitter(K_, cmd.metric);

  /* Randomized matric tree splitter (for nearest neighbor).     */
  RKDTSPLITTER rkdtsplitter(K_, cmd.metric);

  /* Create configuration for all user-define arguments.         */
  CONFIGURATION config(cmd.metric, cmd.n, cmd.m, cmd.k, cmd.s, cmd.stol,
                       cmd.budget, cmd.secure_accuracy);

  /* (Optional) provide neighbors, leave uninitialized otherwise. */
  Data<pair<T, size_t>> NN;

  /* 2nd step: compress K.                                        */
  auto *tree_ptr = gofmm::Compress(K_, NN, splitter, rkdtsplitter, config);

  auto &tree = *tree_ptr;

  /* Construct a random w and compute u = kw.                     */
  Data<T> u = Evaluate(tree, data);

  /* Extract the multiplication result from u into mul_numpy      */
  for (size_t i = 0; i < spdmatrix_size; i++) {
    for (size_t j = 0; j < num_columns_in_data; j++) {
      product_matrix[j + i * num_columns_in_data] = u(i, j);
    }
  }

  delete tree_ptr;

  HANDLE_ERROR(hmlp_finalize());
}

/**
 * @brief Computes the inverse of a SPD matrix K.
 * @param lambda
 * @param inverse_matrix The computed inverse matrix is stored in this.
 * @param matrix_length Length of the 1D array version for the inverse_matrix.
 */
void GofmmTree::InverseOfDenseSpdMatrix(T lambda, double *inverse_matrix,
                                        int matrix_length) {
  gofmm::CommandLineHelper cmd(argv.size(), argv);

  HANDLE_ERROR(hmlp_init());

  /* Compress K into our gof tree                                            */
  /* 1st step: Argument preparation for fitting K into a tree                */
  /** GOFMM metric ball tree splitter (for the matrix partition).            */
  SPLITTER splitter(K_, cmd.metric);

  /** Randomized matric tree splitter (for nearest neighbor).                */
  RKDTSPLITTER rkdtsplitter(K_, cmd.metric);

  /** Create configuration for all user-define arguments.                    */
  CONFIGURATION config(cmd.metric, cmd.n, cmd.m, cmd.k, cmd.s, cmd.stol,
                       cmd.budget, cmd.secure_accuracy);

  /** (Optional) provide neighbors, leave uninitialized otherwise.           */
  Data<pair<T, size_t>> NN;

  /** 2nd step: compress K. */
  auto *tree_ptr = gofmm::Compress(K_, NN, splitter, rkdtsplitter, config);

  auto &tree = *tree_ptr;

  /* Construct the potential to be n x n identity matrix.                    */
  size_t n = cmd.n;

  /* Must change nrhs to be n in the new setup file for inverse.             */
  Data<T> u(n, n);

  /* Construct u as an identity matrix. Using 3 loops to enable parallelism  */
  for (size_t i = 0; i < n; i++) {
    for (size_t j = i + 1; j < n; j++) {
      u(i, j) = 0;
    }
  }

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < i; j++) {
      u(i, j) = 0;
    }
  }

  for (size_t k = 0; k < n; k++) {
    u(k, k) = 1;
  }

  /* Factorize as preparation                                                 */
  gofmm::Factorize(tree, lambda);

  /* w = inv( K + lambda * I ) * u where w is weight and u is potential.
     and rhs is u, the potential. Already know K and u. Using Solve to
     get w, which is stored in rhs.
     Since u is identity martix, w (which overwrites u) will be the inverse
     of w (still need regularization)
                                                                              */
  Solve(tree, u);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      inverse_matrix[j + i * n] = u(i, j);
    }
  }

  delete tree_ptr;

  HANDLE_ERROR(hmlp_finalize());
}

/**
 *  @brief size 0 vector initialization
 */
FileToArgv::FileToArgv() {}

/**
 * @brief Reads all the required input parameters from the given file.
 * @param file_name The name of the file.
 */
FileToArgv::FileToArgv(const char *file_name) {
  std::string line;
  std::vector<std::string> parameters;
  std::ifstream file(file_name);
  std::string parameter;

  while (std::getline(file, parameter)) {
    parameters.push_back(parameter);
  }

  /* Convert vector string to vector char* by pushing */
  for (int i = 0; i < parameters.size(); i++) {
    argv.push_back(parameters[i].c_str());
  }
  parameters.clear();
}

/**
 * @brief Free the argument vector.
 */
FileToArgv::~FileToArgv() { argv.clear(); }

/**
 * @brief Prints all the argvs line by line.
 */
void FileToArgv::PrintArgv() {
  for (auto i = argv.begin(); i != argv.end(); ++i) {
    std::cout << *i << '\n';
  }
}

/**
 * @brief Returns the argument vector.
 */
std::vector<const char *> FileToArgv::ReturnArgv() { return argv; }

/**
 * @brief Calls LaunchHelper by creating a spd matrix K of a size obtained from
 * command line and initialising it with random values.
 * @param filename The name of file which contains command line args.
 */
hmlpError_t CallLaunchHelper(const char *filename) {
  /* Construct vector string */
  FileToArgv argvObj(filename);
  gofmm::CommandLineHelper cmd(argvObj.ReturnArgv().size(),
                               argvObj.ReturnArgv());

  HANDLE_ERROR(hmlp_init()); // Initialize separate memory space at runtime

  /** random spd initialization */
  SPDMatrix<double> K(cmd.n, cmd.n);
  K.randspd(0.0, 1.0);

  hmlpError_t temp = gofmm::LaunchHelper(K, cmd);

  HANDLE_ERROR(hmlp_finalize());

  return temp;
}

/**
 * @brief Overloaded method that also calls LaunchHelper with the spd matrix it
 * is provided with.
 * @param K Spd matrix provided for the LaunchHelper.
 * @param filename Name of the file use to obtain comnnad line args
 */
hmlpError_t CallLaunchHelper(SPDMATRIX_DENSE &K, const char *filename) {

  FileToArgv argvObj(filename);
  gofmm::CommandLineHelper cmd(argvObj.ReturnArgv().size(),
                               argvObj.ReturnArgv());

  HANDLE_ERROR(hmlp_init());

  hmlpError_t temp = gofmm::LaunchHelper(K, cmd);

  HANDLE_ERROR(hmlp_finalize());

  return temp;
}

/** @brief Top level driver that reads arguments from the command line. */
int main(int argc, char *argv[]) {
  try {
    /** Parse arguments from the command line.                          */
    gofmm::CommandLineHelper cmd(argc, argv);
    /** HMLP API call to initialize the runtime                         */
    HANDLE_ERROR(hmlp_init(&argc, &argv));

    /** Run the matrix file provided by users.                          */
    if (!cmd.spdmatrix_type.compare("dense")) {
      using T = float;
      /** Dense spd matrix format.                                      */
      SPDMatrix<T> K(cmd.n, cmd.n, cmd.user_matrix_filename);
      gofmm::LaunchHelper(K, cmd);
    }

    /** Run the matrix file provided by users.                          */
    if (!cmd.spdmatrix_type.compare("ooc")) {
      using T = double;
      /** Dense spd matrix format.                                      */
      OOCSPDMatrix<T> K(cmd.n, cmd.n, cmd.user_matrix_filename);
      gofmm::LaunchHelper(K, cmd);
    }

    /** generate a Gaussian kernel matrix from the coordinates          */
    if (!cmd.spdmatrix_type.compare("kernel")) {
      using T = double;
      /** Read the coordinates from the file.                           */
      Data<T> X(cmd.d, cmd.n, cmd.user_points_filename);
      /** Set the kernel object as Gaussian.                            */
      kernel_s<T, T> kernel;
      kernel.type = GAUSSIAN;
      if (!cmd.kernelmatrix_type.compare("gaussian"))
        kernel.type = GAUSSIAN;
      if (!cmd.kernelmatrix_type.compare("laplace"))
        kernel.type = LAPLACE;
      kernel.scal = -0.5 / (cmd.h * cmd.h);
      /** SPD kernel matrix format (implicitly create).                 */
      KernelMatrix<T> K(cmd.n, cmd.n, cmd.d, kernel, X);
      gofmm::LaunchHelper(K, cmd);
    }

    /** create a random spd matrix, which is diagonal-dominant          */
    if (!cmd.spdmatrix_type.compare("testsuit")) {
      using T = double;
      /** dense spd matrix format                                       */
      SPDMatrix<T> K(cmd.n, cmd.n);
      /** random spd initialization                                     */
      K.randspd(0.0, 1.0);
      gofmm::LaunchHelper(K, cmd);
    }

    if (!cmd.spdmatrix_type.compare("mlp")) {
      using T = double;
      /** Read the coordinates from the file.                           */
      Data<T> X(cmd.d, cmd.n, cmd.user_points_filename);
      /** Multilevel perceptron Gauss-Newton                            */
      MLPGaussNewton<T> K;
      /** Create an input layer                                         */
      Layer<INPUT, T> layer_input(cmd.d, cmd.n);
      /** Create FC layers                                              */
      Layer<FC, T> layer_fc0(256, cmd.n, layer_input);
      Layer<FC, T> layer_fc1(256, cmd.n, layer_fc0);
      Layer<FC, T> layer_fc2(256, cmd.n, layer_fc1);
      /** Insert layers into                                            */
      K.AppendInputLayer(layer_input);
      K.AppendFCLayer(layer_fc0);
      K.AppendFCLayer(layer_fc1);
      K.AppendFCLayer(layer_fc2);
      /** Feed forward and compute all products                         */
      K.Update(X);
    }

    if (!cmd.spdmatrix_type.compare("cov")) {
      using T = double;
      OOCCovMatrix<T> K(cmd.n, cmd.d, cmd.nb, cmd.user_points_filename);
      gofmm::LaunchHelper(K, cmd);
    }

    if (!cmd.spdmatrix_type.compare("jacobian")) {
      using T = double;
      GNHessian<T> K;
      K.read_jacobian(cmd.user_matrix_filename);
      gofmm::LaunchHelper(K, cmd);
    }

    /** HMLP API call to terminate the runtime                           */
    HANDLE_ERROR(hmlp_finalize());
  } catch (const exception &e) {
    cout << e.what() << endl;
    return -1;
  }
  return 0;
} // main

/**
 * @brief Construct a new dense SPD matrix using input from a data file.
 * @param height Height of the spd matrix.
 * @param width Width of the spd matrix.
 * @return The constructed spdmatrix.
 */
SPDMATRIX_DENSE LoadDenseSpdMatrix(uint64_t height, uint64_t width,
                                   std::string const &filename) {
  SPDMATRIX_DENSE K(height, width, filename);

  return K;
}

/**
 * @brief Load values of a numpy matrix into a SPD matrix container.
 * @param numpy_matrix The numpy matrix that is to be loaded in SPDMATRIX_DENSE
 * container.
 * @param num_of_rows Number of rows in input numpy matrix.
 * @param num_of_cols Number of columns in input numpy matrix.
 * @return Reference to the SPD matrix container.
 */
SPDMATRIX_DENSE& LoadDenseSpdMatrixFromConsole(float *numpy_matrix,
                                               int num_of_rows,
                                               int num_of_cols) {
  SPDMATRIX_DENSE K(num_of_rows, num_of_cols);

  int index = -1;
  for (int i = 0; i < num_of_rows; i++) {
    for (int j = 0; j < num_of_cols; j++) {
      index = i * num_of_cols + j;
      K.data()[index] = numpy_matrix[index]; // .data()'s type is double*
    }
  }
  spdmatrix = K;
  return spdmatrix;
}

/**
 * @brief Load a numpy matrix into our self-defined DATA_s container.
 * @param numpy_matrix The numpy matrix that is to be loaded into DATA_s.
 * @param num_of_rows Number of rows in input numpy matrix.
 * @param num_of_cols Number of columns in input numpy matrix.
 * @return return reference to loaded data container.
 * @note If the input from the console is a numpy row matrix, it'll be first
 * converted to a numpy column matrix first in python before being processed by
 * this function. This means num_of_cols is at least 1.
 */
DATA_s& LoadNumpyMatrixFromConsole(float *numpy_matrix, int num_of_rows,
                                   int num_of_cols) {
  DATA_s data_container(num_of_rows, num_of_cols);

  for (size_t i = 0; i < num_of_rows; i++) {
    for (size_t j = 0; j < num_of_cols; j++) {
      data_container(i, j) = numpy_matrix[j + i * num_of_cols];
    }
  }
  loaded_matrix = data_container;
  return loaded_matrix;
}
