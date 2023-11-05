import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg
from scipy.linalg import eig, eigh
from scipy.sparse.linalg import LinearOperator
from build import tools
import sys
sys.path.insert(1, '../python')


# NOTE: make sure "path/to/datafold" is in sys.path or PYTHONPATH if not ,!installed
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.utils.plot import plot_pairwise_eigenvector

random_state = 42

class FullMatrix( LinearOperator ):
    def __init__( self, executable, problem_size, max_leaf_node_size, num_of_neighbors, max_off_diagonal_ranks, num_rhs, user_tolerance, computation_budget,
                distance_type, matrix_type, kernel_type, spd_matrix, weight, dtype="float32" ):
        self.executable = executable
        self.problem_size = problem_size 
        self.max_leaf_node_size = max_leaf_node_size
        self.num_of_neighbors = num_of_neighbors
        self.max_off_diagonal_ranks = max_off_diagonal_ranks
        self.num_rhs = num_rhs
        self.user_tolerance = user_tolerance
        self.computation_budget = computation_budget
        self.distance_type = distance_type
        self.matrix_type = matrix_type
        self.kernel_type = kernel_type

        self.spd_matrix = np.float32( spd_matrix )  # from input

        # Construct a fix spd matrix and load it into SPDMATRIX_DENSE structure
        self.denseSpd = tools.LoadDenseSpdMatrixFromConsole( self.spd_matrix )

        self.weight = np.float32( weight )  # from input

        # Construct a dummy w vector and load it into DATA structure
        self.wData = tools.LoadNumpyMatrixFromConsole( self.weight )
        self.lenMul = self.problem_size * self.num_rhs
        self.shape = self.spd_matrix.shape
        self.dtype = np.dtype( dtype )

    def _matvec( self, x ):
        gofmmCalculator = tools.GofmmTree( self.executable, self.problem_size,
		                                  self.max_leaf_node_size,
		                                  self.num_of_neighbors, self.max_off_diagonal_ranks, self.num_rhs,
		                                  self.user_tolerance, self.computation_budget,
		                                  self.distance_type, self.matrix_type,
		                                  self.kernel_type, self.denseSpd )

        a = x.reshape( self.problem_size,1 )
        weightData = tools.LoadNumpyMatrixFromConsole( np.float32( a ) )
        c = gofmmCalculator.MultiplyDenseSpdMatrix( self.wData, self.lenMul )
        spdMatrix_mul = np.resize( c, ( self.problem_size, self.num_rhs ) )
        return spdMatrix_mul


# Input variables required for the instantiation of FullMatrix
executable = "./test_gofmm"
problem_size = 8192
max_leaf_node_size = 256
num_of_neighbors = 0
max_off_diagonal_ranks = 256
num_rhs = 1
user_tolerance = 1E-3
computation_budget = 0.00
distance_type = "kernel"
matrix_type = "dense"
kernel_type = "gaussian"
rng = np.random.default_rng( random_state )

# Generate kernel matrix from cnn_gp
from cnn_gp.kernels import Sequential, Conv2d, ReLU
import torch

model = Sequential(
    Conv2d(kernel_size=2),
    ReLU(),
    # Conv2d(kernel_size=2),
    # ReLU(),
    Conv2d(kernel_size=2,padding=0),  # equivalent to a dense layer
)

X = torch.randn(problem_size, 3, 2, 2) # batch 1: (bs, c, h, w)
Kxx = model(X, X, same=True) 
K = Kxx.to(torch.float64).numpy()
                                        
# Instantiation of FullMatrix   
weights = np.ones((problem_size, num_rhs))                                      
kernel_matrix_OP = FullMatrix(executable, problem_size, max_leaf_node_size,
                                        num_of_neighbors, max_off_diagonal_ranks, num_rhs, user_tolerance, computation_budget,
                                       	distance_type, matrix_type, kernel_type, K, weights, dtype=np.float32 )

n_eigenpairs = 5
solver_kwargs = {
    "k": n_eigenpairs,
    "which": "LM",
    "v0": np.ones(problem_size),
    "tol": 1e-6,
}

np.random.seed(42) # fix the generated random values
b = np.random.rand(problem_size)
b_tensor = torch.from_numpy(b).to(torch.float64)

# https://github.com/cambridge-mlg/cnn-gp/blob/c9260e6cf40e61d2e5b863bbdae32b46e52bd822/exp_mnist_resnet/classify_gp.py#L17
def solve_system(Kxx, Y):
    print("Running scipy solve Kxx^-1 Y routine")
    assert Kxx.dtype == torch.float64 and Y.dtype == torch.float64, """
    It is important that `Kxx` and `Y` are `float64`s for the inversion,
    even if they were `float32` when being calculated. This makes the
    inversion much less likely to complain about the matrix being singular.
    """
    A = scipy.linalg.solve(
        Kxx.numpy(), Y.numpy(), overwrite_a=True, overwrite_b=False,
        check_finite=False, assume_a='pos', lower=False)
    return torch.from_numpy(A)


solutions_all = solve_system(Kxx.cpu(), b_tensor)
solutions_large = scipy.linalg.solve(kernel_matrix_OP * np.identity(problem_size),b)
print("Solutions of scipy:", solutions_all)
print('\n')
print("Solutions of gofmm:", solutions_large)
solution_errors = solutions_all - solutions_large
print("Frobenius norm Error:", np.linalg.norm(solution_errors, 'fro'))
# print( "Norm solution error: ", np.linalg.norm( solution_errors ) / np.sqrt(problem_size))