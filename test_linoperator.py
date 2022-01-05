import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg
import tools  # gofmm shared lib stuff
from scipy.linalg import eig, eigh
from scipy.sparse.linalg import LinearOperator
import sys
sys.path.insert(1, '../python')


# NOTE: make sure "path/to/datafold" is in sys.path or PYTHONPATH if not ,!installed
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.utils.plot import plot_pairwise_eigenvector

random_state = 42


class gofmm_mulCalculator():
    """This is a child class of the rse_calculator. It speciflizes in
    calculating the rse of matrix multiplication."""
    def __init__( self, executable, problem_size, max_leaf_node_size, num_of_neighbors, max_off_diagonal_ranks, num_rhs, user_tolerance, computation_budget,
                 distance_type, matrix_type, kernel_type, spd_matrix, weight ):
                 
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
       
    def matvec(self):
        gofmmCalculator = tools.GofmmTree( self.executable, self.problem_size,
                                          self.max_leaf_node_size,
                                          self.num_of_neighbors, self.max_off_diagonal_ranks, self.num_rhs,
                                          self.user_tolerance, self.computation_budget,
                                          self.distance_type, self.matrix_type,
                                          self.kernel_type, self.denseSpd )

        # return a 1D array which is a 2D matrix flattened row-wise
        c = gofmmCalculator.MultiplyDenseSpdMatrix( self.wData, self.lenMul )
        # resize it to 2D
        spdMatrix_mul = np.resize( c, ( self.problem_size, self.num_rhs ) )

        return spdMatrix_mul


# TODO:
# - tic/toc for timing: total vs. 1x _matvec call etc. (import time from GOFMM?)
# - subprocess calls to GOFMM? <=> writing/reading the matrix, calling GOFMM externally?
class FullMatrix( LinearOperator ):
    def __init__( self, matrixA, dtype="float32" ):
        self.matrixA = tools.LoadDenseSpdMatrixFromConsole( np.float32( matrixA ) )
        self.shape = matrixA.shape
        self.dtype = np.dtype( dtype )
        self.gofmmCalculator = tools.GofmmTree( executable, problem_size, max_leaf_node_size, num_of_neighbors, max_off_diagonal_ranks,
                                                num_rhs, user_tolerance, computation_budget, distance_type, matrix_type, kernel_type, self.matrixA )


    def _matvec( self, x ):
        # TODO perform matvec with gofmm
        #a=np.float32(x)
        #print(x.shape)
        a=x.reshape( problem_size,1 )
        #print(x)
        rseCalculatorMul.wData = tools.LoadNumpyMatrixFromConsole( np.float32( a ) )
        # xx=np.vstack([x for i in range (num_rhs)])
        #a= tools.load_matrix_from_console(np.float32(xx.reshape(problem_size,num_rhs)))
        #print(spdSize * num_rhs)
        #c = self.gofmmCalculator.mul_denseSPD(a, spdSize )
        #print("test")
        #spdMatrix_mul = np.resize(c,(spdSize, num_rhs))
        #print(spdMatrix_mul[0])
        #print(spdMatrix_mul.shape)
        b= rseCalculatorMul.matvec()
        print( b.shape )
        #a=a.reshape(spdSize,) #np.dot(K,x)#spdMatrix_mul
        #print(a.shape)
        #x.resize(spdSize,)
        #c= np.dot(K,x)
        #print(c.shape)
        return b


executable = "./test_gofmm"
problem_size = 256
max_leaf_node_size = 128
num_of_neighbors = 0
max_off_diagonal_ranks = 128
num_rhs = 1
user_tolerance = 1E-7
computation_budget = 0.00
distance_type = "angle"
matrix_type = "testsuit"
kernel_type = "gaussian"
rng = np.random.default_rng( random_state )

data = rng.uniform( low = ( -2, -1 ), high = ( 2, 1 ), size = ( problem_size, 2 ) )
pcm = pfold.PCManifold( data )
pcm.optimize_parameters()
dmap = dfold.DiffusionMaps(
    kernel=pfold.GaussianKernel(epsilon=pcm.kernel.epsilon),
    n_eigenpairs=5,
    dist_kwargs=dict(cut_off=pcm.cut_off),
)
dmap.fit(pcm, store_kernel_matrix=True)
evecs, evals = dmap.eigenvectors_, dmap.eigenvalues_
plot_pairwise_eigenvector(
    eigenvectors=dmap.eigenvectors_,
    n=1,
    fig_params=dict(figsize=[5, 5]),
    scatter_params=dict(cmap=plt.cm.Spectral, s=1),
)
K = dmap.kernel_matrix_ #generateSPD_fromKDE(n_pts) # np.ones((n_pts,n_pts),dtype=np.float32) #dmap.kernel_matrix_
# print(type(K))
# print(K)
K_sparse = K.copy()
K = K.todense()
# print(type(K))
# print(K)
K = K.astype("float32")

w = np.ones((problem_size, num_rhs))

rseCalculatorMul = gofmm_mulCalculator( executable, problem_size, max_leaf_node_size,
                                        num_of_neighbors, max_off_diagonal_ranks, num_rhs, user_tolerance, computation_budget,
                                       	distance_type, matrix_type, kernel_type, K, w )
                                           
                                           
#b=rseCalculatorMul.matvec().reshape(spdSize)
#print(b)

kernel_matrix_OP = FullMatrix( K, dtype=np.float32 )
# assert(myOp.shape == (4,4))

# ?? inherit from class DmapKernelMethod(BaseEstimator): l. 435  in base.py
# das macht Klasse DiffusionMaps auch.

# symmetric case?
# => directly use
n_eigenpairs = 5
solver_kwargs = {
    "k": n_eigenpairs,
    "which": "LM",
    "v0": np.ones(problem_size),
    "tol": 1e-14,
}
evals_all, evecs_all = scipy.sparse.linalg.eigsh(K_sparse, **solver_kwargs)
exact_values = evals_all[-n_eigenpairs:]
print("evals_all", evals_all[-n_eigenpairs:])
# assumed output:
# [0.35179151 0.36637801 0.48593941 0.50677673 0.74200987]
# TODO check relevance of l.59-68 in eigsolver.py
evals_large, evecs_large = scipy.sparse.linalg.eigsh(kernel_matrix_OP, **solver_kwargs)
print("evals_large", evals_large)
# assumed output:
# [0.35179151 0.36637801 0.48593941 0.50677673 0.74200987]
print(np.dot(evecs_large.T, evecs_all[:, -n_eigenpairs:]))
#print( " 2Norm error: ", np.linalg.norm(exact_values - evals_large) )
absolute_errors = exact_values - evals_large
print( " Norm error: ", np.linalg.norm( absolute_errors ))
# assumed output:
# [[ 1.  0.  0. -0. -0.]    # may vary (signs)
#  [-0.  1. -0.  0.  0.]
#  [ 0.  0.  1. -0. -0.]
#  [-0.  0. -0. -1.  0.]
