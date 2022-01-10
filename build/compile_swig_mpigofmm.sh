########################################
# short script to encapsulate the docker commands to run the 
########################################
swig -o toolswrap.cpp -c++ -python tools.i
# fix include path issues for numpy: add include path for numpy manually in compile command:
mpicc -o tools_wrap.os -c -I/usr/include/python3.7m -I/usr/local/lib/python3.7/dist-packages/numpy/core/include/ -I../gofmm/ -I../include/ -I../frame/ -I ../frame/base/ -I../frame/containers/ toolswrap.cpp -fPIC -DHMLP_USE_MPI=true

mpic++ -O3 -fopenmp -m64 -fPIC -D_POSIX_C_SOURCE=200112L -fprofile-arcs -ftest-coverage -fPIC -DUSE_BLAS -mavx -std=c++11  -lpthread -fopenmp -lm -L/usr/lib/x86_64-linux-gnu -lopenblas tools_wrap.os  -o _tools.so -shared  -Wl,-rpath,/workspace/gofmm/build: libhmlp.so  -DHMLP_USE_MPI=true
