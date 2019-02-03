

source activate deepjetLinux3_gpu

export DEEPJETCORE=`pwd`

export PYTHONPATH=`pwd`/../:$PYTHONPATH
export LD_LIBRARY_PATH=`pwd`/compiled:$LD_LIBRARY_PATH
export PATH=`pwd`/bin:$PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libmkl_core.so:$CONDA_PREFIX/lib/libmkl_sequential.so

export CUDA_HOME=/cvmfs/cms-lpc.opensciencegrid.org/sl7/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
