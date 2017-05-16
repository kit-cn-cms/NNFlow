# Install environment
## On ekpdeepthought

```
export WORKDIR="/passender/pfad"
virtualenv --system-site-packages $WORKDIR
source $WORKDIR/bin/activate
pip install tensorflow-gpu
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-7.5-cudNNV5.1/lib64:/usr/local/cuda-7.5-cudNNV5.1/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda-7.5-cudNNV5.1
export PYTHONPATH=$PYTHONPATH:/usr/local/lib
pip install rootpy
NOTMVA=1 pip2 install --upgrade  root_numpy
pip install pandas
pip install tables
```

Script to activate virtual environment:
```
source $WORKDIR/bin/activate
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-7.5-cudNNV5.1/lib64:/usr/local/cuda-7.5-cudNNV5.1/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda-7.5-cudNNV5.1
export PYTHONPATH=$PYTHONPATH:/usr/local/lib
```