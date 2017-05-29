# NNFlow

A framework to train binary and multinomial neural networks for the separation of ttbar subprocesses and ttH.

The framework is based on a framework for binary classification by Max Welsch and a framework for multinomial classification by Martin Lang.


## Install environment
### On ekpdeepthought

```
export TENSORFLOWINSTALLDIR=$PWD"/TensorFlow"
virtualenv --system-site-packages $TENSORFLOWINSTALLDIR
source $TENSORFLOWINSTALLDIR/bin/activate
pip install tensorflow-gpu
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-7.5-cudNNV5.1/lib64:/usr/local/cuda-7.5-cudNNV5.1/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda-7.5-cudNNV5.1
export PYTHONPATH=$PYTHONPATH:/usr/local/lib
pip install scipy
pip install matplotlib
pip install rootpy
NOTMVA=1 pip2 install --upgrade  root_numpy
pip install pandas
pip install tables
pip install scikit-learn
```

Activate virtual environment in a new shell:
```
export TENSORFLOWDIR = /path/to/tensorflow
source $TENSORFLOWDIR/bin/activate
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-7.5-cudNNV5.1/lib64:/usr/local/cuda-7.5-cudNNV5.1/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda-7.5-cudNNV5.1
export PYTHONPATH=$PYTHONPATH:/usr/local/lib
```


### On ekpbms3

```
export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
source $VO_CMS_SW_DIR/cmsset_default.sh


export SCRAM_ARCH="slc6_amd64_gcc530"
export CMSSW_VERSION="CMSSW_9_0_3"
export CMSSWINSTALLDIR=$PWD"/"$CMSSW_VERSION
export PIPTARGETDIR=$CMSSWINSTALLDIR"/lib/"$SCRAM_ARCH
scram project $CMSSW_VERSION
cd $CMSSW_VERSION/src
cmsenv

wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py --target=$PIPTARGETDIR
rm get-pip.py

python -m pip install --target=$PIPTARGETDIR root_numpy
python -m pip install --target=$PIPTARGETDIR tables
python -m pip install --target=$PIPTARGETDIR --upgrade pandas
```

Activate environment in a new shell:
```
cd /path/to/CMSSW_9_0_3
cmsenv
```


## Work flow
### Preprocessing
- Convert root files to HDF5 files.
- Merge HDF5 files to one file.
- Create a data set (Numpy 2D array) for the training.

### Training
- Train either a binary or a multinomial neural network.
