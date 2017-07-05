# NNFlow

A framework to train binary and multinomial neural networks for the separation of ttbar subprocesses and ttH.

The framework is based on a framework for binary classification by Max Welsch and a framework for multinomial classification by Martin Lang.


## Mirroring
- This repository is hosted on gitlab.cern.ch and all PR should go to this repository.
- In addition, this repository is also mirrored to Github.


## Install NNFlow
### In a virtual environment
```
export NNFLOWINSTALLDIR=$PWD"/NNFlow"
virtualenv --system-site-packages $NNFLOWINSTALLDIR
source $NNFLOWINSTALLDIR/bin/activate

pip install --upgrade  rootpy          # If PyROOT is not installed on your machine, skip these two points.
pip install --upgrade  root_numpy      # In this case, you can not perform step 1 of the preprocessing.

pip install --upgrade tensorflow       # Use this, if CUDA is not installed on your machine.
pip install --upgrade tensorflow-gpu   # Use this, if CUDA is installed on your machine.

pip install git+http://github.com/kit-cn-cms/NNFlow.git#egg=NNFlow

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-7.5-cudNNV5.1/lib64:/usr/local/cuda-7.5-cudNNV5.1/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda-7.5-cudNNV5.1
```

Activate environment in a new shell:
```
export NNFLOWINSTALLDIR=/path/to/NNFlow
source $NNFLOWINSTALLDIR/bin/activate
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-7.5-cudNNV5.1/lib64:/usr/local/cuda-7.5-cudNNV5.1/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda-7.5-cudNNV5.1
```


### In a CMSSW environment
If PyROOT is not installed on your machine, you can use a CMSSW environment to perform the preprocessing. It is not possible to perform the training in this environment.
```
export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
source $VO_CMS_SW_DIR/cmsset_default.sh

export SCRAM_ARCH="slc6_amd64_gcc530"
export CMSSW_VERSION="CMSSW_9_0_3"
export CMSSWINSTALLDIR=$PWD"/"$CMSSW_VERSION
export PIPTARGETDIR=$CMSSWINSTALLDIR"/lib/"$SCRAM_ARCH
scram project $CMSSW_VERSION
cd $CMSSW_VERSION"/src"
cmsenv

wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py --target=$PIPTARGETDIR
rm get-pip.py

python -m pip install --target=$PIPTARGETDIR git+http://github.com/kit-cn-cms/NNFlow.git#egg=NNFlow
```

Activate environment in a new shell:
```
export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
source $VO_CMS_SW_DIR/cmsset_default.sh

cd /path/to/CMSSW_9_2_0
cd src
cmsenv
```


## Work flow
### Preprocessing
- Convert root files to HDF5 files.
- Merge HDF5 files to one file.
- Create a data set (Numpy 2D array) for the training.

### Training
- Train either a binary or a multinomial neural network.
