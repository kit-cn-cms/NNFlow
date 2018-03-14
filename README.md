# NNFlow

A framework to train binary and multiclass neural networks for classification of events in particle physics.

The framework is based on a framework for binary classification by Max Welsch and a framework for multiclass classification by Martin Lang.


## Mirroring
- This repository is hosted on gitlab.cern.ch and all PR should go to this repository.
- In addition, this repository is mirrored to Github.


## Install NNFlow
The framework is designed to be used with Python2.7.


### Installation in a virtual environment
```
export NNFLOWINSTALLDIR=$PWD"/NNFlow_venv"
virtualenv --system-site-packages $NNFLOWINSTALLDIR
source $NNFLOWINSTALLDIR/bin/activate

pip install --upgrade rootpy           # If PyROOT is not installed on your machine, skip these two points.
pip install --upgrade root_numpy       # In this case, you can not perform step 1 of the preprocessing.

pip install --upgrade tensorflow       # Use this, if CUDA is not installed on your machine.
pip install --upgrade tensorflow-gpu   # Use this, if CUDA is installed on your machine.

pip install git+http://github.com/kit-cn-cms/NNFlow.git#egg=NNFlow

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-7.5-cudNNV5.1/lib64:/usr/local/cuda-7.5-cudNNV5.1/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda-7.5-cudNNV5.1
```

Activate environment in a new shell:
```
export NNFLOWINSTALLDIR=/path/to/NNFlow_venv
source $NNFLOWINSTALLDIR/bin/activate
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-7.5-cudNNV5.1/lib64:/usr/local/cuda-7.5-cudNNV5.1/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda-7.5-cudNNV5.1
```


### Installation in a CMSSW environment
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
cd -

wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py --target=$PIPTARGETDIR
rm get-pip.py

python -m pip install --target=$PIPTARGETDIR virtualenv

export NNFLOWINSTALLDIR=$PWD"/NNFlow_venv_CMSSW"
python -m virtualenv --system-site-packages $NNFLOWINSTALLDIR
source $NNFLOWINSTALLDIR/bin/activate

pip install --upgrade rootpy
pip install --upgrade root_numpy

pip install git+http://github.com/kit-cn-cms/NNFlow.git#egg=NNFlow
```

Activate environment in a new shell:
```
export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
source $VO_CMS_SW_DIR/cmsset_default.sh

cd /path/to/CMSSW_9_0_3
cd src
cmsenv
export NNFLOWINSTALLDIR=/path/to/NNFlow_venv_CMSSW
source $NNFLOWINSTALLDIR/bin/activate
```


## Test the framework
To apply the framework, use the scripts in the directory "run_scripts". Firstly, you have to perform preprocessing_1, preprocessing_2 and preprocessing_3. Secondly, you have to use "train_neural_network.py". Thirdly, you can use the suitable analyse_model script. Further explanations are given as comments in the scripts.

You can test the framework with the scripts in the folder "example_run_scripts". All you have to do is to add the pathes to the data sets and to your working directory. To get example data sets, please contact the authors.
