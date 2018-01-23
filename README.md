##Dependencies
The following dependencies are required:

* hyperopt
* networkx==1.11
* pymongo
* nltk
* scikit-learn
* pandas
* numpy
* scipy
* tensorflow==0.12.0-rc0

most of the above can be installed using pip. If you have sudo access, run:

    sudo pip install -U numpy scipy pandas pymongo networkx hyperopt nltk scikit-learn

If you don't, and want to install locally, run:

    pip install -U --user numpy scipy pandas pymongo networkx hyperopt nltk scikit-learn



################################

#installing with anaconda:
conda create --name arkcat python numpy 
pip install --upgrade numpy #it needs to be numpy 1.12.1
pip install scipy 
pip install scikit-learn
pip install setuptools
pip install tensorflow
pip install matplotlib

#not sure why this is needed. there's a bug report about it here: https://github.com/ContinuumIO/anaconda-issues/issues/720 seems like it's only a problem on ubuntu
#conda install  -f  numpy

source activate arkcat

#for hyperopt:
cd hyperopt
#make sure setuptools is the most recent version, or it might throw errors
pip install -e .
pip install pymongo
pip install networkx
pip install pandas

pip install xgboost

#for the dpp sampler
pip install gpy
pip install GPyOpt
pip install matlab_wrapper

#make sure matlab is in $PATH. if it's not, add it to .bashrc with something like export PATH="/projects/matlab2016a/bin:$PATH"
#or, download matlab runtime from here: https://www.mathworks.com/products/compiler/mcr.html
# unzip, then cd into the dir
#if x11 forwarding doesn't work:
./install -mode silent -agreeToLicense yes -destinationFolder <folder>
#if x11 does work, you can use the gui
#the gui will tell you to append something like this next line to your ~/.bashrc file, but using "-mode silent" will omit it:
export LD_LIBRARY_PATH="/home/jesse/software/matlab/v901/runtime/glnxa64:/home/jesse/software/matlab/v901/bin/glnxa64:/home/jesse/software/matlab/v901/sys/os/glnxa64:$LD_LIBRARY_PATH"
#unfortunately, having LD_LIBRARY_PATH not be the empty string causes SSL authentication to break for all git repos on aws ec2. this is super dumb, i don't understand.


#then, add something like this (dependent on destinationFolder) to ~/.bashrc
export PYTHONPATH=$PYTHONPATH:/home/ec2-user/projects/MATLAB/MATLAB_Runtime/v901/extern/engines/python/dist

