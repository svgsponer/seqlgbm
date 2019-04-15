# SEQLGBM and SFSEQL
Repository for SEQuence Learner Gradient Boosting Machine and Static Feature SEQuence Learner.
This repository contains code for reproducing the experiments explained in the paper:
Accurate Sequence Classification by Gradient Boosting with Linear Models by Severin Gsponer, Thach Le Nguyen, and Georgiana Ifrim.


## Install SEQLGBM
### Obtain the code
First clone the repository:
`git clone https://github.com/svgsponer/SEQL.git`

The repository consists of three parts projected to the three folders
- seql: `SEQLGBM` source code
- Protsol: Data and scripts related to the Protein solubility experiments
- UCR: Some data and scripts related to the Time Series Classification task.


### Requirements
SEQL has various requirements:
- C++17 compiler 
- CMake
- Armadillo
- JSON for Modern C++
- Catch2 (only if tests are build)
Furtherdown we provide a rough guidline how to install these requirements.
   
To install SEQLGBM itself run these steps:
```
cd seql
mkdir build
cd build
cmake .. 
cmake -build .
```
This should produce the `seqlr` within the bin `build/bin` directory.
Either call it with the full path or add it to your `$PATH`.
For further configuration it's best to use `ccmake`, which provides a
interface to tune the compilation options.


## Run

To run SEQLGBM use the following syntax

```
seqlr -t <trainfile> -s <testfile> -c <configfile> [--Validate -v <validationfile>] [-n <Name>] [--GBM] 
```
if you don't use the validation file use `/dev/null`


### Protein solubility experiment

The data used for this experiment is available under `Protsol/data`
The configurations we used to obtained the presented results are available in the corresponding folder (SEQL, SFSEQL, SEQLGBM, SFSEQLGBM).

The results can be obtained by running the following command:
#### SEQLGBM
```
seqlr --GBM -t ../data/train_val.seql -s ../data/test.seql -c config.json -v /dev/null -n SEQLGBM
```
#### SFSEQLGBM
```
seqlr --GBM -t ../data/train_val.SF.seql -s ../data/test.SF.seql -c config.json -v /dev/null -n SFSEQLGBM
```
#### SEQL
```
seqlr -t ../data/train_val.seql -s ../data/test.seql -c config.json -v /dev/null -n SEQL
```
#### SFSEQL
```
seqlr -t ../data/train_val.SF.seql -s ../data/test.SF.seql -c config.json -v /dev/null -n SFSEQL
```

In case validation set shall be evaluated:
#### SFSEQL
```
seqlr -t ../data/train.SF.seql -s ../data/test.SF.seql -c config.json --Validate -v ../data/val.SF.seql -n SFSEQL
```

#### Resutls
Results are saved in the .eval.json file.


### UCR Archive
#### Requirements
Besides SEQLGBM the preprocessing of the UCR Archive has following requirements:

- Matlab; to extract the static features
- saxconvert from (https://github.com/lnthach/Mr-SEQL); for the SAXTransformation
- R; to run glmnet

#### Run
We provide a simple script to run a particular experiment on all timeseries problems in a particular folder. The script is available in `UCR/scripts/run_all.py`. It uses the following syntax:
```
python run_all.py -d <DATADIR> -a <SAXSEQL|SFSEQL|SEQLGBM|SFSEQLGBM|GLMNET> [-pp]
```

The `-pp` flag is used to enable preprocessing without it preproecessed files are expected to be found in the same directory.
The script will recursively search for `.tsv` files within the `DATADIR` and apply necessary preprocessing steps followed by running the indicated algorithm on the files.

Example for SFSAXSEQLGBM:
```
python run_all.py -d data -a SFSEQLGBM -pp
```

<!-- #### Results -->
<!-- To collect the results -->

## Install requirements
Here we outline the installation of Aramadillo and JSON library that SEQL relies on. This is more a guideline thatn a proper tutorial.
Please adjust the procedure to your machine and needs.

### Aramadillo
Armadillo is a Linear algebra and scientific computing library for C++.
We first install OpenBlas since Armadillo strongly sugest to have it installed.

#### OpenBLAS
First install openBLAS if not installed already.
```
mkdir OpenBLAS
wget http://github.com/xianyi/OpenBLAS/archive/v0.2.20.tar.gz
tar xvf v0.2.20.tar.gz
cd OpenBLAS-0.2.20
make
make install
```

#### Armadillo
Get Armadillo from http://arma.sourceforge.net/ and compile it.
```
mkdir armadillo
cd armadillo
wget http://sourceforge.net/projects/arma/files/armadillo-8.600.0.tar.xz
tar xvf armadillo-8.600.0.tar.xz
cd armadillo-8.600.0
mkdir build
cd build 
cmake ..
make
make install
```

## JSON for Modern C++
A json library for C++.
Get the source from https://github.com/nlohmann/json, compile and install.
```
mkdir json
cd json
git clone https://github.com/nlohmann/json.git --depth=1
cd json
mkdir build
cd build
cmake ..
make
make install
```

## Catch2
Catch2 is a testing library for C++.
Hence Catch2 is only needed when you intend to run the unit tests.
To enable test run `cmake` with  the option `-DBUILD_TESTING=ON`.

```
mkdir catch2
cd catch2
git clone https://github.com/catchorg/Catch2.git --depth=1
cd catch2
mkdir build
cd build
cmake .
make
make install
cd ../../../
```

To run the test run:
```
bin/test_seql
```
