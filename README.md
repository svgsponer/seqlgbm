# SEQLGBM and SFSEQL
This repository contains code for **SEQ**uence **L**earner **G**radient **B**oosting **M**achine and **S**tatic **F**eature **SEQ**uence **L**earner as well as for reproducing the experiments explained in the paper:

*Accurate Sequence Classification by Gradient Boosting with Linear Models by Severin Gsponer, Thach Le Nguyen, and Georgiana Ifrim*

Abstract:
Sequence classification is the problem of mapping a given sequence of symbols to a target class, and has important applications in text mining, biological sequence classification (e.g., DNA, proteins) and more recently in time series classification.
State-of-the-art sequence classification algorithms rely on modeling the sequences based on the presence of certain patterns called *k*-mers.
Despite the success of such techniques a major drawback of most such methods is that they have to limit their representation to a rather small *k* since the number of possible *k*-mers grows exponentially with *k*.
Furthermore, such string features are rarely combined with available domain-specific numeric features which are often indispensable to achieve good performance.
The contributions of this paper are twofold.
First, we present a generalization of an existing linear sequence learning algorithm *SEQL* which is able to efficiently operate in the space of all possible *k*-mers.
We generalize the ideas of *SEQL* by formulating a new gradient boosting algorithm called *SEQLGBM*, that relies on iterative selection of linear models.
This generalization allows us to learn linear models in the space of all possible *k*-mers for arbitrary differentiable objective functions.
Second, we extend *SEQL* by adapting the underlying learning procedure to combine numeric and string features.
The composable nature of our proposed gradient boosting algorithm allows us to make immediate use of this extension.
We present experiments on a protein solubility prediction task, as well as on the largest available benchmark for time series classification.
Our comparison to the state-of-the-art shows that our proposed algorithm delivers more accurate results, while having the advantage that the two feature spaces (numeric and all sub-strings) are combined in a principled fashion within a single learning algorithm.



## Install SEQLGBM
Clone the repository by running:
`git clone https://github.com/svgsponer/SEQL.git`

The repository consists of three main folders:
- seql: `SEQLGBM` source code 
- Protsol: Data and scripts related to the protein solubility experiment
- UCR: Some data and scripts related to the time series classification task


### Installation Requirements
Before you compile SEQLGBM make sure you have all requirments installed:
- C++17 compiler (e.g., >=gcc-7 or >=clang-7)
- CMake (version >= 3.8)
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
You can either call it with the full path or add the `bin` directory to your `$PATH` envioment variable.
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
- R; with glmnet package installed

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

#### Results
You can use the script `scripts/collect_results.py` to collect results across mutliple datasets and safe them in a csv file.
Use the following syntax:
```
python collect_results -b <base_csv> -o <output_file> --item <path> <name> [--item <path> <name>]
```
The collected results are joined with the data already present in base_csv, this can be used to add already collected results. The item path will be searched for result files and each one added.
The final output is a csv with the error for each algorithm and dataset.

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
