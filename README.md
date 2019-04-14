# SEQLGBM and SFSEQL
Repository for SEQuence Learner Gradient Boosting Machine and Static Feature SEQuence Learner

Content will follow soon.

## Install

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
