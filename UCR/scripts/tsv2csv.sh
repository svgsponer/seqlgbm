#/bin/bash

sed 's/\t/,/g' $1 > ${1%.*}.csv 
