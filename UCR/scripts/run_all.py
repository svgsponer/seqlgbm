import subprocess
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from os import chdir
import sys
import progressbar
import json
from scipy.stats import wilcoxon
from tim import Tim
import click
SCRIPTPATH = Path(__file__).resolve().parent
CWD = Path.cwd()
MATLABDIR = Path.joinpath(SCRIPTPATH, "matlab/")
R_SCRIPT =  Path.joinpath(SCRIPTPATH, "GLM_TS.r")

tsv2csv_cmd = "sed 's/\t/,/g' %s > %s"
R_cmd = "Rscript --vanilla  %s %s > %s"
ts_cmd = "awk '/Time series/ {print $4}' %s"
sax_cmd = "sax_convert -i %s -o %s -n %s -N %s -w 16 -a 4 -m 1 > saxconfig"
cut_cmd = "cut -d\" \" -f1 --complement %s > %s"
paste_cmd = "paste -d\" \" %s %s > %s"
seql_cmd = "seqlr -t %s -s %s -c %s  > out"
seql_gbm_cmd = "seqlr -t %s -s %s -c %s --GBM > out"


def tsv2csv(fp):
    subprocess.call(tsv2csv_cmd % (fp, fp.with_suffix(".csv")), shell=True)


@Tim("Extract SF", "time")
def extract_sf(files):
    print("Extract SF")
    matlab_cmd = "matlab -nodisplay -nosplash -nodesktop -nojvm -r \"cd %s;" % MATLABDIR
    for fp in files:
        cmd = "svg_get_sf('%s');" % fp.with_suffix(".csv")
        matlab_cmd = matlab_cmd + cmd
    matlab_cmd = matlab_cmd + "exit;\""
    print(matlab_cmd)
    subprocess.call(matlab_cmd, shell=True)


def extract_sf_zs(files):
    print("Extract SF z normalized")
    matlab_cmd = "matlab -nodisplay -nosplash -nodesktop -nojvm -r \"cd %s;" % MATLABDIR
    for fp in files:
        cmd = "svg_get_sf_zs('%s');" % fp.with_suffix(".csv")
        matlab_cmd = matlab_cmd + cmd
    matlab_cmd = matlab_cmd + "exit;\""
    print(matlab_cmd)
    subprocess.call(matlab_cmd, shell=True)


@Tim("Create Sax", "time")
def create_sax(files):
    print("Create SAX representation")
    for fp in progressbar.progressbar(files):
        f = Path(fp)
        # print(fp)
        tslength = subprocess.check_output(
            ts_cmd % fp.with_name("README.md"), shell=True).decode("ASCII")
        # print("TS Length: %s" % tslength)
        if '\n' in tslength:
            tslength = tslength.split('\n')[0]
        try:
            wl = int(tslength) * 0.2
        except Exception:
            wl = 64
        subprocess.call(sax_cmd % (fp.with_suffix(".csv"),
                                   fp.with_suffix(".SAX.mrseql"), wl, wl+1), shell=True)
        subprocess.call(cut_cmd % (fp.with_suffix(".SAX.mrseql"),
                                   fp.with_suffix(".SAX.seql")), shell=True)
        subprocess.call(cut_cmd % (fp.with_suffix(".SAX.seql"),
                                   fp.with_suffix(".SAX")), shell=True)


@Tim("Combine rep", "time")
def combine(files):
    print("Combine SAX and SF features:")
    for fp in progressbar.progressbar(files):
        f = Path(fp)
        subprocess.call(paste_cmd % (fp.with_suffix(".SF"), fp.with_suffix(
            ".SAX"), fp.with_suffix(".tmp")), shell=True)
        subprocess.call("echo \#\"%s\" 1 18 1 > %s" %
                        (fp.stem, fp.with_suffix(".SF.SAX.seql")), shell=True)
        subprocess.call("cat %s >> %s" % (fp.with_suffix(
            ".tmp"), fp.with_suffix(".SF.SAX.seql")), shell=True)
        subprocess.call("sed -i 's/,/ /g' %s" %
                        fp.with_suffix(".SF.SAX.seql"), shell=True)


def collect_folders(bp):
    files = list(bp.rglob("*.tsv"))
    folders = set()
    for f in files:
        print("Process: ", f)
        folders.add(str(f)[0:str(f).rfind("_")])
    return folders


@Tim("SAXSEQL", "time")
def run_SAXSEQL(folders):
    print("Run SAXSEQL:")
    config_f = CWD.joinpath("config.cfg.json")
    for fld in progressbar.progressbar(folders):
        fp = Path(fld)
        name = fp.name
        tf = fp.with_name(name + "_TRAIN")
        testf = fp.with_name(name + "_TEST")
        # print("RUN: ",name, tf , testf, config_f)
        the_d = CWD.joinpath(name)
        the_d.mkdir()
        chdir(the_d)
        subprocess.call(seql_cmd % (tf.with_suffix(".SAX.seql"),
                                    testf.with_suffix(".SAX.seql"), config_f),  shell=True)
        chdir(CWD)

@Tim("SAXSEQLGBM", "time")
def run_SAXSEQLGBM(folders):
    print("Run SAXSEQLGBM:")
    config_f = CWD.joinpath("config.cfg.json")
    for fld in progressbar.progressbar(folders):
        fp = Path(fld)
        name = fp.name
        tf = fp.with_name(name + "_TRAIN")
        testf = fp.with_name(name + "_TEST")
        # print("RUN: ",name, tf , testf, config_f)
        the_d = CWD.joinpath(name)
        the_d.mkdir()
        chdir(the_d)
        subprocess.call(seql_cmd % (tf.with_suffix(".SAX.seql"),
                                    testf.with_suffix(".SAX.seql"), config_f),  shell=True)
        chdir(CWD)


@Tim("SFSEQL", "time")
def run_SFSEQL(folders):
    print("Run SFSEQL:")
    config_f = CWD.joinpath("config.cfg.json")
    for fld in progressbar.progressbar(folders):
        fp = Path(fld)
        name = fp.name
        tf = fp.with_name(name + "_TRAIN")
        testf = fp.with_name(name + "_TEST")
        # print("RUN: ",name, tf , testf, config_f)
        the_d = CWD.joinpath(name)
        the_d.mkdir()
        chdir(the_d)
        subprocess.call(seql_cmd % (tf.with_suffix(".SF.SAX.seql"),
                                    testf.with_suffix(".SF.SAX.seql"), config_f),  shell=True)
        chdir(CWD)


@Tim("SEQLGBM", "time")
def run_SFSEQLGBM(folders):
    print("Run SEQLGBM:")
    config_f = CWD.joinpath("config.cfg.json")
    for fld in progressbar.progressbar(folders):
        fp = Path(fld)
        name = fp.name
        tf = fp.with_name(name + "_TRAIN")
        testf = fp.with_name(name + "_TEST")
        # print("RUN: ",name, tf , testf, config_f)
        the_d = CWD.joinpath(name)
        the_d.mkdir()
        chdir(the_d)
        subprocess.call(seql_gbm_cmd % (tf.with_suffix(".SF.SAX.seql"),
                                    testf.with_suffix(".SF.SAX.seql"), config_f),  shell=True)
        chdir(CWD)


@Tim("GLMNET", "time")
def run_glmnet(folders):
    for fld in progressbar.progressbar(folders):
        fp = Path(fld)
        name = fp.name
        # print("RUN: ",name, tf , testf, config_f)
        the_d = CWD.joinpath(name)
        the_d.mkdir()
        chdir(the_d)
        subprocess.call(R_cmd % (R_SCRIPT, fld+"_", the_d.joinpath("glmnet.eval.json")), shell=True)
        chdir(CWD)


@Tim("Total GLMNET", "time")
def GLMNET(folders, pp, files):
    if pp:
        extract_sf(files)
    run_glmnet(folders)


@Tim("Total SAXSEQL", "time")
def SAXSEQL(folders, pp,files):
    if pp:
        create_sax(files)
    run_SAXSEQL(folders)

@Tim("Total SAXSEQLGBM", "time")
def SAXSEQLGBM(folders, pp, files):
    if pp:
        create_sax(files)
    run_SAXSEQLGBM(folders)

@Tim("Total SFSEQL", "time")
def SFSEQL(folders, pp):
    if pp:
        extract_sf(files)
        create_sax(files)
        combine(files)
    run_SFSEQL(folders)

@Tim("Total SFSAXGBM", "time")
def SFSEQLGBM(folders, pp,files):
    if pp:
        extract_sf(files)
        create_sax(files)
        combine(files)
    run_SFSEQLGBM(folders)

@click.command()
@click.option('-d', '--data-dir', type=click.Path(exists=True), required=True)
@click.option('-a', '--algo', type=click.Choice(['SFSEQL', 'SAXSEQL', 'GLMNET', 'SFSEQLGBM', 'SAXSEQLGBM']), prompt="Which method do you want to use:")
@click.option('-p', '--preprocess', 'pp', is_flag=True,  default=False)
def cli(data_dir, algo, pp):
    bp = Path(data_dir).expanduser().resolve()
    files = list(bp.rglob("*.tsv"))
    folders = set()

    if pp:
        print("Convert tsv -> csv")

    with Tim("tsv2csv", "time"):
        for f in progressbar.progressbar(files):
            if pp:
                tsv2csv(f)
            folders.add(str(f)[0:str(f).rfind("_")])
        
    if algo == 'SFSEQL':
        SFSEQL(folders, pp, files)
    elif algo == 'SAXSEQL':
        SAXSEQL(folders, pp, files)
    elif algo == 'SFSEQLGBM':
        SFSEQLGBM(folders, pp, files)
    elif algo == 'SAXSEQLGBM':
        SAXSEQLGBM(folders, pp, files)
    elif algo == 'GLMNET':
        GLMNET(folders, pp)
    else:
        print("Algo not know")


if __name__ == '__main__':
    cli()
