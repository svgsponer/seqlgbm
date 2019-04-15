import click
import pandas as pd
from pathlib import Path
import json
from types import SimpleNamespace
import subprocess


def collect_folders(bp):
    files = list(bp.rglob("*.tsv"))
    folders = set()
    for f in files:
        folders.add(str(f)[0:str(f).rfind("_")])
    return folders


def collect_results_glmnet(bp):
    ret = []
    # folders = collect_folders(bp)
    files = bp.rglob("*.eval.json")
    for f in files:
        e = SimpleNamespace()
        line = subprocess.check_output(
            ['tail', '-1', f]).decode("ASCII")
        try:
            e.err = line.split()[1]
        except:
            print("no error defined for: ", f)
            continue
            e.err = "1"
        e.fname = f.parent.name
        ret.append((e.fname, e.err))
    return ret


def collect_results_SEQL(bp):
    ret = []
    files = bp.rglob("*.eval.json")
    for f in files:
        fp = Path(f)
        with open(fp) as ef:
            res = json.load(ef)
        try:
            err = res["Error"]/100
        except KeyError:
            err = 1 - res["Accuracy"]

        name = fp.parent.name
        ret.append((name, err))
    return ret


@click.command()
@click.option("-b", "--base_df")
@click.option("-o", "--output", type=click.Path(file_okay=True, writable=True), required=True)
@click.option('--item', nargs=2, type=click.Tuple([str,str]), multiple=True, required=True)
def cli(base_df, output,item): 
    if not base_df is None:
        print('Base dataframe to use: %s' % base_df)
        df = pd.read_csv(Path(base_df).expanduser().resolve())
        df = df.set_index('Dataset')
    else:
        df = pd.DataFrame()

    for p, n in item:
        print('Collecting results for %s in: %s' % (n, p))
        p = Path(p).resolve()
        ret = collect_results_SEQL(p)
        n_df = pd.DataFrame(ret, columns=['Dataset', n])
        n_df = n_df.set_index('Dataset')
        print("Found %s entries for %s" % (n_df.shape[0], n))
        if df.empty:
            df = n_df.join(df, how='outer')
        else:
            df = n_df.join(df, how='inner')

    print("Total entries: %s " % df.shape[0])
    df.to_csv(Path(output).resolve())


if __name__ == '__main__':
    cli()
