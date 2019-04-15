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
@click.option("-s", "--sfseql", type=click.Path(exists=True))
@click.option("-g", "--glmnet", type=click.Path(exists=True))
@click.option("-a", "--saxseql", type=click.Path(exists=True))
@click.option("-q", "--sfseqlgbm", type=click.Path(exists=True))
@click.option("-l", "--seqlgbm", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(file_okay=True, writable=True),
              required=True)
def cli(base_df, output, sfseql, glmnet, saxseql, sfseqlgbm, seqlgbm):
    if not base_df is None:
        print('Base dataframe to use: %s' % base_df)
        df = pd.read_csv(Path(base_df).expanduser().resolve())
        df = df.set_index('Dataset')
    else:
        df = pd.DataFrame()

    if not sfseql is None:
        print('Collecting results of SFSEQL in: %s' % sfseql)
        p = Path(sfseql).resolve()
        ret = collect_results_SEQL(p)
        sfseql_df = pd.DataFrame(ret, columns=['Dataset', 'SFSEQL'])
        sfseql_df = sfseql_df.set_index('Dataset')
        print("Found %s entries for SFSEQL" % sfseql_df.shape[0])
        if df.empty:
            df = sfseql_df.join(df, how='outer')
        else:
            df = sfseql_df.join(df, how='inner')

    if not sfseqlgbm is None:
        print('Collecting results of SFSEQLGBM in: %s' % sfseqlgbm)
        p = Path(sfseqlgbm).resolve()
        ret = collect_results_SEQL(p)
        sfseqlgbm_df = pd.DataFrame(ret, columns=['Dataset', 'SFSEQLGBM'])
        sfseqlgbm_df = sfseqlgbm_df.set_index('Dataset')
        print("Found %s entries for SFSEQLGBM" % sfseqlgbm_df.shape[0])
        if df.empty:
            df = sfseqlgbm_df.join(df, how='outer')
        else:
            df = sfseqlgbm_df.join(df, how='inner')

    if not seqlgbm is None:
        print('Collecting results of SEQLGBM in: %s' % seqlgbm)
        p = Path(seqlgbm).resolve()
        ret = collect_results_SEQL(p)
        sfseqlgbm_df = pd.DataFrame(ret, columns=['Dataset', 'SAX.SEQLGBM'])
        sfseqlgbm_df = sfseqlgbm_df.set_index('Dataset')
        print("Found %s entries for SAX.SEQLGBM" % sfseqlgbm_df.shape[0])
        if df.empty:
            df = sfseqlgbm_df.join(df, how='outer')
        else:
            df = sfseqlgbm_df.join(df, how='inner')

    if not glmnet is None:
        print('Collecting results of glment in: %s' % glmnet)
        p = Path(glmnet).resolve()
        glm = collect_results_glmnet(p)
        glm_df = pd.DataFrame(glm, columns=['Dataset', 'SFglmnet'])
        glm_df = glm_df.set_index('Dataset')
        print("Found %s entries for GLMNET" % glm_df.shape[0])
        if df.empty:
            df = glm_df.join(df, how='outer')
        else:
            df = glm_df.join(df, how='inner')

    if not saxseql is None:
        print('Collecting results of saxseql in: %s' % saxseql)
        p = Path(saxseql).resolve()
        ret = collect_results_SEQL(p)
        saxseql_df = pd.DataFrame(ret, columns=['Dataset', 'MSAXSEQL'])
        saxseql_df = saxseql_df.set_index('Dataset')
        print("Found %s entries for SAXSEQL" % saxseql_df.shape[0])
        if df.empty:
            df = saxseql_df.join(df, how='outer')
        else:
            df = saxseql_df.join(df, how='inner')

    print("Total entries: %s " % df.shape[0])
    df.to_csv(Path(output).resolve())


if __name__ == '__main__':
    cli()
