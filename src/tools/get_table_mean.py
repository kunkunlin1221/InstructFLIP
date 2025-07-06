import pandas as pd
from astropy.table import Table
from fire import Fire


def main(latex_fpath: str, mcio_only: bool = False):
    df = Table.read(latex_fpath).to_pandas()
    df.dropna(inplace=True)
    df.set_index('col0', inplace=True)
    if mcio_only:
        df = df.iloc[1:5]
    else:
        df = df.iloc[1:]
    avg = pd.DataFrame([df.mean().tolist()], index=['mean'], columns=df.columns).round(2)
    print(avg)


Fire(main)
