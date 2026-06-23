"""Build experiments/data/nhanes.csv from the CDC NHANES 2017-2018 public release.

Downloads the demographics (DEMO_J) and body-measures (BMX_J) SAS-XPT files, merges
on the respondent id, keeps adults (>=18) with a measured height and weight, and writes
columns: height_cm, weight_kg, sex.  Requires pandas (for SAS-XPT parsing); the
experiment script run_heightweight.py only needs the resulting plain CSV.
"""
import os, ssl, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "nhanes.csv")
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
CTX = ssl._create_unverified_context()


def fetch(name):
    dst = os.path.join("/tmp", name)
    if not os.path.exists(dst):
        with urllib.request.urlopen(f"{BASE}/{name}", context=CTX) as r, open(dst, "wb") as f:
            f.write(r.read())
    return dst


demo = pd.read_sas(fetch("DEMO_J.xpt"))[["SEQN", "RIAGENDR", "RIDAGEYR"]]
bmx = pd.read_sas(fetch("BMX_J.xpt"))[["SEQN", "BMXHT", "BMXWT"]]
df = demo.merge(bmx, on="SEQN")
df = df[df.RIDAGEYR >= 18].dropna(subset=["BMXHT", "BMXWT"])
df["sex"] = df.RIAGENDR.map({1: "M", 2: "F"})
out = df[["BMXHT", "BMXWT", "sex"]].rename(columns={"BMXHT": "height_cm", "BMXWT": "weight_kg"})
out.to_csv(OUT, index=False)
print(f"wrote {OUT}: {len(out)} adults (F={sum(out.sex == 'F')}, M={sum(out.sex == 'M')})")
