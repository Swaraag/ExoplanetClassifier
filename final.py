import pandas as pd
import matplotlib as plt

df = pd.read_csv("cumulative_2026.03.09_14.43.47.csv", comment="#")
df_candidates = df[df["koi_disposition"] == "CANDIDATE"]
df = df[df["koi_disposition"] != "CANDIDATE"]

print(df["koi_disposition"].value_counts())
print(df.isnull().sum())
print(df[(df["kepler_name"].notnull()) & (df["koi_disposition"] == "FALSE POSITIVE")])