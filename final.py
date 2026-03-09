import pandas as pd

df = pd.read_csv("cumulative_2026.03.09_14.43.47.csv", comment="#")
df_candidates = df[df["koi_disposition"] == "CANDIDATE"]
df = df[df["koi_disposition"] != "CANDIDATE"]