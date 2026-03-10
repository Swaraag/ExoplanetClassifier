import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from xgboost import XGBClassifier

def pre_process():
    """Removes unnecessary columns from the df data and stores other potentially important columns in other dataframes"""
    raw_df = pd.read_csv("cumulative_2026.03.09_14.43.47.csv", comment="#")
    # storing candidates separately for testing at the end
    df = raw_df.copy()

    # removing leakage - they contain the answer for the classifier already, as well as admin/meta info
    # koi_score is the biggest issue because it provides NASA's precomputed likelihood score for likelihood of being an exoplanet
    df = df.drop(columns=["koi_score", "koi_pdisposition", "koi_fpflag_nt", "koi_fpflag_ss", 
                        "koi_fpflag_co", "koi_fpflag_ec", "koi_disp_prov", "koi_vet_stat", 
                        "koi_vet_date", "koi_comment"])
    # keep for later but don't need in the df itself
    df_misc = df[["kepid", "kepoi_name", "kepler_name"]]
    df = df.drop(columns=["kepid", "kepoi_name", "kepler_name"])
    # other non-useful data to drop
    df = df.drop(columns=["koi_quarters", "koi_tce_plnt_num", "koi_tce_delivname", "ra", "dec"])
    # more irrelavant data
    df = df.drop(columns=[
        'koi_time0bk', 'koi_time0',
        'koi_eccen', 'koi_longp',
        'koi_fittype', 'koi_limbdark_mod', 'koi_trans_mod', 'koi_parm_prov', 'koi_sparprov',
        'koi_ldm_coeff1', 'koi_ldm_coeff2', 'koi_ldm_coeff3', 'koi_ldm_coeff4',
        'koi_datalink_dvr', 'koi_datalink_dvs',
        'koi_fwm_stat_sig', 'koi_fwm_sra', 'koi_fwm_sdec', 'koi_fwm_srao',
        'koi_fwm_sdeco', 'koi_fwm_prao', 'koi_fwm_pdeco',
        'koi_dicco_mra', 'koi_dicco_mdec', 'koi_dicco_msky',
        'koi_dikco_mra', 'koi_dikco_mdec', 'koi_dikco_msky',
        'koi_count',
        'koi_kepmag', 'koi_gmag', 'koi_rmag', 'koi_imag',
        'koi_zmag', 'koi_jmag', 'koi_hmag', 'koi_kmag'
    ])

    # these 4 columns only have null values anyways
    df = df.drop(columns=["koi_sage", "koi_ingress", "koi_model_chisq", "koi_model_dof"])

    # final 
    df_cand = df[df["koi_disposition"] == "CANDIDATE"]
    df = df[df["koi_disposition"] != "CANDIDATE"]
    return raw_df, df, df_cand, df_misc

def tt_split(df):
    X = df.drop(columns=["koi_disposition", "rowid"])
    y = df["koi_disposition"].replace({"CONFIRMED": 1, "FALSE POSITIVE": 0}).astype(int)
    return X, y, *train_test_split(X, y, test_size=0.2, random_state=42)

def build_rfc_pipeline():
    # using median imputation for the SimpleImputer because 
    imputer = SimpleImputer(strategy='median')
    # class_weight = 'balanced' solves the problem where there are more False positives than there are confirmed values
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    pipeline = Pipeline(steps=[('imputer', imputer), ('RFC_model', model)])
    return pipeline

def build_xgb_pipeline(y):
    # using median imputation for the SimpleImputer because 
    imputer = SimpleImputer(strategy='median')
    # XGBClassifier doesn't have class_weight='balanced' like scikit-learn models, but they have alternatives with little more work
    model = XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight = (y==0).sum()/(y==1).sum())
    pipeline = Pipeline(steps=[('imputer', imputer), ('XGB_model', model)])
    return pipeline

def fit_predict(pipeline, X_train, y_train, X_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return y_pred

def pred_cand(pipeline, df_cand, df_misc, raw_df):
    df_misc_cand = df_misc[raw_df["koi_disposition"] == "CANDIDATE"]
    df_cand = df_cand.drop(columns=["koi_disposition", "rowid"])
    cand_pred = pipeline.predict(df_cand)
    cand_prob = pipeline.predict_proba(df_cand)[:,1]
    df_cand.insert(loc=2, column="prediction", value=cand_pred)
    df_cand.insert(loc=2, column="prediction_prob", value=cand_prob)
    df_cand.insert(loc=2, column="kepoi_name", value=df_misc_cand["kepoi_name"].values)

    koi_disp_map = {0: "NOT EXOPLANET", 1: "EXOPLANET"}
    df_cand["prediction"] = df_cand["prediction"].map(koi_disp_map)

    return cand_pred, cand_prob, df_cand