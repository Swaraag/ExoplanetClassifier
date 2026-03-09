import pandas as pd

def pre_process():
    """Removes unnecessary columns from the df data and stores other potentially important columns in other dataframes"""
    raw_df = pd.read_csv("cumulative_2026.03.09_14.43.47.csv", comment="#")
    # storing candidates separately for testing at the end
    df_candidates = raw_df[raw_df["koi_disposition"] == "CANDIDATE"]
    df = raw_df[raw_df["koi_disposition"] != "CANDIDATE"]

    # removing leakage - they contain the answer for the classifier already, as well as admin/meta info
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
    return df, df_candidates, df_misc