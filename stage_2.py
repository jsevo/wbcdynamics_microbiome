period = "total"
WITHIVS = ""
WBCTYPE = "neutrophils"  #"lymphocytes" #"monocytes"
LEVEL = "genus"
#load modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pymc3 as pm
import warnings
username = getpass.getuser()

def _hdi(a):
    """calculate HDIs to extent pymc3 summary"""
    def hfx(x):
        return (pd.DataFrame(
            pm.hpd(x, alpha=a),
            columns=[
                'hdi_%.1f' % (a / 2 * 100),
                'hdi_%.1f' % (100 - a / 2 * 100)
            ]))

    return (hfx)

# patients stratified by graft stem cell source ("transplant type", TPT)
for TPT in ["PBSC", "TCD", "BM", "cord"]:
    ## load "stage 1" feature selection results
    column_metadata = pd.read_csv(
        "/Users/%s/projects/data/wbc/aggregated_metadata_%s.csv" % (username,
                                                                    WBCTYPE), )
    column_metadata.loc[[
        "neutrophils", "lymphocytes", "monocytes", "eosinophils", "platelets"
    ], "metadata"] = "wbc"
    X_data = pd.read_csv(
        "/Users/%s/projects/data/wbc/X_data_%s.csv" % (username, WBCTYPE),
        index_col="Unnamed: 0")
    X_data_stage1 = X_data.copy()
    from sklearn.linear_model import Stage1CV, Stage1, ElasticNetCV, ElasticNet, LinearRegression
    from sklearn.model_selection import GroupKFold
    group_kfold = GroupKFold(n_splits=10)
    X = X_data_stage1[np.unique(
        ["log_%s" % WBCTYPE] + ["pid", "TCD", "BM", "cord"] +
        column_metadata.query("include==True").index.to_list())].copy()

    # log transform microbiota and WBC predictors
    X[[
        "neutrophils", "lymphocytes", "monocytes", "eosinophils", "platelets"
    ]] = np.log10(X[[
        "neutrophils", "lymphocytes", "monocytes", "eosinophils", "platelets"
    ]] + 0.1 / 2.)
    genus_cols = column_metadata.query(
        "metadata == 'genus' and include==True").index.to_list()
    X[genus_cols] = np.log10(X[genus_cols] + 2e-6)

    # stratify data per TPT
    if TPT == "PBSC":
        X = X.loc[(X.cord != 1.0) & (X.TCD != 1.0) & (X.BM != 1.0)]
    elif TPT == "TCD":
        X = X.loc[(X.cord != 1.0) & (X.TCD == 1.0) & (X.BM != 1.0)]
    elif TPT == "cord":
        X = X.loc[(X.cord == 1.0) & (X.TCD != 1.0) & (X.BM != 1.0)]
    elif TPT == "BM":
        X = X.loc[(X.cord != 1.0) & (X.TCD != 1.0) & (X.BM == 1.0)]
    X = X.drop(
        columns=[x for x in ["PBSC", "TCD", "BM", "cord"] if x in X.columns])

    X = X.drop(columns=["pid", "gr"])

    with pm.Model() as model:
        #### priors
        # base line maximum specific rate, i.e. intercept
        gr = pm.Normal("gr", 0, sigma=100)
        # predictor coefficients
        beta = pm.Normal("beta", 0, sigma=100, shape=X.shape[1] - 1)
        # model uncertainty
        sigma = pm.HalfCauchy(
            "model_uncertainty", 2
        )  # equivalent results with sigma = pm.InverseGamma("model_uncertainty", 1,1)
        # linear model
        mu = pm.Deterministic(
            "mu", gr + pm.math.dot(
                X.drop(columns=["log_%s" % WBCTYPE]).values.astype(np.float),
                beta))
        y_observed = pm.Normal(
            'y_observed',
            mu=mu,
            sigma=sigma,
            observed=X["log_%s" % WBCTYPE].astype(np.float).values)

        #### MCMC sampling
        prior_samples_withmi = pm.sample_prior_predictive(vars=["mu"])
        # uses No-U-turn sampling (NUTS) by default
        posterior_withmi = pm.sample(
            draws=2000, chains=5, cores=1, n_init=1000)
        # save posterior
        pm.save_trace(
            posterior_withmi,
            "/Users/%s/projects/results/%s_posterior_%s" % (WBCTYPE, TPT),
            overwrite=True,
        )
        summary_ = pm.summary(
            posterior_withmi,
            stat_funcs=[
                _hdi(0.05),
                _hdi(0.1),
                _hdi(0.16),
                _hdi(0.5),
            ],
            extend=True,
            varnames=["beta"])
        summary_.to_csv(
            "/Users/%s/projects/results/wbc/%s_posterior_%s_summary.csv" %
            (WBCTYPE, TPT))
