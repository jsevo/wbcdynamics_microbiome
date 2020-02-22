#def pre_reg(WBCTYPE, period)
WBCTYPE = "neutrophils"
LEVEL = "genus"
#load modules
import arviz as az
import getpass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pymc3 as pm
import warnings
from sklearn.preprocessing import MinMaxScaler

#### LOAD DATA TABLES, FILTER DATA FOR QUALIFYING SAMPLES...
# username = getpass.getuser()
# treatments_all = pd.read_csv(
#     '/Users/%s/projects/data/wbc/cMED.csv' % username, )
# wbc_all = pd.read_csv(
#     '/Users/%s/projects/data/wbc/cWBC.csv' % username, )
# wbc_all["day"] = wbc_all["anon-date"] - wbc_all["anon-bmt-date"]
# ...

#### MAKE REGRESSION TABLES
# calculate outcomes "y", i.e. daily rates of white blood cell changes (dw) and interval predictors (dtreatments: immunomodulatory medications administered in interval, gm_wbc: the current state of the blood, dinfections: positive blood cultures in interval, pidmeta: patient / HCT meta data)


#### STAGE 1 FEATURE SELECTION REGRESSION
# also: stage 2 in regularized ElasticNet regression instead of Bayesian for internal checks
def regression_stage1(dmi,
                      divs,
                      dtreatments,
                      gm_wbc,
                      dinfections,
                      pidmeta,
                      dw,
                      wbc_type='neutrophils'):
    """Stage 1 Elastic Net regression
    dmi: microbiota composition for intervals
    divs: microbiota diversity for intervals, contains sample additional sample info (pid)
    dtreatments: immunomodulatory medications and treatments administered during interval
    gm_wbc: geometric mean of absolute white blood cell counts during interval
    dinfections: positive blood cultures detected during interval
    pidmeta: patient and HCT meta data
    dw: daily change in WBC count, "y"
    wbc_type: WBC type considered
    """
    # identify patients with microbiota data (stage 2) to exclude in stage 1
    # by inner-joining with microbiota diversity table
    Xmi = pd.merge(divs, dw, on=["pid", "anon-date"], how="inner")
    mi_pids = Xmi.reset_index().pid.unique()
    # stage 1 feature selection regression (and stage 2 as regularized ML version instead of full Bayesian for internal checks)
    for MIINDICATOR in ["stage1", "stage2"]:
        X = pd.merge(
            dw,
            gm_wbc,
            on=["pid", "anon-date"],
            how="inner",
            suffixes=["", "_REMOVEgmwbc"])
        if MIINDICATOR == "stage2":
            # ML-version of the Bayesian model for stage 2, include microbiome
            X = pd.merge(
                X,
                dmi,
                on=["pid", "anon-date"],
                how="inner",
                suffixes=["", "_REMOVE_dmi"])
            X = pd.merge(
                X,
                divs.reset_index()[["pid", "anon-date", "inverseSimpson"]],
                on=["pid", "anon-date"],
                how="inner",
                suffixes=["", "_REMOVE_ivs"])
        X = pd.merge(
            X,
            dtreatments,
            on=["pid", "anon-date"],
            how="left",
            suffixes=["", "_REMOVEdreatments"])
        X = pd.merge(
            X,
            dinfections,
            on=["pid", "anon-date"],
            how="left",
            suffixes=["", "_REMOVEdinfections"])
        X = pd.merge(
            X,
            pidmeta,
            on=["pid", "n_bmt"],
            how="left",
            suffixes=["", "_REMOVEpidmeta"])
        X = X.loc[~X["log_%s" % WBCTYPE].isna()]
        X = X[[x for x in X.columns if "REMOVE" not in x]]
        X = X.drop(columns=["n_bmt"])
        X_columns = X.columns

        if MIINDICATOR == "stage1":
            # mi_pids: patients with microbiota data
            X = X.loc[X.pid.apply(lambda v: v not in mi_pids)]
        elif MIINDICATOR == "stage2":
            # ML-version of the Bayesian model for stage 2
            X = X.loc[X.pid.apply(lambda v: v in mi_pids)]

        # intercepts per transplant type
        X = X.join(pd.get_dummies(X["hct_source"]))
        # drop original HCT type column
        X = X.drop(columns=['hct_source', "PBSC"])  #PBSC as reference
        # intercepts per intensity
        X = X.join(pd.get_dummies(X["Intensity"].fillna("unknown")))
        X = X.drop(columns=["ABLATIVE", "unknown",
                            "Intensity"])  #ABLATIVE as reference
        # intercepts female
        X = X.join(pd.get_dummies(X.sex)['F'])
        X = X.drop(columns='sex')

        # only after engraftment and before day 100
        X = X.query('day>6 ').copy()  #smallest observed engraftment da
        # only analyze daily changes
        X = X.query('dt == 1').copy()
        #
        # from join, fill gaps
        X.loc[:, dinfections.columns] = X[dinfections.columns].fillna(0)
        X.loc[:, dtreatments.columns] = X[dtreatments.columns].fillna(0)
        # missing patient ages, fill with mean for ML feature selection
        X["age"] = X["age"].fillna(X["age"].mean())

        ### drop columns
        # delta time columns
        dtcols = [x for x in X.columns if 'dt' in x]
        X = X.drop(columns=dtcols)
        # time point columns
        anoncols = [x for x in X.columns if 'anon' in x]
        X = X.drop(columns=anoncols)
        # patient id columns
        pidcols = [x for x in X.columns if ('pid' in x) and (x != 'pid')]
        remaining_pids = X.reset_index().pid.unique()
        print("pid count in regression", len(remaining_pids))
        X = X.drop(columns=pidcols)
        print("shape before dropna", X.shape)
        X = X.dropna()
        print("shape after dropna (should not change)", X.shape)

        # drop all zero columns
        drop_zero_columns = (X.sum() == 0) & (X.max() == 0)
        drop_zero_columns = drop_zero_columns.loc[
            drop_zero_columns.values].index
        X = X.drop(columns=drop_zero_columns)
        # drop HCT day
        daycolumns = [
            x for x in X.columns if ("day" in x) and (x not in ["day", "eday"])
        ]
        X.drop(columns=daycolumns, inplace=True)

        #### data transformations and standardizations
        from sklearn.preprocessing import StandardScaler
        X.loc[:, [
            'neutrophils', 'lymphocytes', 'monocytes', 'eosinophils',
            'platelets'
        ]] = np.log10(X[[
            'neutrophils', 'lymphocytes', 'monocytes', 'eosinophils',
            'platelets'
        ]] + 0.1 / 2)
        from sklearn.linear_model import ElasticNetCV
        # Fit and summarize OLS model
        # CV on patient sub samples, X contains 'pid' column
        Xpid = X.copy()
        X = X.drop(columns=['pid'])
        _drug_cols = [x for x in X.columns if x in dtreatments.columns]
        _other_cols = [x for x in X.columns if x not in dtreatments.columns]
        groups = Xpid.pid
        from sklearn.model_selection import GroupKFold
        group_kfold = GroupKFold(n_splits=10)
        cv = list(
            group_kfold.split(
                X.drop(columns='log_%s' % wbc_type),
                X['log_%s' % wbc_type],
                groups))
        mod = ElasticNetCV(
            cv=cv, positive=False, normalize=False, fit_intercept=True)
        res = mod.fit(
            MinMaxScaler().fit_transform(X.drop(columns='log_%s' % wbc_type)),
            X['log_%s' % wbc_type],
        )
        r2 = mod.score(
            MinMaxScaler().fit_transform(X.drop(columns='log_%s' % wbc_type)),
            X['log_%s' % wbc_type],
        )
        print('chosen alpha: %f' % mod.alpha_)
        print("R2 = %f" % r2)
        coefs = pd.Series(
            res.coef_,
            index=X.drop(columns='log_%s' % wbc_type).columns).replace(
                0, np.nan).dropna().sort_values()
        coefs["gr"] = mod.intercept_
        coefs["N"] = len(np.unique(Xpid.pid))
        coefs["n"] = X.shape[0]
        coefs["r2"] = r2
        coefs["alpha"] = mod.alpha_
    return (coefs)
