##run in the deep learning conda env
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from scipy.stats import mode
import statsmodels.api as sm
import itertools
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
from LabData.DataLoaders.GutMBLoader import GutMBLoader
from LabData.DataLoaders.SubjectLoader import SubjectLoader
from LabData.DataLoaders.DEXALoader import DEXALoader
from LabData.DataLoaders.BloodTestsLoader import BloodTestsLoader
from LabData.DataLoaders.RetinaScanLoader import RetinaScanLoader
from LabData.DataLoaders.UltrasoundLoader import UltrasoundLoader
from LabData.DataLoaders.ABILoader import ABILoader
from LabData.DataLoaders.DEXALoader import DEXALoader
from LabData.DataLoaders.ECGTextLoader import ECGTextLoader
from statsmodels.discrete.discrete_model import Logit
from sklearn.metrics import roc_auc_score
import os
from LabData.DataLoaders.ECGTextLoader import ECGTextLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from LabData.DataLoaders.MedicalConditionLoader import MedicalConditionLoader
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, homogeneity_score, \
    completeness_score
from matplotlib.colors import to_rgb
from sklearn.model_selection import StratifiedKFold, KFold
from torch.distributions import Normal
import seaborn as sns

def make_diseases_df(all_inds, id_list = list(range(100)) + list(range(1000, 1011, 1)), only_baseline = False):
    ##Only consider reported cases at baseline
    b = MedicalConditionLoader().get_data(study_ids=id_list).df.reset_index()
    b = b.sort_values(by = "Date")
    if only_baseline:
        b = b.loc[~b["RegistrationCode"].duplicated(keep = "first"),:]
    counts = b.set_index("RegistrationCode").loc[list(set(all_inds).intersection(set(b.RegistrationCode.values)))].medical_condition.value_counts()
    diseases = list(counts.index[counts/len(counts) >= 0.02])
    ##not worth predicting:
    diseases.remove("6A05") ##ADHD
    diseases.remove("RA01") ##covid -not reliable
    ##remove depression
    diseases = list(filter(lambda x: not x.startswith("Block"), diseases))
    res = pd.DataFrame(np.zeros([len(list(set(all_inds))), len(diseases)]), dtype = int)
    res["RegistrationCode"] = list(set(all_inds))
    res = res.set_index(["RegistrationCode"])
    res.columns = diseases
    b = b.set_index("RegistrationCode").loc[list(set(all_inds).intersection(set(b["RegistrationCode"])))].reset_index()
    for cond_string in diseases:
        people_with_disease = list(set(b.loc[b.medical_condition == cond_string, :]["RegistrationCode"]))
        res.loc[people_with_disease, cond_string] = 1
    return res

def cv(x, features, what = "bmi", task = "r", how = "ols"):
    test_preds_dict = {}
    true_dict = {}
    if what in x.columns: x = x.drop([what], axis = 1)
    x_merged = x.merge(features[what].astype(float), left_index=True, right_index=True)[[what] + list(x.columns)]
    x_merged = x_merged.fillna(0)
    y = x_merged[what].values
    x_merged =x_merged[x.columns]
    x_index = x_merged.index.values.copy() ##store the original index here
    x_merged = x_merged.values
    splitter = StratifiedKFold if task == "c" else KFold
    train_splitter = splitter(n_splits=5, shuffle=True, random_state=42)
    print("Using " + str(len(x)))
    for train_inds, test_inds in train_splitter.split(x_merged, y):
        X_train, Y_train = x_merged[train_inds, :], y[train_inds]
        X_test, Y_test = x_merged[test_inds, :], y[test_inds]
        test_people = x_index[test_inds]
        if task == "r":
            if how == "ols":
                model = sm.regression.linear_model.OLS(Y_train, sm.add_constant(X_train, has_constant='skip')).fit()
                test_preds_dict[tuple(test_people)] = model.predict(sm.add_constant(X_test, has_constant='skip'))
            else:
                model = LGBMRegressor(max_depth=3, learning_rate=0.01, n_estimators=500)
                model.fit(X_train, Y_train)
                test_preds_dict[tuple(test_people)] = model.predict(X_test)
        else:
            if how == "ols":
                model = Logit(Y_train, sm.add_constant(X_train, has_constant='skip')).fit_regularized()
                test_preds_dict[tuple(test_people)] = model.predict(sm.add_constant(X_test, has_constant='skip'))
            else:
                model = LGBMClassifier(max_depth=3, learning_rate=0.01, n_estimators=500)
                model.fit(X_train, Y_train)
                test_preds_dict[tuple(test_people)] = model.predict(X_test)
        true_dict[tuple(test_people)] = Y_test
    test_preds = np.concatenate(list(test_preds_dict.values()))
    test_preds = pd.Series(test_preds)
    test_preds.index = np.concatenate(list(test_preds_dict.keys()))
    true = np.concatenate(list(true_dict.values()))
    true = pd.Series(true)
    true.index = np.concatenate(list(true_dict.keys()))
    if task == "r":
        res = pd.DataFrame({"preds": test_preds, "true": true})
    else:
        res = pd.DataFrame({"preds": test_preds, "true": true})
    return res, model

def make_age_df():
    dfmeta = GutMBLoader().get_data("segal_species", study_ids=[10]).df_metadata.reset_index().set_index(["RegistrationCode"])[["Date", "age"]]
    dfmeta = dfmeta.sort_values("Date", ascending = True)
    dfmeta = pd.DataFrame(dfmeta.loc[~dfmeta.index.duplicated(keep = "first"), :]["age"])
    return dfmeta

if __name__ == "__main__":
    do_blood = False
    if do_blood:
        bt_df = BloodTestsLoader().get_data().df
        bt_df_meta = BloodTestsLoader().get_data().df_metadata
        cbc_lab_ind = bt_df_meta[bt_df_meta['kupat_holim'] == 'tenk-cbc'].index
        bt_df_cbc = bt_df.loc[cbc_lab_ind].dropna(how='all', axis=1)
        bt_df_cbc_age = bt_df_meta.loc[cbc_lab_ind]['age'].reset_index().set_index("RegistrationCode")
        bt_df_cbc = bt_df_cbc.reset_index(drop=False).set_index("RegistrationCode")
        bt_df_cbc_age = bt_df_cbc_age.loc[~bt_df_cbc_age.index.duplicated(keep="last"), :]
        bt_df_cbc = bt_df_cbc.loc[~bt_df_cbc.index.duplicated(keep="last"), :]
        b = bt_df_cbc.loc[~bt_df_cbc.index.duplicated(), :].dropna()
    np.random.seed(1)
    do_emb = True
    if do_emb:
        orig_embs = pd.read_csv(
            "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/rna_foundation_model/test_emb.csv").rename(mapper = {"Unnamed: 0": "SampleName"}, axis = 1).set_index("SampleName")
        metadata = pd.read_csv(
            "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/10K_MicrobiomeModelling/DATA/10K_D2_Lifeline_deep_bf500_full_mb_metadata.csv")
        metadata.set_index("SampleName", inplace=True)
        orig_embs_with_regcode = orig_embs.merge(metadata, left_index = True, right_index = True).reset_index(drop = True).set_index("RegistrationCode")[orig_embs.columns]
        age = make_age_df()
        print(cv(orig_embs_with_regcode, age, "age", "r")[0].corr())
    do_comparator = False
    n_hvg = 14000
    if do_comparator:
        greg = pd.read_pickle(
            "/net/mraid20/ifs/wisdom/segal_lab/jasmine/RNA/rna_options_final/" + "after_batch_correction_filtered_1000_no_regress_5mln_sample.df")
        extra = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/RNA/metadata_with_techn.csv")
        extra["RegistrationCode"] = list(map(lambda x: "10K_" + str(x), extra.participant_id))
        extra.set_index("RegistrationCode", inplace=True)
        age = make_age_df(greg, extra)
        ##predict age from the top 1200 most variably expressed genes
        X = pd.read_pickle(
            "/net/mraid20/ifs/wisdom/segal_lab/jasmine/RNA/rna_options_final/" + "after_batch_correction_filtered_1000_no_regress_5mln_sample.df")
        X = np.log1p(np.clip(X, -0.999, None, )).loc[:, np.abs(X.std()).sort_values(ascending = False).index[0:n_hvg]]
        e = X.merge(age, left_index = True, right_index = True)
        e = e.loc[~e.index.duplicated(keep = "first"),:]
        batches = np.array_split(e.index.values, 5)
        test_preds = {}
        i = 0
        for test_indices in batches:
            train_indices = list(set(list(e.index)) - set(test_indices))
            if how == "ols":
                model = sm.OLS(e.loc[e.index, "age"].loc[train_indices].astype(float).values,
                           e.iloc[:, :n_hvg].loc[train_indices].values).fit()
            else:
                model = lgb.LGBMRegressor(max_depth = 3, learning_rate = 0.01, n_estimators = 1500).fit(e.iloc[:, :n_hvg].loc[train_indices].values, e.loc[e.index, "age"].loc[train_indices].astype(float).values)
            test_preds.update(zip(test_indices, model.predict(e.iloc[:, :n_hvg].loc[test_indices])))
            i += 1
            print(i)
        test_preds = pd.Series(test_preds)
        results = pd.DataFrame({"actual": e.loc[test_preds.index, "age"].astype(float), "predicted": test_preds})
        results = results.loc[results["actual"] > 0, :]
        results.plot.scatter("actual", "predicted", alpha = 0.6)
        results.to_csv("~/Desktop/rna_age.csv")
        plt.xlabel("Actual age (years)")
        plt.ylabel("Predicted age (years)")
        plt.show()
        load_results = False
        if load_results:
            results = pd.read_csv("~/Desktop/rna_age.csv").set_index("Unnamed: 0")
            results.index.name = "RegistrationCode"
        do = "disease"
        if do == "retina":
            r = RetinaScanLoader().get_data().df
            r = r.reset_index(["Date"], drop = True)
            r = r.loc[r.index.get_level_values(1) == "r_eye"].reset_index("eye", drop = True)
        else:
            r = make_diagnoses_df(results)
        r = r.loc[~r.index.duplicated(),:]
        rna_age_df = pd.DataFrame(results.diff(axis = 1).iloc[:, 1]).merge(r, left_index = True, right_index = True)
        np.abs(rna_age_df.corr().iloc[:, 0]).sort_values()


