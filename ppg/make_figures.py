import numpy as np
import pandas as pd
import torch
import seaborn as sns
from scipy.stats import mode
from remake_embs import saved_emb_dir, load_things, saved_model_dir, make_par_labels
import statsmodels.api as sm
import itertools
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
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
from datasets import ecg_dataset_pheno, tenk_tabular_dataset, sleep_dataset_pheno
from single_run_wandb import sweep_configuration as config
from LabData.DataLoaders.ECGTextLoader import ECGTextLoader
from all_to_ecg import load_split
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

def eval_clusters_medical_conditions(samples_df, ground_truth_df, k):
    kmeans = KMeans(n_clusters=k, random_state=0)
    clusters = kmeans.fit_predict(samples_df.apply(lambda x: (x - x.mean())/x.std()))
    metrics_results = []
    for label_name in ground_truth_df.columns:
        ground_truth = ground_truth_df[label_name]
        nmi = normalized_mutual_info_score(ground_truth, clusters)
        ami = adjusted_mutual_info_score(ground_truth, clusters)
        homogeneity = homogeneity_score(ground_truth, clusters)
        completeness = completeness_score(ground_truth, clusters)
        metrics_results.append({
            'Label': label_name,
            'NMI': nmi,
            'AMI': ami,
            'Homogeneity': homogeneity,
            'Completeness': completeness
        })
    metrics_df = pd.DataFrame(metrics_results)
    return metrics_df

def count_pars(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def prune_based_on_corr(df, thresh):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > thresh)]
    print(f"dropped {len(to_drop)} correlated columns")
    df = df.drop(to_drop, axis=1)
    return df

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

def do_format_PCA(df, n = 2):
    df_pca = pd.DataFrame(PCA(n).fit_transform(df))
    df_pca.index = df.index
    return df_pca

def make_age_df():
    recording_dates_df = ECGTextLoader().get_data(research_stage="baseline", study_ids=list(range(100)) + list(range(1000, 1011, 1))).df.reset_index().set_index(["RegistrationCode"])
    recording_dates_df = recording_dates_df.sort_values('Date', ascending = True)["Date"]
    recording_dates_df = recording_dates_df.loc[~recording_dates_df.index.duplicated(keep="first")]
    subs = SubjectLoader().get_data(study_ids=list(range(100)) + list(range(1000, 1011, 1))).df
    subs = subs.reset_index(["Date"], drop=True).dropna(subset=["yob", "month_of_birth"])
    subs_merged_raw = pd.DataFrame(recording_dates_df).merge(subs, left_index=True, right_index=True)
    age_computation = subs_merged_raw.loc[:, ["month_of_birth", "yob"]]
    age_computation.columns = ["month", "year"]
    age_computation["day"] = 15
    subs_merged_raw["birthdate"] = pd.to_datetime(age_computation)
    subs_merged_raw["Date"] = pd.to_datetime(subs_merged_raw["Date"])
    age_raw = list(map(lambda x: (x[1] - x[0]).days, zip(subs_merged_raw["birthdate"], subs_merged_raw["Date"])))
    age_df_raw = pd.DataFrame(age_raw)
    age_df_raw["RegistrationCode"] = list(subs_merged_raw.index)
    age_df_raw["Date"] = list(subs_merged_raw.Date)
    age_df_raw = age_df_raw.rename({0: "age"}, axis=1)
    age_df_raw = age_df_raw.set_index(["RegistrationCode", "Date"])
    ##for compatibility
    age_df_raw = age_df_raw.sort_values("Date")
    age_df_raw = age_df_raw.loc[~age_df_raw.index.get_level_values(0).duplicated(keep = "first"),:]
    age_df_raw = age_df_raw.reset_index(["Date"], drop = True)
    return age_df_raw

def make_labdata_dfs(tabular_datasets, index):
    ##make covariate, predictor, and baseline dfs
    comparator = pd.concat([ds.df for ds in tabular_datasets], axis = 1).fillna(0)
    comparator = comparator.drop([x for x in comparator.columns if x.startswith("Age_") or x.startswith("age_")], axis = 1) ##PRS loader age
    comparator.index.name = "RegistrationCode"
    comparator = comparator.loc[index,:]
    s = make_age_df()
    nightingale = pd.read_csv(
        "/net/mraid20/export/genie/LabData/Data/metabolomics/Nightingale Metabolomics/results_with_reg_code.csv").set_index(
        "RegistrationCode").iloc[:, 2:]  ##drop the sample id column and the redundant ingeger index column
    nightingale = nightingale.iloc[:, 0:250]  ##also drop some unnnecessary columns at the end
    nightingale = nightingale.loc[~nightingale.index.duplicated(keep="first"), :].drop(["Sample id"], axis = 1)
    ##set "TAG" to be a missing value, so it's dropped by default
    nightingale = nightingale.replace('TAG', np.nan)
    features_df = s.merge(nightingale, left_index = True, right_index = True)
    comparator = comparator.loc[:, ~comparator.columns.duplicated()]
    return features_df, comparator

def pred_diseases(df, disease_df, how):
    results = {}
    for disease in disease_df.columns:
        r, _ = cv(df, disease_df, task = "c", how = how, what = disease)
        results[disease] = roc_auc_score(r.iloc[:, 1], r.iloc[:, 0])
    results = pd.Series(results).sort_values(ascending = False)
    return results

def compare_two_disease_predictors(z_mech, z_blackbox, disease_df, how = "ols"):
    res = {}
    for label, df1 in z_mech.items():
        a = pred_diseases(df1, disease_df, how)
        if "time" in label:
            to_use = label.split("_time")[0] + "_black_box_" + "time"
        else:
            to_use = label + "_black_box"
        b = pred_diseases(z_blackbox[to_use], disease_df, how)
        a.name = "mech_time"
        b.name = "black_box_time"
        res[label] = pd.DataFrame(a).merge(b, left_index = True, right_index = True)
    return res


def make_panel_two(z_first_all_mech, comparator_all, all_mech, all_comparator, consider, how="ols"):
    z_mech_with_all = z_first_all_mech.copy()
    z_blackbox_with_all = comparator_all.copy()
    z_mech_with_all.update({"All": all_mech})
    z_blackbox_with_all.update({"All": all_comparator})
    fig, axes = plt.subplots(4, 2, figsize=(20, 20), sharex=True)
    for k, ax in zip(list(z_mech_with_all.keys()), axes.flat):
        make_total_plot(None, z_mech_with_all[k], z_blackbox_with_all[k.split("_time")[0]] if k != "All" else z_blackbox_with_all[k], features_df, consider, ax = ax, label = k.split("_time")[0], how = how)
    fig.suptitle("5 fold CV: Downstream prediction from embedings and learned prior")
    plt.show()

#assumes disease name is the first column
def make_panel_one(reses, all_concat_res):
    reses_ = reses.copy()
    reses_.update({"All": all_concat_res["All_time"]})
    fig, axes = plt.subplots(5, 2, figsize=(25, 20))
    for arr, ax in zip(reses_.items(), axes.flat):
        which, res_df = arr[0], arr[1]
        res_df_ = res_df.sort_values(ascending = False, by = "mech_time").iloc[0:20, :]
        res_df_ = res_df_.melt(ignore_index = False, var_name='Base Data', value_name='Value').reset_index()
        sns.barplot(res_df_, x="index", y="Value", hue="Base Data", ax = ax, palette = ["grey", "black"])
        ax.set_ylabel("Test AUC")
        ax.set_xlabel("")
        ax.set_ylim([0.5, 1])
        ax.set_title(f"{which}")
    fig.suptitle("Comparing mechanistic and black box embeddings")
    plt.show()

def make_total_plot(pca_240, all_concat_columns_ours, blackbox_embs, features_df, interest, ax = None, label = None, how = "ols"):
    res = {}
    if pca_240 is not None:
        for var in interest:
            res[var] = {"backbox_embs": cv(blackbox_embs, features_df, var, "r", how = how)[0].corr()["preds"]["true"],
                         "mech_embs": cv(all_concat_columns_ours, features_df, var, "r", how = how)[0].corr()["preds"]["true"],
                        "240_PCs": cv(pca_240, features_df, var, "r, how = how")[0].corr()["preds"]["true"]}
    else:
        common_subjects = list(set(list(blackbox_embs.index)).intersection(set(list(all_concat_columns_ours.index))))
        for var in interest:
            res[var] = {"backbox_embs": cv(blackbox_embs.loc[common_subjects, :], features_df, var, "r", how = how)[0].corr()["preds"]["true"],
                        "mech_embs": cv(all_concat_columns_ours.loc[common_subjects, :], features_df, var, "r", how = how)[0].corr()["preds"]["true"]}
    res = pd.DataFrame(res).T
    if ax is None:
        plt.figure(figsize=(10, 10))
        X_axis = np.arange(len(res.index))
        Y_axis = np.arange(0, 1.1, 0.1)
        plt.xticks(X_axis, list(map(label_mapper.get, res.index)))
        plt.yticks(Y_axis)
        if pca_240 is not None:
            plt.bar(x =X_axis - 0.4, height = res["240_PCs"], width = 0.4, color = "green", label = "Tabular training Data Data: PCs")
            plt.bar(x =X_axis - 0.8, height = res["backbox_embs"], width = 0.4, color = "black", label = "Blackbox Embeddings")
            plt.bar(x =X_axis, height = res["mech_embs"], width = 0.4, color = "blue" ,label = "Mechanistic Embeddings")
        else:
            plt.bar(x=X_axis - 0.4, height=res["backbox_embs"], width=0.4, color="black", label="Raw features")
            plt.bar(x=X_axis, height=res["mech_embs"], width=0.4, color="blue", label="Learned Embeddings")
        plt.legend()
        plt.xticks(rotation=45)
        plt.xlabel("Clinical feature")
        plt.ylabel("r")
        plt.title(f"Downstream Clinical Feature Prediction: {label}")
        plt.show()
    else:
        X_axis = np.arange(len(res.index))
        Y_axis = np.arange(0, 1.1, 0.1)
        ax.set_xticks(X_axis, list(map(label_mapper.get, res.index)), rotation=45)
        ax.set_yticks(Y_axis)
        ax.bar(x =X_axis - 0.2, height = res["backbox_embs"], width = 0.2, color = "blue", label = "Prior")
        ax.bar(x =X_axis, height = res["mech_embs"], width = 0.2, color = "black" ,label = "ECG")
        ax.legend()
        ax.set_ylabel("r")
        ax.set_title(label)
    return res

def stability_across(all_concat_cols):
    bases = list(set(list(map(lambda x: x.split("__")[0], all_concat_cols.columns))))
    corrmats =  dict(zip(bases, list(map(lambda y: all_concat_cols[list(filter(lambda x: y in x, all_concat_cols.columns))].corr(),bases ))))
    fig, axes = plt.subplots(10, 3, figsize=(30, 30), dpi = 150)
    for arr, ax in zip(corrmats.items(), axes.flat):
        sns.histplot(arr[1].iloc[np.triu_indices_from(arr[1],k=1)].values.flatten(),  ax  = ax, bins = 100)
        ax.set_title(arr[0])
        ax.set_xlabel("r across par")
    fig.suptitle("Stability of pars over different loaders")
    plt.show()
    return corrmats

def stability_within(all_concat_cols):
    bases = list(set(list(map(lambda x: x.split("__")[1], all_concat_cols.columns))))
    waves = ["_a", "_theta", "_b"]
    corrmats = {}
    for base in bases:
        for wave in waves:
            wave_base_cols = list(filter(lambda x: base in x and wave in x,  all_concat_cols))
            corrmats[(base, wave)] = all_concat_cols[wave_base_cols].corr()
    fig, axes = plt.subplots(6, 4, figsize=(30, 30), dpi = 150)
    for arr, ax in zip(corrmats.items(), axes.flat):
        sns.histplot(arr[1].iloc[np.triu_indices_from(arr[1],k=1)].values.flatten(),  ax  = ax, bins = 100)
        ax.set_title(arr[0][0] + arr[0][1])
        ax.set_xlabel("r across par")
    fig.suptitle("Within eaach loader, correlate amplitude/time/spread of all waves")
    plt.show()
    return corrmats


def make_all(z):
    all_concat = pd.concat(list(z.values()), axis = 0)
    all_concat_cols = pd.concat(list(map(lambda x : x[1].rename(dict(zip(x[1].columns, list(map(lambda z: z + "__" + str(x[0]), x[1].columns)))), axis = 1), z.items())), axis = 1)
    all_concat.index.name = "RegistrationCode"
    all_min = all_concat.groupby("RegistrationCode").min()
    all_min.columns = list(map(lambda x: x + "_min", all_min.columns))
    all_max = all_concat.groupby("RegistrationCode").max()
    all_max.columns = list(map(lambda x: x + "_max", all_max.columns))
    all = pd.concat([all_min, all_max], axis = 1)
    return all, all_concat_cols

def read_load_embs(saved_emb_dir,):
    z_first_all_integrated = {}
    z_first_all_solo = {}
    saved_embs = list(filter(lambda x: "integrated" in x, os.listdir(saved_emb_dir)))
    for emb in saved_embs:
        z_first_all_integrated.update({emb.split(".csv")[0]: pd.read_csv(f"{saved_emb_dir}{emb}").rename({"Unnamed: 0": "RegistrationCode"}, axis = 1).set_index("RegistrationCode")})
    saved_embs = list(filter(lambda x: "integrated" not in x, os.listdir(saved_emb_dir)))
    for emb in saved_embs:
        z_first_all_solo.update({emb.split(".csv")[0]: pd.read_csv(f"{saved_emb_dir}{emb}").rename(
            {"Unnamed: 0": "RegistrationCode"}, axis=1).set_index("RegistrationCode")})
    return z_first_all_solo, z_first_all_integrated

def load_all_splits():
    inds = []
    for i in range(1, 10):
        train_people, eval_people, test_people = load_split(i)
        inds += train_people
        inds += eval_people
        inds += test_people
    inds = list(set(inds))
    return inds

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    z_first_all_solo_ecg, z_first_all_integrated_ecg = read_load_embs(saved_emb_dir)
    tabular_datasets, tabular_domains, tabular_domain_labels, checkpoints, datasets_to_checkpoints = load_things(saved_model_dir, do_integrated = True)
    #all_blackbox, all_concat_blackbox_cols = make_all(z_first_all_blackbox)
    all_mech_ecg, all_concat_mech_ecg = make_all({k:v for k,v in z_first_all_integrated_ecg.items()})
    all_mech_features, all_concat_mech_features = make_all({k:v for k,v in z_first_all_integrated_ecg.items()})
    diseases = make_diseases_df(z_first_all_integrated_ecg["BodyMeasures_integrated"].index)
    features_df, comparator = make_labdata_dfs(tabular_datasets, all_mech_features.index)
    do_more = False
    labels = ['age', "Creatinine", 'Total_C', 'non_HDL_C', 'LDL_TG', 'HDL_TG', ]
    short_labels = ['age', "Creatinine", "waist(cm)", 'Total_C', 'non_HDL_C', 'LDL_TG', 'HDL_TG']
    label_mapper = dict(zip(labels, short_labels))

    print(cv(z_first_all_integrated_ecg["BodyMeasures_integrated_time"], features_df, what="age", how="ols", task="r")[
        0].corr())

    do_validation_figure_clincal = False
    if do_validation_figure_clincal:
        ecgtext = ECGTextLoader().get_data(research_stage="baseline",
                                           study_ids=list(range(100)) + list(range(1000, 1011, 1))).df
        ecgtext = ecgtext.fillna(0).select_dtypes(exclude=object).reset_index().set_index("RegistrationCode").drop(
            ["Date"], axis=1)
        most_relevant_ecg_features = [
            "pr_ms",  # PR Interval
            "qrs_ms",  # QRS Duration
            "qtc_ms",  # Corrected QT Interval
            "t_axis"  # T-wave Axis
        ]

        fig, axes = plt.subplots(6, 1, figsize=(7, 20))
        axes = axes.flatten()

        # List of keys to plot, all integrated except last one (Non-Integrated DEXA)
        integrated_keys = list(z_first_all_integrated_ecg.keys())

        # Select the 'Non-Integrated DEXA' key
        not_integrated_key = 'DEXA_time'  # Assuming this key structure

        for idx, k in enumerate(integrated_keys):
            if idx >= 5:
                break

            # Merge integrated embeddings
            ecgtext_embs_merged = ecgtext.merge(z_first_all_integrated_ecg[k][['T_iminus_b', 'R2_sd', 'P_a', 'Q_a', 'R_a']], left_index=True, right_index=True)
            corrmat_integrated = ecgtext_embs_merged.corr()
            of_interest_integrated = corrmat_integrated.iloc[-5:, :-5]
            just_interesting_integrated = of_interest_integrated[most_relevant_ecg_features]

            sns.heatmap(
                just_interesting_integrated,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                cbar_kws={'shrink': 0.7, 'label': 'Correlation Coefficient'},
                linewidths=0.5,
                annot_kws={"size": 14},
                xticklabels=True,
                yticklabels=True,
                ax=axes[idx]
            )
            axes[idx].set_title(f"Integrated: {k.split('_')[0]}", fontsize=18)
            axes[idx].tick_params(axis='x', labelrotation=45, labelsize=14)
            axes[idx].tick_params(axis='y', labelsize=12,  labelrotation=45)

        # Non-Integrated DEXA Panel
        ecgtext_embs_not_integrated_merged = ecgtext.merge(z_first_all_solo_ecg[not_integrated_key][['T_iminus_b', 'R2_sd', 'P_a', 'Q_a', 'R_a']],
                                                           left_index=True, right_index=True)
        corrmat_not_integrated = ecgtext_embs_not_integrated_merged.corr()
        of_interest_not_integrated = corrmat_not_integrated.iloc[-5:, :-5]
        just_interesting_not_integrated = of_interest_not_integrated[most_relevant_ecg_features]

        sns.heatmap(
            just_interesting_not_integrated,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={'shrink': 0.7, 'label': 'Correlation Coefficient'},
            linewidths=0.5,
            annot_kws={"size": 14},
            xticklabels=True,
            yticklabels=True,
            ax=axes[5]  # Bottom right panel
        )
        axes[5].set_title("Non-Integrated", fontsize=18)
        axes[5].tick_params(axis='x', labelrotation=45, labelsize=14)
        axes[5].tick_params(axis='y', labelsize=12)

        # Turn off any unused subplot
        for i in range(6, len(axes)):
            fig.delaxes(axes[i])

        # Adjust layout and add a super title
        plt.tight_layout()
        plt.suptitle("CardioPRIME Validation: Learned Feature Validation", fontsize=24, y=1.02)
        plt.show()


    do_validation_figure_voltage = False
    if do_validation_figure_voltage:
        fig, axes = plt.subplots(6, 1, figsize=(7, 20))
        axes = axes.flatten()

        # List of keys to plot, all integrated except last one (Non-Integrated DEXA)
        integrated_keys = list(z_first_all_integrated_ecg.keys())

        # Select the 'Non-Integrated DEXA' key
        not_integrated_key = 'DEXA_time'  # Assuming this key structure

        for idx, k in enumerate(integrated_keys):
            if idx >= 5:
                break

            # Merge integrated embeddings
            ecgtext_embs_merged = ecgtext.merge(z_first_all_integrated_ecg[k], left_index=True, right_index=True)
            corrmat_integrated = ecgtext_embs_merged.corr()
            of_interest_integrated = corrmat_integrated.iloc[-25:, :-25]
            best_cols = list(of_interest_integrated.abs().max().sort_values(ascending = False).iloc[0:5].index.values)
            best_rows = list(of_interest_integrated.abs().max(axis = 1).sort_values(ascending = False).iloc[0:5].index.values)
            just_interesting_integrated = of_interest_integrated.loc[best_rows, best_cols]

            sns.heatmap(
                just_interesting_integrated,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                cbar_kws={'shrink': 0.7, 'label': 'Correlation Coefficient'},
                linewidths=0.5,
                annot_kws={"size": 14},
                xticklabels=True,
                yticklabels=True,
                ax=axes[idx]
            )
            axes[idx].set_title(f"Integrated: {k.split('_')[0]}", fontsize=18)
            axes[idx].tick_params(axis='x', labelrotation=45, labelsize=14)
            axes[idx].tick_params(axis='y', labelsize=12,  labelrotation=45)

        # Non-Integrated DEXA Panel
        ecgtext_embs_not_integrated_merged = ecgtext.merge(z_first_all_solo_ecg[not_integrated_key],
                                                           left_index=True, right_index=True)
        corrmat_not_integrated = ecgtext_embs_not_integrated_merged.corr()
        of_interest_not_integrated = corrmat_not_integrated.iloc[-25:, :-25]
        best_cols = list(of_interest_not_integrated.abs().max().sort_values(ascending=False).iloc[0:5].index.values)
        best_rows = list(of_interest_not_integrated.abs().max(axis=1).sort_values(ascending=False).iloc[0:5].index.values)
        just_interesting_not_integrated = of_interest_not_integrated.loc[best_rows, best_cols]

        sns.heatmap(
            just_interesting_not_integrated,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={'shrink': 0.7, 'label': 'Correlation Coefficient'},
            linewidths=0.5,
            annot_kws={"size": 14},
            xticklabels=True,
            yticklabels=True,
            ax=axes[5]  # Bottom right panel
        )
        axes[5].set_title("Non-Integrated", fontsize=18)
        axes[5].tick_params(axis='x', labelrotation=45, labelsize=14)
        axes[5].tick_params(axis='y', labelsize=12)

        # Turn off any unused subplot
        for i in range(6, len(axes)):
            fig.delaxes(axes[i])

        # Adjust layout and add a super title
        plt.tight_layout()
        plt.suptitle("CardioPRIME Validation: Learned Feature Validation", fontsize=24, y=1.02)
        plt.show()

    disease_plot = False
    if disease_plot:
        k_means = 6
        # Create a figure with subplots
        num_panels = len(z_first_all_integrated_ecg.items())
        cols = 1  # Set the number of columns to 1 for a tall figure
        rows = num_panels  # One row per k, v pair
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))  # Adjust figure size to be tall

        # Flatten axes for easier indexing
        axes = axes.flatten()

        icd_mapping = {
            "5C80": "Primary hypercholesterolaemia",
            "DB60": "Haemorrhoids unspecified",
            "ME84": "Spinal pain",
            "BA00": "Essential hypertension",
            "5A40": "Intermediate hyperglycaemia",
            "CA08.0": "Allergic rhinitis",
            "DA42.1": "Helicobacter gastritis",
            "ND56.2": "Fracture unspecified",
            "GC08.Z": "Urinary Tract Infection",
            "ED80": "Acne",
            "8A80": "Migraine",
            "5B81": "Obesity",
            "GA31": "Female Infertility",
        }

        # Iterate over each k, v pair and create a heatmap
        for idx, (k, v) in enumerate(z_first_all_integrated_ecg.items()):
            diseases_here = make_diseases_df(list(v.index.values))
            diseases_here = diseases_here[diseases_here.sum().sort_values(ascending = False).iloc[0:10].index]
            not_integrated_k = k.split("_")[0] + "_" + k.split("_")[-1]

            # Compute new evaluation for integrated embeddings
            eval_clusters_integrated = eval_clusters_medical_conditions(v, diseases_here.loc[v.index], k =k_means).set_index("Label").clip(1e-8)

            # Compute new evaluation for non-integrated embeddings
            matched = z_first_all_solo_ecg[not_integrated_k]
            eval_clusters_not_integrated = eval_clusters_medical_conditions(matched,
                                                                            diseases_here.loc[matched.index], k =k_means).set_index("Label").clip(1e-8)

            diff = eval_clusters_integrated/eval_clusters_not_integrated
            diff.index =  list(map(icd_mapping.get, diff.index))
            diff[diff > 1e2] = None
            diffmean = pd.DataFrame(diff.mean(axis = 0, )).T
            diffmean.index = ["Mean"]
            diff = pd.concat([diff, diffmean], axis = 0)

            # Create the heatmap for integrated embeddings

            # Create a mask for NaN values
            mask = diff.isna()  # Mask cells where `diff` is NaN

            # Create the heatmap

            # Create the heatmap
            heatmap = sns.heatmap(
                diff.fillna(0),  # Replace NaNs with 0 for visualization
                annot=diff.applymap(lambda x: "*" if pd.isna(x) else f"{x:.5f}"),  # Add annotations
                fmt="",  # Prevent formatting issues with mixed types
                cmap="RdYlGn",  # Diverging colormap (red-yellow-green)
                cbar_kws={'shrink': 0.7, 'label': 'Evaluation Integrated/Original'},
                linewidths=0.5,  # Add lines between cells
                annot_kws={"size": 10},  # Annotation font size
                xticklabels=True,
                yticklabels=True,
                ax=axes[idx],  # Specify the current axis for the heatmap
                vmin=0,  # Minimum value for the colormap
                vmax=2,  # Maximum value for the colormap
                center=1,  # Center the colormap at 1
            )

            # Add a darker blue overlay for NaN cells
            for (i, j), val in np.ndenumerate(diff.values):
                if pd.isna(val):  # Check if the value is NaN
                    axes[idx].add_patch(plt.Rectangle((j, i), 1, 1, color=(0.2, 0.4, 0.8, 0.9), ec=None))  # Darker blue

            # Redraw the plot to apply changes
            plt.draw()
            axes[idx].set_title(f"Integrated/Original: {k.split('_')[0] if k.split('_')[0] != 'iglu' else 'CGM iglu features'}", fontsize=14)
            axes[idx].tick_params(axis='x', labelrotation=45, labelsize=10)
            axes[idx].tick_params(axis='y', labelsize=10)
        # Turn off unused subplots
        for i in range(len(z_first_all_integrated_ecg), len(axes)):
            fig.delaxes(axes[i])

        # Adjust layout
        plt.tight_layout()
        plt.suptitle("Diseases", fontsize=18, y=1.02)
        ##for the asterix blue to work
        plt.draw()
        plt.show()