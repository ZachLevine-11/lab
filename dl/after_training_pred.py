import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#from sklearn.metrics import roc_auc_score
from general_pred import do_CV, predict_binary_phenotype
from wandb_main import make_age_df
import seaborn as sns
import umap
from sklearn.decomposition import PCA
from LabData.DataLoaders.SubjectLoader import SubjectLoader
from LabData.DataLoaders.ECGTextLoader import ECGTextLoader
from ECGDataset import ecg_dataset_pheno
from sklearn.metrics import roc_auc_score
from sklearn.impute import KNNImputer
import torch
from scipy.stats import pearsonr
from GeneticsPipeline.helpers_genetic import read_status_table
from ecopy import Mantel
from sklearn.decomposition import PCA
import umap
import statsmodels as sm

def pred_age(df, ageSeries):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alphas = [0]
    weights = [0,]
    rs = {}
    test_folds = np.array_split(list(df.index.get_level_values(0)), 5)
    for alpha in alphas:
        for L1_wt in weights:
            res = do_CV(ageSeries, "age", list(df.index.get_level_values(0)), list(df.index.get_level_values(0)), test_folds, df, device, alpha, L1_wt, which = "age")
            r, P = pearsonr(res.iloc[:, 0].values, res.iloc[:, 1].values)
            rs[(alpha, L1_wt)] = {"r": r, "P": P, "N": len(res)}
    return pd.DataFrame(rs).T

def pred_gender(df, genderSeries):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alphas = [1]
    weights = [0,]
    rs = {}
    test_folds = np.array_split(list(df.index.get_level_values(0)), 5)
    for alpha in alphas:
        for L1_wt in weights:
            res = do_CV(genderSeries, "has", list(df.index.get_level_values(0)), list(df.index.get_level_values(0)), test_folds, df, device, alpha, L1_wt, which = "DiseaseDiagnoses")
            auc = roc_auc_score(res.iloc[:, 0].values, res.iloc[:, 1].values)
            rs[(alpha, L1_wt)] = {"auc": auc, "N": len(res)}
    return pd.DataFrame(rs).T

def make_distance_between(embeddings_one, embeddings_two, fname = "dist_all_unmatched_first_first_have_second_only", save = True):
    i = 0
    total_len = len(embeddings_one)*len(embeddings_two)
    res = {}
    for id_one in embeddings_one.index:
        for id_two in embeddings_two.index:
            res[(id_one, id_two)] = torch.nn.CosineSimilarity()(torch.Tensor(embeddings_one.loc[id_one, :].to_numpy()).reshape(1, -1), torch.Tensor(embeddings_two.loc[id_two, :].to_numpy()).reshape(1, -1)).item()
            i += 1
            if i % 100000 == 1:
                print(100*i/total_len)
    res = pd.Series(res)
    if save:
        res.to_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/dist_all/" + fname + ".csv")
        print("finished")
    else:
        return res

def plot_ecg(signal, single_panel = True, fname = None, colours = None):
    if single_panel:
        fig, axes = plt.subplots(12,1, figsize = (20, 20))
        channel = 0
        for row in axes:
            if colours is None:
                row.plot(list(range(len(signal.iloc[channel, 0:len(signal.columns)]))), signal.iloc[channel, 0:len(signal.columns)])
            else:
                row.scatter(list(range(len(signal.iloc[channel, 0:len(signal.columns)]))), signal.iloc[channel, 0:len(signal.columns)], c = colours.flatten(), cmap = "cool", alpha = 0.3)
            channel += 1
        plt.savefig(fname)
    else:
        for channel in range(12):
            plt.plot(list(range(10000)), signal.iloc[channel, 0:len(signal.columns)])
        plt.savefig(fname)


##from pandas plink: https://github.com/limix/pandas-plink/blob/main/pandas_plink/_read_rel.py
def _1d_to_2d(values, n):
    from numpy import tril, tril_indices_from, zeros
    K = zeros((n, n))
    K[tril_indices_from(K)] = values
    K = K + tril(K, -1).T
    return K

def read_rel_file_internal(filepath):
    from numpy import float64
    rows = []
    with open(filepath, "r") as f:
        for row in f:
            rows += [float64(v) for v in row.strip().split("\t")]
    return rows

def match_rel_files():
    status_table_all = read_status_table()
    status_table_all = status_table_all.sort_values(by=["date", "version"], ascending=True)
    try:
        status_table_qc = status_table_all[status_table_all.passed_qc].copy()
    except ValueError:
        status_table_qc = status_table_all.dropna()
        status_table_qc = status_table_qc[status_table_qc.passed_qc].copy()
    ids_rel = list(pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/relatedness/cov.rel.id", sep = "\t", header = None, comment = "#")[1].values)
    ids_rel.remove("Dummy")
    ids_rel_tenk = list(map(status_table_qc.set_index("gencove_id").RegistrationCode.to_dict().get, ids_rel))
    rel = read_rel_file_internal("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/relatedness/cov.rel")
    rel =_1d_to_2d(rel, 9454)
    rel = pd.DataFrame(rel)
    rel = rel.iloc[1:, 1:] ##drop dummy row/col
    rel.index = ids_rel_tenk
    rel.columns = ids_rel_tenk
    rel = rel.loc[~rel.index.isna(), ~rel.columns.isna()] ##drop columns with no matching tenk id (i.e repeated runs from the different APIs that didn't pass gencove QC)
    df_emb = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/dist_all/dist_all_unmatched_first_first_have_second_only.csv")
    df_emb = df_emb.set_index(["Unnamed: 0", "Unnamed: 1"])
    ##intersect the two indexes of the dfs
    rel = rel.loc[list(set(df_emb.index.get_level_values(1)).intersection(rel.index)), list(set(df_emb.index.get_level_values(1)).intersection(rel.index))]
    df_emb = df_emb.loc[(rel.index, rel.columns),:]
    df_emb = df_emb.unstack()
    df_emb.index.name = None
    df_emb.columns.name = None
    df_emb.columns = df_emb.columns.get_level_values(1)
    m = Mantel(np.abs(df_emb).values, np.abs(rel).values, nperm=999, tail = "lower")
    return rel, df_emb, m


if __name__ == "__main__":
    splits_path = "/net/mraid20/export/jasmine/zach/cross_modal/splits/"
    files = list(pd.read_csv(splits_path + "files.csv")["0"])
    ds = ecg_dataset_pheno(files=files)
    do_disease_diff = False
    if do_disease_diff:
        subs_10k = SubjectLoader().get_data(study_ids=["10K"]).df
        subs_cvd = SubjectLoader().get_data(study_ids=[1001]).df
        ##embeddings of #all people
        embeddings_one_all = pd.read_csv(
            "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/no_ft/one_no_linear.csv").set_index(
            ["RegistrationCode", "Date"])
        ##grab the embeddings of just healthy and just CVD cohort
        embeddings_one_10k = embeddings_one_all.loc[embeddings_one_all.index.get_level_values(0).intersection(
            list(set(subs_10k.index.get_level_values(0)))), :]
        embeddings_one_cvd = embeddings_one_all.loc[embeddings_one_all.index.get_level_values(0).intersection(
            list(set(subs_cvd.index.get_level_values(0)))), :]
        embeddings_one_10k["CVD"] = 0
        embeddings_one_cvd["CVD"] = 1
        ##match the two cohorts through random downsampling
        embeddings_one_10k = embeddings_one_10k.loc[
                             np.random.choice(embeddings_one_10k.index.get_level_values(0), size=11000, replace=False), :]
        e_all = pd.concat([embeddings_one_cvd, embeddings_one_10k])
        train_people = list(np.random.choice(e_all.index.get_level_values(0), 10000,
                                             replace=False))  ##need at least emb dim samples in train for reg to work
        test_people = list(
            np.random.choice(list(set(e_all.index.get_level_values(0)) - set(train_people)), 1000, replace=False))
        a = predict_binary_phenotype(e_all["CVD"], e_all.iloc[:, :-1], "CVD", "embeddings", "cpu", "linear",
                                     train_people, test_people, [])
        roc_auc_score(a[1], a[0])
    do_PCA = False
    if do_PCA:
        subs = SubjectLoader().get_data(study_ids=list(range(100)) + list(range(1000, 1011, 1))).df
        embeddings_one = pd.read_csv(
            "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/no_ft/one_no_linear.csv").drop(
            ["Date"], axis=1).set_index("RegistrationCode")
        merged = embeddings_one.merge(subs.reset_index().set_index("RegistrationCode"), left_index=True, right_index=True)
        merged = merged.loc[:, list(embeddings_one.columns) + ["StudyTypeID"]]
        res = pd.DataFrame(umap.UMAP(metric="cosine", n_neighbors=4, min_dist=0, spread=1).fit_transform(
            PCA(64).fit_transform(merged.iloc[:, :-1])))
        sns.scatterplot(x=res.iloc[:, 0], y=res.iloc[:, 1], hue=list(map(str, merged["StudyTypeID"])), alpha=0.7)
        plt.show()
        embeddings_one_date = pd.read_csv(
            "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/no_ft/one_no_linear.csv").set_index(
            ["RegistrationCode", "Date"])
        age_df = make_age_df(embeddings_one_date, ds)
        merged_age = embeddings_one_date.merge(age_df.reset_index(["Date"], drop=True), left_index=True,
                                               right_index=True).reset_index(["Date"], drop=True)
        sns.scatterplot(x=res.iloc[:, 0], y=res.iloc[:, 1], c=merged_age["age"] // 365, alpha=0.7)
        plt.show()
    make_ecgtext_from_scratch = False
    if make_ecgtext_from_scratch:
        ecgtext = ECGTextLoader().get_data(study_ids=ds.id_list).df
        ecgtext = ecgtext.loc[~ecgtext.index.get_level_values(0).duplicated(keep="first"), :]
        ecgtext = ecgtext.drop(["st_t", "non_confirmed_diagnosis", "qrs", "conclusion"], axis=1)
        for var in ecgtext.columns:
            try:
                ecgtext[var] = ecgtext[var].astype(np.float64)
            except ValueError:
                ecgtext.drop(var, axis=1, inplace=True)
        backup_id = ecgtext.index.get_level_values(0).copy()
        backup_date = ecgtext.index.get_level_values(1).copy()
        imputer_X = KNNImputer().fit(ecgtext)
        ecgtext = imputer_X.transform(ecgtext)
        ecgtext = pd.DataFrame(ecgtext)
        ecgtext["RegistrationCode"] = backup_id
        ecgtext["Date"] = backup_date
        ecgtext.set_index(["RegistrationCode", "Date"], inplace=True)
    load_imputed_ecgtext = True
    if load_imputed_ecgtext:
        ##date is not formatted properly here
        ecgtext = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/ecgtext_imputed/ecgtext_all_subjects.csv").set_index(["RegistrationCode", "Date"])
    do_age_pred = False
    if do_age_pred:
        embeddings_one = pd.read_csv(
            "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/no_ft/one_no_linear.csv").set_index(
            ["RegistrationCode", "Date"])
        age_df = make_age_df(embeddings_one, ds)
        ##age_df has the date encoded as a differently typed object than emb_df, and so when you do the merge the rows will be empty
        ##we drop date from the merged df, but emb_df is on the left and always has it so not an issue
        ##just drop it here
        preds_dl = pred_age(embeddings_one, age_df.reset_index(["Date"], drop=True))
        preds_hrv = pred_age(ecgtext, age_df.reset_index(["Date"], drop=True))
    try_gender = False
    if try_gender:
        embeddings_one = pd.read_csv(
            "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/no_ft/one_no_linear.csv").set_index(
            ["RegistrationCode", "Date"])
        subs = SubjectLoader().get_data(study_ids=ds.id_list).df
        subs = subs.reset_index(["Date"], True).dropna(subset=["yob", "month_of_birth"])
        genderSeries = pd.DataFrame(subs["gender"]).dropna().astype(float)
        genderSeries.columns = ["has"]
        gender_pred_dl = pred_gender(embeddings_one, genderSeries)
        gender_pred_hrv = pred_gender(ecgtext, genderSeries)
    do_dist = False
    if do_dist:
        embeddings_one = pd.read_csv(
            "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/no_ft/one_no_linear.csv").drop(
            ["Date"], axis=1).set_index("RegistrationCode")
        embeddings_two = pd.read_csv(
            "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/no_ft/two_no_linear.csv").drop(
            ["Date"], axis=1).set_index("RegistrationCode")
        one_only_repeat = embeddings_one.loc[embeddings_two.index.values, :]
        distance_matched = pd.Series({id: torch.nn.CosineSimilarity()(
            torch.Tensor(one_only_repeat.loc[id, :].to_numpy()).reshape(1, -1),
            torch.Tensor(embeddings_two.loc[id, :].to_numpy()).reshape(1, -1)).item() for id in one_only_repeat.index})
        distance_matched.index.name = "RegistrationCode"
        distance_matched = pd.DataFrame(distance_matched).sort_values(0)
        distance_matched["rank"] = list(range(len(distance_matched)))
        distance_matched.to_csv("~/Desktop/distance_matched.csv")
        ##for gwas
    #   df = distance_matched.iloc[:, 0:2]
    #   df = df.loc[df["0"] > 0, :]
    #   df.to_csv( "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/no_ft/distance_only_positive.csv")
       ###
        distance_all = make_distance_between(one_only_repeat, embeddings_two, save = False)
        distance_all.index.name = "RegistrationCodes"
        distance_all = pd.DataFrame(distance_all).sort_values(0)
        distance_all["rank"] = list(range(len(distance_all)))
        distance_all.to_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/dist_all_unmatched_first_second.csv")
        baselines = {"negative_3_smallest": list(
            map(lambda combined: ds.__getitem__(combined[1], combined[0][0]).numpy(), list(
                zip(list(map(lambda x: ds.get_date(x), distance_matched.iloc[0:3, ].index)),
                    distance_matched.iloc[0:3, ].index)))),
                     "positive_3_smallest": list(
                         map(lambda combined: ds.__getitem__(combined[1], combined[0][0]).numpy(), list(zip(list(
                             map(lambda x: ds.get_date(x),
                                 distance_matched.loc[distance_matched[0] > 0, :].iloc[0:3, ].index)),
                                                                                                            distance_matched.loc[
                                                                                                            distance_matched[
                                                                                                                0] > 0,
                                                                                                            :].iloc[
                                                                                                            0:3, ].index)))),
                     "positive_3_biggest": list(
                         map(lambda combined: ds.__getitem__(combined[1], combined[0][0]).numpy(), list(
                             zip(list(map(lambda x: ds.get_date(x), distance_matched.iloc[-3:, ].index)),
                                 distance_matched.iloc[-3:, ].index))))}
        follow_ups = {"negative_3_smallest": list(
            map(lambda combined: ds.__getitem__(combined[1], combined[0][1]).numpy(), list(
                zip(list(map(lambda x: ds.get_date(x), distance_matched.iloc[0:3, ].index)),
                    distance_matched.iloc[0:3, ].index)))),
                      "positive_3_smallest": list(
                          map(lambda combined: ds.__getitem__(combined[1], combined[0][1]).numpy(), list(zip(list(
                              map(lambda x: ds.get_date(x),
                                  distance_matched.loc[distance_matched[0] > 0, :].iloc[0:3, ].index)),
                                                                                                             distance_matched.loc[
                                                                                                             distance_matched[
                                                                                                                 0] > 0,
                                                                                                             :].iloc[
                                                                                                             0:3, ].index)))),
                      "positive_3_biggest": list(
                          map(lambda combined: ds.__getitem__(combined[1], combined[0][1]).numpy(), list(
                              zip(list(map(lambda x: ds.get_date(x), distance_matched.iloc[-3:, ].index)),
                                  distance_matched.iloc[-3:, ].index))))}

        for key in baselines.keys():
            for i in range(len(baselines[key])):
                baseline_person_ecg = pd.DataFrame(baselines[key][i])
                follow_up_person_ecg = pd.DataFrame(follow_ups[key][i])
                plot_ecg(baseline_person_ecg, True,
                         fname="/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/selected_patients/" + key + "/" + str(
                             i) + "_baseline.png")
                plot_ecg(follow_up_person_ecg, True,
                         fname="/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/selected_patients/" + key + "/" + str(
                             i) + "_follow_up.png")
        plt.hist(x=distance_matched.iloc[:, 0], bins=25, alpha=0.2, color="green", density=True, label="Matched Visits")
        plt.hist(x=distance_all.iloc[:, 2], bins=25, alpha=0.2, color="blue", density=True, label="All Recordings")
        plt.legend()
        plt.title("Distance between first and second appointments")
        plt.show()
    big_generation = False
    if big_generation:
        embeddings_one = pd.read_csv(
            "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/no_ft/one_no_linear.csv").drop(
            ["Date"], axis=1).set_index("RegistrationCode")
        embeddings_two = pd.read_csv(
            "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/no_ft/two_no_linear.csv").drop(
            ["Date"], axis=1).set_index("RegistrationCode")
        one_only_repeat = embeddings_one.loc[embeddings_two.index.values, :]
        res = make_distance_between(one_only_repeat, one_only_repeat, save = True)
