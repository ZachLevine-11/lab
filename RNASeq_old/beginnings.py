import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from LabData.DataLoaders.SubjectLoader import SubjectLoader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
from LabData.DataLoaders.BloodTestsLoader import BloodTestsLoader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
import datetime as dt
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
import datetime as dt
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from matplotlib import cm

def rack_extractor(rack_string):
    try:
        rack_string = str(rack_string)
        if rack_string == "Rack_":
            return [0]
        elif "-" in rack_string:
            return list(range(int(rack_string.split(" ")[-1].split("B")[0].split("-")[0]), int(rack_string.split(" ")[-1].split("B")[0].split("-")[1]) + 1))
        else:
            return [int(rack_string.split(" ")[-1].split("B")[0])]
    except (TypeError, ValueError) as e:
        return [0]

def correct_counts(counts_df, metadata_df_used, runs_summary, drop_duplicates=True):
    metadata_df_used["Run_Number"] = -1
    for i in range(len(metadata_df_used["Sample Description"])):
        try:
            metadata_df_used["Run_Number"].iloc[i] = int(str(metadata_df_used["Sample Description"].iloc[i]).split("Rack")[-1].split("_")[0])
        except ValueError:
            pass
    norm_counts = metadata_df_used.merge(right=counts_df, left_on= metadata_df.columns[0], right_on="SampleName", how="inner")
    norm_counts = norm_counts.set_index("participant_id")
    if drop_duplicates:
        norm_counts = norm_counts.loc[~norm_counts.index.duplicated(keep="last"), :]
    backup_total_counts = norm_counts["corrected_count_sum"]
    backup_run_number = norm_counts["Run_Number"]
    metacols = list(metadata_df_used.columns)
    metacols.remove("participant_id") ##setting the index removes this column so we can't drop it, do this to avoid the error in the drop column
    norm_counts = norm_counts.drop(metacols, axis = 1) ##keep only numeric gene columns
    norm_counts = norm_counts.loc[:, list(norm_counts.dtypes[norm_counts.dtypes == "int64"].index)]
    counts = norm_counts.divide(backup_total_counts, axis="rows")
    counts["Methodology"] = "Unknown"
    for id in backup_run_number.index:
        run_number = backup_run_number.loc[id]
        for i in range(len(runs_summary["Plates_Included_Numeric"])):
            if run_number in runs_summary["Plates_Included_Numeric"].iloc[i]:
                methodology = runs_summary.iloc[i,:]["Library Prep. Metadology "]
                counts.loc[id, "Methodology"] = methodology
    return counts, backup_total_counts  ##corrects for how many reads each sample started with

def compare_same_person_subsampled(thresh, counts_with_replicates_df, id):
    individual_df = counts_with_replicates_df.loc[counts_with_replicates_df.index == id, :]
    inds = individual_df.index
    for i in list(range(len(inds))):
        for j in list(filter(lambda x: x != i, list(range(len(inds))))):
            s1 = np.log10(individual_df.iloc[i, :].clip(thresh))
            s2 = np.log10(individual_df.iloc[j, :].clip(thresh))
        print(np.corrcoef(s1, s2))
        plt.scatter(s1, s2)
        plt.show()

def compare_different_people(threshs, counts_no_replicates_df, n):
    inds = np.random.choice(counts_no_replicates_df.index, n)
    for thresh in threshs:
        for i in inds:
            for j in list(filter(lambda x: x != i, inds)):
                s1 = np.log10(counts_no_replicates_df.loc[i, :].clip(thresh))
                s2 = np.log10(counts_no_replicates_df.loc[j, :].clip(thresh))
                s1_thresh = np.log10(counts_no_replicates_df.loc[i, counts_no_replicates_df.loc[i, :] >= thresh])
                s2_thresh = np.log10(counts_no_replicates_df.loc[j, counts_no_replicates_df.loc[j, :] >= thresh])
                print("s1 has ", len(s1_thresh.index)/len(s1.index), " genes above lower clip threshold")
                print("s2 has ", len(s2_thresh.index)/len(s2.index), " genes above genes above lower clip threshold")
                maxsize = min(len(s1_thresh.index), len(s2_thresh.index)) ##match size through random sampling
                s1_thresh_sampled = s1_thresh.sort_values(ascending = False).iloc[0:maxsize]
                s2_thresh_sampled = s2_thresh.sort_values(ascending = False).iloc[0:maxsize]
                print("clip corr: " + str(np.corrcoef(s1, s2)))
                plt.scatter(s1, s2)
                plt.title("clip " + str(thresh))
                plt.show()
                print("thresh corr: "+ str(np.corrcoef(s1_thresh_sampled, s2_thresh_sampled)))
                plt.scatter(s1_thresh_sampled, s2_thresh_sampled)
                plt.title("thresh " + str(thresh))
                plt.show()

def clip_and_log(df, thresh):
    df = df.clip(thresh)
    df = np.log10(df)
    return df

##get reduced dimensionality data (i.e from PCA or tSNE) and add the index to it, merging afterwards with loaderdata and sample metadata
def make_regcode_and_merge(log_values_proj, orig_notlogged_df, loaderdata):
    log_values_proj = pd.DataFrame(log_values_proj)
    log_values_proj["RegistrationCode"] = orig_notlogged_df.index
    log_values_proj = log_values_proj.set_index("RegistrationCode")
    merged = log_values_proj
    merged["Methodology"] = orig_notlogged_df["Methodology"]
    merged = loaderdata.merge(how = "inner", left_index = True, right_index = True, right = merged)
    merged = merged.merge(how = "inner", left_index = True, right = metadata_df, right_on = "participant_id")
    merged = merged.loc[~merged.index.duplicated(keep = "last"),:]
    return merged

def do_PCA(df, thresh, loaderdata, pheno, do_full = False): ##expects the batch number as the last column in df
    logvalues = clip_and_log(df.iloc[:, :-1], thresh)
    if do_full:
        pca_all = PCA(n_components=10)
        pca_all.fit(logvalues)
        plt.plot(list(range(len(pca_all.explained_variance_ratio_[0:10]))), pca_all.explained_variance_ratio_[0:10])
        plt.xlabel("PC [1-10]")
        plt.ylabel("%EXP Var")
        plt.show()
    ##count umis for each gene, if gene A shows up 100 times with 5 umis, that means I only started with 5 A gene reads, and the value sholud be 5
    pca_reduced = PCA(n_components=2)
    log_values_proj = pca_reduced.fit_transform(X = logvalues)
    merged = make_regcode_and_merge(log_values_proj = log_values_proj, orig_notlogged_df = df, loaderdata = loaderdata)
    ##confounders [batch, gender, bmi, date, age, (each prep is for 384 samples), the sequencing run divides it into 192 twice, two runs per preparation batch, ex
    if pheno == "Methodology":
        plt.scatter(x = merged[0], y = merged[1], c = pd.factorize(merged[pheno])[0])
    else:
        plt.scatter(x = merged[0], y = merged[1], c = merged[pheno])
    plt.show()
    return merged

def predict_pheno(counts_df, loaderdata, thresh, pheno = "sample_date_numerical"):
    df = do_PCA(counts_df, thresh, loaderdata, pheno)
    reg = LinearRegression().fit(df[[0,1]], df[pheno])
    y_pred = reg.predict(df[[0,1]])
    var_exp = explained_variance_score(df[pheno], y_pred)
    return var_exp


def prepare_f(df):
    df["gene_name"] = list(map(lambda x: themap.get(x.split(".")[0]), df.gene_id))  ##drop the version code after the period, as the gene is uniquelys encoded by the part before it
    return pd.DataFrame(df.groupby("gene_name").mean().loc[:, "TPM"]).T


def pick_thresh(df, threshs = [1e-4, 1e-5], iters = 200):
    theirs_theirs = {}
    our_ours = {}
    our_theirs = {}
    for thresh in threshs:
        merged_all = clip_and_log(df, thresh=thresh)
        their_corr_to_theirs = np.corrcoef(merged_all.iloc[-1, :], merged_all.iloc[-2, :]) / 3 + np.corrcoef(merged_all.iloc[-2, :], merged_all.iloc[-3, :]) / 3 + np.corrcoef(merged_all.iloc[-3, :], merged_all.iloc[-1, :]) / 3
        theirs_theirs[thresh] = their_corr_to_theirs[0,1]
        a, b = [], []
        for iter in list(range(iters)):
            ind1 = np.random.choice(list(range(len(df))), 1)[0] #by default, numpy returns an array, so pick the number it chose
            ind2 = np.random.choice(list(range(len(df))), 1)[0]
            ind3 = np.random.choice(list(range(len(df))), 1)[0]
            if len(set([ind1, ind2, ind3])) == 3: ##if there are no duplicate indices
                a.append(np.mean([np.corrcoef(merged_all.iloc[-their_ind, :], merged_all.iloc[our_ind, :])[0,1] for our_ind in [ind1, ind2, ind3] for their_ind in [1,2,3]]))
                our_corr_to_ours = np.corrcoef(merged_all.iloc[ind1, :], merged_all.iloc[ind2, :])/3 +  np.corrcoef(merged_all.iloc[ind2, :], merged_all.iloc[ind3, :])/3 +  np.corrcoef(merged_all.iloc[ind3, :], merged_all.iloc[ind1, :])/3
                b.append(our_corr_to_ours[0,1])
        our_theirs[thresh] = np.mean(a)
        our_ours[thresh] = np.mean(b)
    res = pd.DataFrame({"within_HPP_correlation" : our_ours, "between_correlation": our_theirs, "within_them_correlation" :theirs_theirs})
    res_longer = pd.melt(res.reset_index(), id_vars="index", value_vars=res.columns) ##pivot longer to plot more easily
    res_longer = res_longer.replace({"within_them_correlation":2, "between_correlation": 1, "within_HPP_correlation": 0}) ##use ordinal encoding
    plt.scatter(res_longer["index"], res_longer.value, c=res_longer.variable)
    plt.show()
    return res, res_longer

def flatten_values(df):
    df_flat = pd.DataFrame(np.array(df).flatten()) ##flatten to a vector to explore distribution
    vals = df_flat.value_counts()
    vals.index = list(map(lambda x: x[0], vals.index)) ##remove the tuple dimension from the index

def do_tsne(df, thresh, pheno):
    logged = clip_and_log(df, thresh)
    merged= make_regcode_and_merge (log_values_proj = TSNE().fit_transform(logged), orig_notlogged_df = logged, loaderdata = loaderdata)
    plt.scatter(merged[0], merged[1], c = merged[pheno])
    return merged

def do_split_and_normalize(data, Y, normalize = True):
    ##eran also wants to use the same sklearn methods
    ##eran wants to use the same train test split
    X_train, X_test, Y_train, Y_test = train_test_split(data, Y, test_size=0.2, random_state=0)
    orig_cols = data.columns
    ##save the original index, for reasons described below
    Y_trainindex, Y_testindex =  Y_train.index, Y_test.index
    X_trainindex, X_testindex =  X_train.index, X_test.index
    if normalize:
        scaler_X = StandardScaler().fit(X_train)
        X_train = scaler_X.transform(X_train)
        X_test = scaler_X.transform(X_test)
    ##reshape because we need another dimension for sklearn fitting or scaling methods to work
    if normalize:
        scaler_Y = StandardScaler().fit(np.array(Y_train).reshape(-1, 1))
        Y_train = scaler_Y.transform(np.array(Y_train).reshape(-1, 1))
        Y_test = scaler_Y.transform(np.array(Y_test).reshape(-1, 1))
    ##we want to be able to use pd based methods
    Y_train, Y_test = pd.DataFrame(Y_train), pd.DataFrame(Y_test)
    X_train, X_test = pd.DataFrame(X_train), pd.DataFrame(X_test)
    ##converting to an n array destroys the index, so replace it afrer normalization
    Y_train.index, Y_test.index =  Y_trainindex, Y_testindex
    X_train.index, X_test.index = X_trainindex, X_testindex
    X_train.columns, X_test.columns = orig_cols, orig_cols
    return X_train, X_test, Y_train, Y_test

def compare_cbc_and_rna_exp(counts_df, loaderdata, cbc_df, cbc_df_age, picked = "most_var", thresholds = [0.1, 0.5], response = "age", normalize = True, sizes_train_prop = [0.05, 0.075, 0.1, 0.15, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1]):
    data = counts_df.iloc[:, :-1].clip(1e-5)
    data = np.log10(data)
    #0.08: 1400, avg mean highest
    ##jacob used 0.5, I originally used 0.8
    i = 0
    models = [Lasso, Ridge, XGBRegressor]
    model_names = {Lasso: "Lasso", Ridge: "Ridge", XGBRegressor: "XGBoost"}
    total = len(models) * len(thresholds) * len(sizes_train_prop)
    cbc_df = cbc_df.drop(["Date"], axis=1).dropna()
    reses = {}
    for thresh in thresholds:
        if picked == "most_var":
            picked_genes = list(data.std()[data.std() > data.std().mean() + thresh*data.std().std()].index)
        elif picked == "random":
            picked_genes = np.random.choice(list(filter(lambda x: data.loc[:, x].std()> 1e-8 if data.loc[:, x].dtype == "float64" else False, data.columns)), 700)
        elif picked == "most_expressed":
            picked_genes = list(data.std()[data.mean() > data.mean().mean() + thresh*data.mean().std()].index)
        rna_df = data.loc[:, picked_genes]
        loaderdata = loaderdata.loc[~loaderdata.index.duplicated(keep="last"), :]
        data_all = cbc_df.merge(rna_df.merge(loaderdata, right_index=True, left_index=True, how='inner'), right_index=True, left_index=True, how='inner')
        if response != "age":
            Y = data_all[response]
        else:
            cbc_df_age["real_age"] = cbc_df_age["age"] ##there's already an age column in the loaderdata, so make sure you use the one from the rnaseq
            ##Furthermore, since that column already exists, when you merge it will suffix both, and the "age" column won't exist
            Y = cbc_df_age.merge(data_all, left_index = True, right_index = True, how = "inner")["real_age"]
        common_ids = list(set(data_all.index).intersection(set(cbc_df_age.index)))
        X_train, X_test, Y_train, Y_test = do_split_and_normalize(data_all.loc[common_ids, list(cbc_df.columns) + picked_genes], Y.loc[common_ids], normalize = normalize)
        for train_size in sizes_train_prop:
            effective_x_inds = np.random.choice(list(range(len(X_train))), int(train_size*len(X_train)))
            effective_x = X_train.iloc[effective_x_inds, :]
            effective_y = Y_train.iloc[effective_x_inds]
            for model in models:
                print(100*i/total, "% finished")
                if model != XGBRegressor:
                    fitted_model_RNA = model(fit_intercept=True, alpha = 0.1).fit(effective_x.loc[:, picked_genes], effective_y)
                    fitted_model_CBC = model(fit_intercept=True, alpha = 0.1).fit(effective_x.loc[:, cbc_df.columns], effective_y)
                else:
                    fitted_model_RNA = model(max_depth=5, n_estimators=1000, subsample = 0.7, learning_rate = 0.1, alpha =0.1).fit(effective_x.loc[:, picked_genes], effective_y) #eta 0.001 if this doesn't work
                    fitted_model_CBC = model(max_depth=5, n_estimators=1000, subsample = 0.7, learning_rate = 0.1, alpha = 0.1).fit(effective_x.loc[:, cbc_df.columns], effective_y)
                fitted_model_predictions_RNA = fitted_model_RNA.predict(X_test.loc[:, picked_genes])
                fitted_model_predictions_CBC = fitted_model_CBC.predict(X_test.loc[:, cbc_df.columns])
                res_RNA = pd.DataFrame({"pred": fitted_model_predictions_RNA.flatten(), "true": Y_test.to_numpy().flatten()})
                res_CBC = pd.DataFrame({"pred": fitted_model_predictions_CBC.flatten(), "true": Y_test.to_numpy().flatten()})
                reses[(model_names[model], train_size, thresh, len(picked_genes))] = {"RNA_Corr": res_RNA.corr().iloc[0,1], "CBC_Corr" : res_CBC.corr().iloc[0,1]}
                i += 1
    reses = pd.DataFrame(reses).T
    reses.index.names = ["model", "train_size", "RNA_variability_thresh", "number_of_picked_genes"]
    reses = reses.reset_index()
    ##reduce the two correlation columns to one correlation value column and a second "RNA vs CBC" identifier column
    reses = pd.melt(reses, id_vars = ["model", "train_size", "RNA_variability_thresh", "number_of_picked_genes"])
    return reses

if __name__ == "__main__":
    np.random.seed(0)
    base_dir = "/net/mraid20/export/jasmine/RNA/"
    metadata_df = pd.read_csv(base_dir + "metadata_without_participant_info.csv")
    metadata_df["participant_id"] = list(map(lambda x: "10K_" + str(x).split(".0")[0], metadata_df["participant_id"]))
    subs = SubjectLoader().get_data(study_ids=["10K"]).df.reset_index().set_index("RegistrationCode")
    body = BodyMeasuresLoader().get_data(study_ids=["10K"]).df.reset_index().set_index("RegistrationCode")
    loaderdata = subs.merge(how="inner", left_index=True, right_index=True, right=body).dropna(subset=["gender", "bmi"])

    ##### from nastya
    bt_df = BloodTestsLoader().get_data().df
    bt_df_meta = BloodTestsLoader().get_data().df_metadata
    cbc_lab_ind = bt_df_meta[bt_df_meta['kupat_holim'] == 'tenk-cbc'].index
    bt_df_cbc = bt_df.loc[cbc_lab_ind].dropna(how='all', axis=1)
    bt_df_cbc_age = bt_df_meta.loc[cbc_lab_ind]['age'].reset_index().set_index("RegistrationCode")
    bt_df_cbc = bt_df_cbc.reset_index().set_index("RegistrationCode")
    bt_df_cbc_age = bt_df_cbc_age.loc[~bt_df_cbc_age.index.duplicated(keep="last"), :]
    bt_df_cbc = bt_df_cbc.loc[~bt_df_cbc.index.duplicated(keep="last"), :]

    runs_summary = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/RNASeq/runs_summary.csv")
    runs_summary["Plates_Included_Numeric"] = list(map(lambda x: rack_extractor(x) if x is not None else [0], runs_summary["Plate Info"]))
    runs_summary = runs_summary.dropna()
    original_counts = pd.read_csv(base_dir + "corrected_counts.csv")  ##corrects for UMI, PCR amplification is not uniform
    counts_no_replicates, total_counts_no_reps = correct_counts(original_counts, metadata_df, runs_summary, True)
    counts_no_replicates_tde1 = counts_no_replicates.loc[list(
        map(lambda x: str(x) == "TDE1 (1uL TDE1, 14 cycles amplification)", counts_no_replicates["Methodology"])), :]
    counts_no_replicates_nextera = counts_no_replicates.loc[
                                   list(map(lambda x: str(x) == "Nextera XT", counts_no_replicates["Methodology"])), :]
    make_plot = True
    if make_plot:
        reses_tde1 = compare_cbc_and_rna_exp(counts_no_replicates_tde1, loaderdata, bt_df_cbc, bt_df_cbc_age)
        reses_nextera = compare_cbc_and_rna_exp(counts_no_replicates_nextera, loaderdata, bt_df_cbc, bt_df_cbc_age)
        reses_tde1["Method"] = "TDE1"
        reses_nextera["Method"] = "nextera"
        reses_both = pd.concat([reses_tde1, reses_nextera])
        reses_both["Type"] = reses_both.apply(
            lambda row: str(row["RNA_variability_thresh"]) + " " + str(
                row["variable"]), axis=1)
        for model in reses_both["model"].unique():
            for method in reses_both["Method"]:
                reses_both_eff = reses_both.loc[reses_both["model"] == model,:]
                reses_both_eff = reses_both_eff.loc[reses_both_eff["Method"] == method,:]
                colours = [cm.jet(x) for x in np.linspace(0, 1, len(reses_both_eff["Type"].unique()))]
                plt.figure(figsize = (15,15))
                for i in range(len(reses_both_eff["Type"].unique())):
                    kind = reses_both_eff["Type"].unique()[i]
                    df = reses_both_eff.loc[reses_both_eff["Type"] == kind,:]
                    plt.plot(df["train_size"], df["value"], c= colours[i], label = kind)
                plt.legend()
                plt.title(model + "" + str(method))
                plt.xlabel("Sample Size Frac (train)")
                plt.ylabel("Prediction Correlation to True Values")
                os.chdir("/home/zacharyl/")
                plt.savefig(str(model) + "" + str(method) + ".jpg")
                plt.close()
