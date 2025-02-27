import numpy as np
import pandas as pd
import torch
from LabQueue.qp import qp
from LabUtils.addloglevels import sethandlers
from torch import optim
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from sklearn.impute import KNNImputer
import statsmodels.api as sm
import os
from torch import nn as nn
from LabData.DataLoaders.SubjectLoader import SubjectLoader
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from LabData.DataLoaders.ItamarSleepLoader import ItamarSleepLoader
from LabData.DataLoaders.SerumMetabolomicsLoader import SerumMetabolomicsLoader
from LabData.DataLoaders.RetinaScanLoader import RetinaScanLoader
from LabData.DataUtils.DataProcessing import NormDistCapping
from LabData.DataLoaders.GutMBLoader import GutMBLoader
from LabData.DataLoaders.ECGTextLoader import ECGTextLoader
from LabData.DataLoaders.MedicalConditionLoader import MedicalConditionLoader
from LabData.DataLoaders.PRSLoader import PRSLoader
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
import seaborn as sns
import umap
from sklearn.decomposition import PCA

norm_dist_capping = {"sample_size_frac": 0.95, "remove_sigmas": 5}
keep = "first"

def drop_imbalance(df):
    df_imbalanced = df.columns[list(
        map(lambda colName: max(df[colName].value_counts().values) > 0.95 * len(
            df[colName].dropna()), df.columns))]
    df_now = df.drop(labels=df_imbalanced, axis=1)
    print("Dropped " + str(len(df_imbalanced)) + " cols based on imbalance")
    return df_now

def fix_norm_dist_capping_type_conversion(notCappedDf, cappedDf):
    df_col_types_before_norm_dist_capping = notCappedDf.dtypes
    for i in range(len(cappedDf.columns)):
        if cappedDf.dtypes.values[i] != df_col_types_before_norm_dist_capping[i]:
            cappedDf[cappedDf.columns[i]] = cappedDf[cappedDf.columns[i]].astype(
                df_col_types_before_norm_dist_capping[i])
    return cappedDf

def make_mb_df(id_list):
    df_all_data = fix_norm_dist_capping_type_conversion(
        GutMBLoader().get_data("segal_species", study_ids=id_list).df.copy(),
        GutMBLoader().get_data("segal_species", study_ids=id_list,
                          norm_dist_capping=norm_dist_capping).df.copy())
    dfmeta = GutMBLoader().get_data("segal_species", study_ids=id_list).df_metadata
    df = df_all_data.reset_index(drop=False)
    df["RegistrationCode"] = df.reset_index(drop=False).SampleName.apply(dfmeta.RegistrationCode.to_dict().get)
    df = df.set_index("RegistrationCode").drop("SampleName", axis=1)
    df = df.loc[~df.index.duplicated(keep = keep),:]
    df = np.log10(df.clip(0.0001))
    df_now = drop_imbalance(df)
    df_now = df_now.fillna(-4)
    return df_now

def get_elapsed_seconds(time_str):
    if pd.isna(time_str): return np.nan
    h, m, s = time_str.split(":")
    return float(h) * 3600 + float(m) * 60 + float(s)

def PhysicalTime_seconds(date_time_str):
    if pd.isna(date_time_str): return np.nan
    elapsedTime = date_time_str.split(" ")[-1]
    return get_elapsed_seconds(elapsedTime)

##The RT cluster is already logged, don't do it again
def make_metab_df(id_list):
    df_metab = fix_norm_dist_capping_type_conversion(
            SerumMetabolomicsLoader().get_data(precomputed_loader_fname="metab_10k_data_RT_clustering_pearson08_present05_baseline",
                              study_ids=id_list).df.copy(),
            SerumMetabolomicsLoader().get_data(precomputed_loader_fname="metab_10k_data_RT_clustering_pearson08_present05_baseline",
                              study_ids=id_list, norm_dist_capping=norm_dist_capping).df.copy())
    df_metab["RegistrationCode"] = list(map(lambda serum: '10K_' + serum.split('_')[0], df_metab.index.values))
    df_metab = df_metab.set_index("RegistrationCode")
    df_metab = df_metab.loc[~df_metab.index.duplicated(keep = keep),:]
    df_metab = drop_imbalance(df_metab)
    return df_metab

#Inspired by stack overflow
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, mode = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.mode = mode
        if self.mode == "max":
            self.best_metric = -np.inf
        else:
            self.best_metric = np.inf

    def early_stop(self, curr_auc, model):
        if self.mode == "max":
            if curr_auc > self.best_metric:
                self.best_metric = curr_auc
                self.counter = 0
                self.best_model = model
            elif curr_auc < (self.best_metric - self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        else: # mode == min
            if curr_auc < self.best_metric:
                self.best_metric = curr_auc
                self.counter = 0
                self.best_model = model
            elif curr_auc > (self.best_metric + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        return False

def train_predictor(model, device, X_train, X_test, Y_train, Y_test, X_validation, Y_validation, baseData, mode = "continuous"):
    model = model.to(device)
    if mode == "disease_pred":
        loss_fn = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr = 1e-4)
        epochs = 250
        batch_size = 16
        early_stopper = EarlyStopper(patience=10, min_delta=0, mode= "max") ##very patient because sometimes it takes a while for the AUC to increase at the begining, the one downside with this approach is that if we are too patient and then return the model after damaging it after overfitting, that's not good either.
    else:
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        epochs = 200
        batch_size = 16
        early_stopper = EarlyStopper(patience=10, min_delta=0, mode= "min")
    for epoch in range(epochs):
        arrs = np.array_split(np.random.choice(X_train.shape[0], size = X_train.shape[0], replace = False), X_train.shape[0]//batch_size)
        model.train()
        avg_loss = 0
        for arr in arrs:
            X_batch = torch.Tensor(X_train[arr,:]).to(device)
            Y_batch = torch.Tensor(Y_train[arr]).to(device).flatten()
            pred = model(X_batch).flatten()
            loss = loss_fn(pred, Y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss/len(arrs)
        print(avg_loss.item(), ", train, epoch: ",  epoch)
        with torch.no_grad():
            avg_loss_eval = 0
            model.eval()
            arrs_eval = np.array_split(np.random.choice(X_validation.shape[0], size =X_validation.shape[0], replace=False),
                                  X_validation.shape[0]/ batch_size)
            for arr_eval in arrs_eval:
                X_batch_eval = torch.Tensor(X_test[arr_eval, :]).to(device)
                Y_batch_eval = torch.Tensor(Y_test[arr_eval]).to(device).flatten()
                pred = model(X_batch_eval).flatten()
                avg_loss_eval += loss_fn(pred, Y_batch_eval) / len(arr_eval)
            print("eval loss: ", avg_loss_eval.item())
            if early_stopper.early_stop(avg_loss_eval, model):
                print("stopping at epoch: ", epoch)
                avg_loss_test = 0
                model.eval()
                arrs_test = np.array_split(np.random.choice(X_test.shape[0], size=X_test.shape[0], replace=False),X_test.shape[0] / batch_size)
                for arr_test in arrs_test:
                    X_batch_test = torch.Tensor(X_test[arr_test, :]).to(device)
                    Y_batch_test = torch.Tensor(Y_test[arr_test]).to(device).flatten()
                    pred = model(X_batch_test).flatten()
                    loss_test = loss_fn(pred, Y_batch_test)
                    avg_loss_test += loss_test / len(arrs_test)
                print("test, error: ", avg_loss_test)
                return early_stopper.best_model
    return model

def normalize_no_spillover(X_train, X_test, Y_train, Y_test, X_validation = None, Y_validation = None, scale_y = True):
    print("scaling")
    ##save the original index as scaling casts to np array
    imputer_X = KNNImputer().fit(X_train)
    X_train = imputer_X.transform(X_train)
    X_test = imputer_X.transform(X_test)
    scaler_X = StandardScaler().fit(X_train)
    X_train = scaler_X.transform(X_train)
    X_test = scaler_X.transform(X_test)
    if X_validation is not None:
        X_validation = scaler_X.transform(X_validation)
    if scale_y:
        scaler_Y = StandardScaler().fit(np.array(Y_train).reshape(-1, 1))
        Y_train = scaler_Y.transform(np.array(Y_train).reshape(-1, 1)).reshape(-1)
        Y_test = scaler_Y.transform(np.array(Y_test).reshape(-1, 1)).reshape(-1)
        if Y_validation is not None:
            Y_validation = scaler_Y.transform(np.array(Y_validation).reshape(-1, 1)).reshape(-1)
    if X_validation is None:
        return X_train, X_test, Y_train, Y_test
    else:
        return X_train, X_test, Y_train, Y_test, X_validation, Y_validation

def combine_test_validation(df_emb, featureSeries, train_ids, how, validation_ids, test_ids, to_pred, scale_y = True):
    X_validation, Y_validation = None, None
    df_merged = df_emb.merge(featureSeries, how="inner", left_index=True, right_index=True).reset_index("Date", drop=True)
    X_train = df_merged.loc[train_ids, list(filter(lambda x: x != to_pred, df_merged.columns))].to_numpy(
        dtype=np.float64)
    Y_train = df_merged.loc[train_ids, to_pred].to_numpy(dtype=np.float64)
    if how != "dl":
        X_test = df_merged.loc[
            test_ids + validation_ids, list(filter(lambda x: x != to_pred, df_merged.columns))].to_numpy(
            dtype=np.float64)
        Y_test = df_merged.loc[test_ids + validation_ids, to_pred].to_numpy(dtype=np.float64)
        X_train, X_test, Y_train, Y_test = normalize_no_spillover(X_train, X_test, Y_train,
                                                                  Y_test,
                                                                  X_validation=X_validation,
                                                                  Y_validation=Y_validation,
                                                                  scale_y=scale_y)
    else:
        X_test = df_merged.loc[test_ids, list(filter(lambda x: x != to_pred, df_merged.columns))].to_numpy(
            dtype=np.float64)
        X_validation = df_merged.loc[validation_ids, list(filter(lambda x: x != to_pred, df_merged.columns))].to_numpy(
            dtype=np.float64)
        Y_test = df_merged.loc[test_ids, to_pred].to_numpy(dtype=np.float64)
        Y_validation = df_merged.loc[validation_ids, to_pred].to_numpy(dtype=np.float64)
        X_train, X_test, Y_train, Y_test, X_validation, Y_validation = normalize_no_spillover(X_train, X_test, Y_train,
                                                                                              Y_test,
                                                                                              X_validation=X_validation,
                                                                                              Y_validation=Y_validation,
                                                                                              scale_y=scale_y)
        print("validating predictor on ", len(validation_ids), "people, corresponding to ", X_validation.shape[0],
              " samples")
    print("training predictor on ", len(train_ids), "people, corresponding to ", X_train.shape[0], " samples")
    print("testing predictor on ", X_test.shape[0], "people, corresponding to ", X_test.shape[0], " samples")
    return X_train, X_test, Y_train, Y_test, X_validation, Y_validation

def predict_binary_phenotype(featureSeries, df_emb, to_pred, baseData, device, how, train_ids, test_ids, validation_ids, alpha = 0, L1_wt=1):
    X_train, X_test, Y_train, Y_test, X_validation, Y_validation =  combine_test_validation(df_emb, featureSeries, train_ids, how, validation_ids, test_ids, to_pred, scale_y = False)
    if how == "dl":
        from tcn import SmallPredictor
        from torcheval.metrics.functional import binary_auprc
        model = train_predictor(SmallPredictor(192), device, X_train, X_test, Y_train, Y_test, X_validation, Y_validation, baseData=baseData, mode  = "anythingelse")
        model.eval()
        predicted_test = model(torch.Tensor(X_test).to(device)).detach().cpu().numpy().flatten()
    elif how == "xgboost":
        model = XGBClassifier(n_estimators=1000, max_depth=4, learning_rate=0.02, objective='binary:logistic').fit(X_train, Y_train)
        predicted_test = model.predict(X_test)
    else:
        model = sm.Logit(Y_train, X_train).fit_regularized(alpha = alpha, L1_wt = L1_wt)
        predicted_test = model.predict(X_test)
    predicted_test = pd.DataFrame(predicted_test)
    true_test = pd.DataFrame(Y_test)
    return predicted_test, true_test

def predict_continuous_phenotype(featureSeries, df_emb, to_pred, baseData, device, how, train_ids, test_ids, validation_ids, alpha = 0, L1_wt=1):
    X_train, X_test, Y_train, Y_test, X_validation, Y_validation =  combine_test_validation(df_emb, featureSeries, train_ids, how, validation_ids, test_ids, to_pred, scale_y = True)
    if how == "dl":
        from tcn import SmallPredictor ##this probably won't work on the queue, but cluster11 probably it will
        model = train_predictor(SmallPredictor(192), device, X_train, X_test, Y_train, Y_test, X_validation, Y_validation, baseData=baseData, mode  = "anythingelse")
        model.eval()
        predicted_test = model(torch.Tensor(X_test).to(device)).detach().cpu().numpy().flatten()
    elif how == "xgboost":
        model = XGBRegressor(n_estimators=2000, max_depth=5, learning_rate=0.1).fit(X_train, Y_train)
        predicted_test = model.predict(X_test)
    else:
        model = sm.OLS(Y_train, X_train).fit_regularized(alpha = alpha, L1_wt = L1_wt)
        predicted_test = model.predict(X_test)
    test_r = pearsonr(predicted_test, Y_test)
    predicted_test = pd.DataFrame(predicted_test)
    true_test = pd.DataFrame(Y_test)
    return predicted_test, true_test, test_r

def make_RNASeq_df():
    def rack_extractor(rack_string):
        try:
            rack_string = str(rack_string)
            if rack_string == "Rack_":
                return [0]
            elif "-" in rack_string:
                return list(range(int(rack_string.split(" ")[-1].split("B")[0].split("-")[0]),
                                  int(rack_string.split(" ")[-1].split("B")[0].split("-")[1]) + 1))
            else:
                return [int(rack_string.split(" ")[-1].split("B")[0])]
        except (TypeError, ValueError) as e:
            return [0]

    def correct_counts(counts_df, metadata_df_used, runs_summary, drop_duplicates=True):
        metadata_df_used["Run_Number"] = -1
        for i in range(len(metadata_df_used["Sample Description"])):
            try:
                metadata_df_used["Run_Number"].iloc[i] = int(
                    str(metadata_df_used["Sample Description"].iloc[i]).split("Rack")[-1].split("_")[0])
            except ValueError:
                pass
        norm_counts = metadata_df_used.merge(right=counts_df, left_on=metadata_df.columns[0], right_on="SampleName",
                                             how="inner")
        norm_counts = norm_counts.set_index("participant_id")
        if drop_duplicates:
            norm_counts = norm_counts.loc[~norm_counts.index.duplicated(keep="last"), :]
        backup_total_counts = norm_counts["corrected_count_sum"]
        backup_run_number = norm_counts["Run_Number"]
        metacols = list(metadata_df_used.columns)
        metacols.remove(
            "participant_id")  ##setting the index removes this column so we can't drop it, do this to avoid the error in the drop column
        norm_counts = norm_counts.drop(metacols, axis=1)  ##keep only numeric gene columns
        norm_counts = norm_counts.loc[:, list(norm_counts.dtypes[norm_counts.dtypes == "int64"].index)]
        counts = norm_counts.divide(backup_total_counts, axis="rows")
        counts["Methodology"] = "Unknown"
        for id in backup_run_number.index:
            run_number = backup_run_number.loc[id]
            for i in range(len(runs_summary["Plates_Included_Numeric"])):
                if run_number in runs_summary["Plates_Included_Numeric"].iloc[i]:
                    methodology = runs_summary.iloc[i, :]["Library Prep. Metadology "]
                    counts.loc[id, "Methodology"] = methodology
        return counts, backup_total_counts

    base_dir = "/net/mraid20/export/jasmine/RNA/"
    metadata_df = pd.read_csv(base_dir + "metadata_without_participant_info.csv")
    metadata_df["participant_id"] = list(
            map(lambda x: "10K_" + str(x).split(".0")[0], metadata_df["participant_id"]))
    runs_summary = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/RNASeq/runs_summary.csv")
    runs_summary["Plates_Included_Numeric"] = list(
            map(lambda x: rack_extractor(x) if x is not None else [0], runs_summary["Plate Info"]))
    runs_summary = runs_summary.dropna()

    original_counts = pd.read_csv(base_dir + "corrected_counts.csv")  ##corrects for UMI, PCR amplification is not uniform
    counts_no_replicates, total_counts_no_reps = correct_counts(original_counts, metadata_df, runs_summary, True)
    counts_no_replicates_tde1 = counts_no_replicates.loc[list( map(lambda x: str(x) == "TDE1 (1uL TDE1, 14 cycles amplification)", counts_no_replicates["Methodology"])), :]
    print(counts_no_replicates_tde1.iloc[:, -1])
    counts_no_replicates_tde1 = counts_no_replicates_tde1.iloc[:, :-1].clip(1e-5)
    counts_no_replicates_tde1 = np.log10(counts_no_replicates_tde1)
    return counts_no_replicates_tde1

def make_cgm_df():
    df_cgm = pd.read_csv("/net/mraid20/export/genie/LabData/Analyses/ayyak/CGM/iglu/iglu_no_tails.csv").set_index(
            "id").drop("Unnamed: 0", axis=1)
        ##throw away second part of index
    df_cgm.index = list(map(lambda longName: longName.split("/")[0], df_cgm.index.values))
    df_cgm.index.name = "RegistrationCode"
        ##perform the same outlier removal as above
    df_cgm = fix_norm_dist_capping_type_conversion(
        pd.read_csv("/net/mraid20/export/genie/LabData/Analyses/ayyak/CGM/iglu/iglu_no_tails.csv").set_index(
            "id").drop("Unnamed: 0", axis=1),
        cappedDf=NormDistCapping(sample_size_frac=0.95, remove_sigmas=5).fit_transform(df_cgm))
        ##perform the same grouping of multiple entries as above
    df_cgm = df_cgm.loc[~df_cgm.index.duplicated(keep = keep),:]
    df_cgm = drop_imbalance(df_cgm)
    return df_cgm

def make_retina_df(id_list):
        ##Encode left and right eye measurements as separate columns instead of indexes
    df = RetinaScanLoader().get_data(study_ids=id_list).df.copy().unstack()  ##Don't groupby reg here because the extra index (right vs left eye) will mess things up
    df.columns = list(map(lambda col_tuple: str(col_tuple[0]) + "_" + str(col_tuple[1]), df.columns))
    df = fix_norm_dist_capping_type_conversion(df, cappedDf=NormDistCapping(sample_size_frac=0.95,
                                                                                remove_sigmas=5).fit_transform(
        df))
    df = df.loc[~df.index.duplicated(keep=keep), :]
    df_retina = drop_imbalance(df)
    return df_retina

def make_sleep_df(id_list):
    df = fix_norm_dist_capping_type_conversion(notCappedDf=ItamarSleepLoader().get_data(study_ids=id_list, groupby_reg = "first").df.copy(),
                                               cappedDf=ItamarSleepLoader().get_data(study_ids=id_list,
                                               norm_dist_capping=norm_dist_capping, groupby_reg = "first").df.copy())
    df["PhysicalSleepTime"] = df["PhysicalSleepTime"].apply(PhysicalTime_seconds)
    df["PhysicalWakeTime"] = df["PhysicalWakeTime"].apply(PhysicalTime_seconds)
    for col in df.dtypes.index[df.dtypes != "float64"]:
        df[col] = df[col].apply(lambda x: x[0] if isinstance(x,list) else x)  ##drop lists from the dataframe if they're an entry, an edge case
    df = df.drop(["BraceletMessage", "StudyStartTime", "StudyEndTime"], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.reset_index(["night", "Date"], drop = True)
    df = df.loc[~df.index.duplicated(keep = keep),:].drop(["StudyStatus", "Warnings"], axis = 1)
    return df

def do_CV(df_operative, var, all_people, operative_people, test_folds, emb_df, device, alpha, L1_wt, which):
    trues, preds = [], []
    theSeriestoPred = df_operative[var].dropna()
    missing_people_for_this_far = list(set(emb_df.index.get_level_values(0)) - set(theSeriestoPred.index))
    for test_validation_people in test_folds:
        test_validation_people = list(test_validation_people)
        train_people = list(set(all_people) - set(test_validation_people))
        validation_people = list(
            np.random.choice(test_validation_people, replace=False, size=len(test_validation_people) // 2))
        test_people = list(set(test_validation_people) - set(validation_people))
        train_operative = list(set(train_people).intersection(operative_people) - set(missing_people_for_this_far))
        test_operative = list(set(test_people).intersection(operative_people) - set(missing_people_for_this_far))
        validation_operative = list(
            set(validation_people).intersection(operative_people) - set(missing_people_for_this_far))
        if which != "DiseaseDiagnoses":
            predicted_test, true_test, test_r = predict_continuous_phenotype(theSeriestoPred,
                                                                         emb_df, var, "DL", device,
                                                                         "ols", train_operative, test_operative,
                                                                         validation_operative, alpha=alpha, L1_wt=L1_wt)
        else:
            predicted_test, true_test = predict_binary_phenotype(theSeriestoPred,
                                                                         emb_df, var, "DL", device,
                                                                         "ols", train_operative, test_operative,
                                                                         validation_operative, alpha=alpha, L1_wt=L1_wt)
        trues.append(true_test)
        preds.append(predicted_test)
    res = pd.concat([pd.concat(trues, axis=0), pd.concat(preds, axis=0)], axis=1)
    return res

def make_diagnoses_df(base_data, id_list = list(range(100)) + list(range(1000, 1011, 1))):
    b = MedicalConditionLoader().get_data(study_ids=id_list).df.reset_index()
    counts = base_data.reset_index(["Date"], drop = True).merge(b, left_index=True, right_on="RegistrationCode",  how="inner").medical_condition.value_counts()
    diseases = list(counts.index[counts > 100])
    ##not worth predicting:
    diseases.remove("ME84") ##Nerve Disease
    diseases.remove("6A05") ##ADHD
    diseases.remove("CA08.0") ##Common cold
    diseases.remove("RA01") ##covid -not reliable
    diseases.remove("CA0A") ##Common cold
    diseases.remove("AB31") ##Vertigo
    diseases.remove("ND56.2") ##Injury
    diseases.remove("ED80") ##Acne
    diseases.remove("DB50.0") ##Anal fissure
    ##remove depression
    diseases = list(filter(lambda x: not x.startswith("Block"), diseases))
    res = pd.DataFrame(np.zeros([len(list(set(base_data.index.get_level_values(0)))), len(diseases)]), dtype = int)
    res["RegistrationCode"] = list(set(base_data.index.get_level_values(0)))
    res = res.set_index(["RegistrationCode"])
    res.columns = diseases
    b = b.set_index("RegistrationCode").loc[list(set(base_data.index.get_level_values(0)).intersection(set(b["RegistrationCode"])))].reset_index()
    for cond_string in diseases:
        people_with_disease = list(set(b.loc[b.medical_condition == cond_string, :]["RegistrationCode"]))
        res.loc[people_with_disease, cond_string] = 1
    return res

def make_PRS_df():
    p = PRSLoader().get_data().df
    p_meta = PRSLoader().get_data().df_columns_metadata.h2_description.to_dict()
    p.columns = list(map(lambda x: p_meta.get(x), p.columns))
    p = p.loc[:, list(filter(lambda x: type(x) != float and x is not None, p.columns))]
    return p

##will not run on cluster 11
def do(gender, which = "RNA", baseData = "embeddings", fileName = ""):
    id_list = list(range(100)) + list(range(1000, 1011, 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if which == "RNA":
        df = make_RNASeq_df()
        ##3k most variably expressed genes
        df.index.name = "RegistrationCode"
    elif which == "PRS":
        df = make_PRS_df()
    elif which == "CGM":
        df = make_cgm_df()
        df  = df.loc[~df.index.duplicated(keep = "first")]
    elif which == "Retina":
        df = make_retina_df(id_list).reset_index(["Date"], True)
        df  = df.loc[~df.index.duplicated(keep = "first")]
    elif which == "GutMB":
        df = make_mb_df(id_list)
    elif which == "Sleep":
        df = make_sleep_df(id_list)
    elif which == "Metabolomics":
        df = make_metab_df(id_list)
    ##from Nastya
    elif which == "Nightingale Metabolomics":
        df = pd.read_csv("/net/mraid20/export/genie/LabData/Data/metabolomics/Nightingale Metabolomics/results_with_reg_code.csv").set_index("RegistrationCode").iloc[:, 2:] ##drop the sample id column and the redundant ingeger index column
        df = df.iloc[:, 0:250] ##also drop some unnnecessary columns at the end
        df = df.loc[~df.index.duplicated(keep="first"), :]
        ##set "TAG" to be a missing value, so it's dropped by default
        df = df.replace('TAG', np.nan)
    if baseData == "embeddings":
        emb_df = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/" + fileName).set_index(["RegistrationCode", "Date"])
    else:

        ecg_with_date = ECGTextLoader().get_data(research_stage="baseline", study_ids=list(range(100)) + list(range(1000, 1011, 1))).df.reset_index(["Date"], drop=False)
        ecg_with_date = ecg_with_date.loc[pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/no_ft/one_no_linear.csv")["RegistrationCode"], :]
        ecg_with_date = ecg_with_date.fillna(0).select_dtypes(exclude=object)
        ecg_with_date = ecg_with_date.loc[~ecg_with_date.index.get_level_values(0).duplicated(keep="first"),:]
        # some people for whom we only have the pheno data from one apt may have more than one apt in ECGText, causing duplicates here
        ##pick the first appointment for these people, which would be the one we have the raw data for
        for var in ecg_with_date.columns:
            if var != "Date":
                try:
                    ecg_with_date[var] = ecg_with_date[var].astype(np.float64)
                except ValueError:
                    ecg_with_date.drop(var, axis=1, inplace=True)
        emb_df = ecg_with_date.set_index(["Date"], append = True)
    if which  == "DiseaseDiagnoses":
        df = make_diagnoses_df(emb_df, id_list)
    print("read in splits saved previously")
    all_people = list(set(pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/no_ft/all_no_linear.csv").RegistrationCode))
    test_folds = np.array_split(all_people, 5)
    subs = SubjectLoader().get_data(study_ids= id_list).df.reset_index(["Date"], drop=True)
    subs = subs.loc[~subs.index.duplicated(keep="first"), "gender"]
    subs = subs.dropna().astype(int)
    if gender == 2:
        this_gender = set(subs.index.values)
    else:
        this_gender = set(subs.loc[subs == gender].index.values)
    alphas = [1]
    weights = [0]
    if which == "DiseaseDiagnoses":
        operative_people = list(this_gender.intersection(set(df.index.get_level_values(0))))
    else:
        operative_people = list(this_gender.intersection(set(df.index)))
    df_operative = df.loc[operative_people, :]
    if which == "RNA":
        df_operative = df_operative.loc[:, list(df_operative.std().sort_values(ascending=False).index[0:3000])]
    rs = {}
    for var in df_operative.columns:
        try:
            df[var].astype(np.float64)
        except ValueError:
            continue
        for alpha in alphas:
            for L1_wt in weights:
                res = do_CV(df_operative, var, all_people, operative_people, test_folds, emb_df, device, alpha, L1_wt, which)
                try:
                    if which != "DiseaseDiagnoses":
                        r, P = pearsonr(res.iloc[:, 0].values, res.iloc[:, 1].values)
                        rs[(var, alpha, L1_wt)] = {"r": r, "P": P, "N": len(res)}
                    else:
                        auc = roc_auc_score(res.iloc[:, 0].values, res.iloc[:, 1].values)
                        rs[(var, alpha, L1_wt)] = {"AUC": auc, "N": len(res)}
                        ##this takes so long, sometimes the queue kills it after a week, so save the intermediate output each time just in case
                        pd.DataFrame(rs).T.to_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_predictors/" + baseData + "/" + which + "_" + str(gender) + ".csv")
                except Exception:
                    pass
    pd.DataFrame(rs).T.to_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_predictors/" + baseData + "/" + which + "_" + str(gender) + ".csv")

def do_gender_strat_pred(loaders, baseDatas, fileName):
    tickets = {}
    with qp(jobname="zach", _mem_def=256, _trds_def=18, delay_batch = 64) as q:
        q.startpermanentrun()
        for loader in loaders:
            for baseData in baseDatas:
                if loader == "RNA":
                    genderList = [0,1]
                else:
                    genderList = [2]
                for gender in genderList: #2:both
                    tickets[gender] = q.method(do, (gender, loader, baseData, fileName))
        for k,v in tickets.items():
            try:
                tickets[k] = q.waitforresult(v)
            except Exception:
                pass

def read_gene_annotation_file(fname = "/net/mraid20/export/jasmine/zach/cross_modal/gencode.gtf"):
    gencode = pd.read_table(fname, comment = "#", sep = "\t", names = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand',
                                                   'frame', 'attribute'])
    gencode["gene"] = list(map(lambda x: x.split("gene_name ")[-1].split(";")[0].strip('"'), gencode["attribute"]))
    gencode = gencode.drop(["source", "score"], axis = 1)
    return gencode

def gene_name_to_chromosome(gene, gencode):
    try:
        temp = gencode.loc[gencode.gene == gene,"seqname"].iloc[0].split("chr")[-1]
        print(temp)
    except Exception:
        temp = None
    return temp

##important: need to rank genes by variability of expression first
def multi_stage_correction(res_ranked):
    i = 1
    res_ranked["P*"] = 1
    while i <= len(res_ranked):
        res_ranked.loc[:,"P*"].iloc[i-1] = multipletests(pvals=res_ranked.iloc[0:i,:]["P"].to_numpy().flatten(), method="fdr_bh", returnsorted=False)[1][i-1]
        i += 1
    return res_ranked

def annotate_chr(res, gencode):
    i = 0
    for i in range(len(res.index.values)):
        res["CHR"][i] = gene_name_to_chromosome(res.index.values[i], gencode = gencode)
        print(i)
        i += 1
    return res

def read_res_file(name):
    df = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_predictors/" + name)
    if "Diagnoses" in name:
        df.columns = ["feature", "alpha", "weight", "AUC", "N"]
    else:
        df.columns = ["feature", "alpha", "weight", "r", "P", "N"]
    df = df.set_index(["feature", "alpha", "weight"])
    return df

def plot_r(df, title, fname):
    plt.figure()
    g = sns.scatterplot(x = np.abs(df["r_text"]), y = np.abs(df["r_emb"]), hue = df["P*_emb"] < 0.05, palette = sns.dark_palette("grey", n_colors=2, reverse=True, as_cmap=False, input='rgb'))
    plt.xlabel("Baseline Clinical |r|")
    plt.ylabel("Embeddings |r|")
    handles, labels = g.get_legend_handles_labels()
    g.legend(handles=handles, labels=["N", 'Y'], title="Significance")
    g.plot([0, 0.4], [0, 0.4], '-', linewidth=2, color = "black")
    plt.title(title)
    plt.savefig(fname)

def plot_bars(df, title, fname):
    plt.figure()
    sns.barplot(x = df.index.values, y=np.abs(df["r"]), hue = df["Source"], palette = sns.color_palette(["#c8e4b4", "#8894b4"]))
    plt.legend()
    plt.title(title)
    plt.ylabel("| r |")
    plt.savefig(fname)

def make_figures(which = "RNA", type = "dist"):
    if which == "RNA":
        women_emb = read_res_file("embeddings/RNA_0.csv")
        men_emb = read_res_file("embeddings/RNA_1.csv")
        women_text = read_res_file("ECGText/RNA_0.csv")
        men_text = read_res_file("ECGText/RNA_1.csv")
        women_emb = women_emb.loc[women_emb.index.get_level_values(1) == 1,:]
        men_emb = men_emb.loc[men_emb.index.get_level_values(1) == 1,:]
        women_text = women_text.loc[women_text.index.get_level_values(1) == 1,:]
        men_text = men_text.loc[men_text.index.get_level_values(1) == 1,:]
        women_emb = multi_stage_correction(women_emb.loc[women_emb.index.get_level_values(2) == 0.0,:]).reset_index(["alpha", "weight"], drop = True)
        men_emb = multi_stage_correction(men_emb.loc[men_emb.index.get_level_values(2) == 0.0,:]).reset_index(["alpha", "weight"], drop = True)
        women_text = multi_stage_correction(women_text.loc[women_text.index.get_level_values(2) == 0.0,:]).reset_index(["alpha", "weight"], drop = True)
        men_text = multi_stage_correction(men_text.loc[men_text.index.get_level_values(2) == 0.0,:]).reset_index(["alpha", "weight"], drop = True)
        if type == "dist":
            men_df = men_emb.merge(men_text, left_index=True, right_index=True,
                                                                suffixes=["_emb", "_text"])
            women_df = women_emb.merge(women_text, left_index=True, right_index=True,
                                                                      suffixes=["_emb", "_text"])
           ##for Yeela
           # men_df.loc[men_df["P*_emb"] < 0.05,:].to_csv("~/Desktop/RNA_Men.csv")
           # women_df.loc[women_df["P*_emb"] < 0.05,:].to_csv("~/Desktop/RNA_Women.csv")

            plot_r(men_df, "RNA: Men, Distribution", "/home/zacharyl/Desktop/RNA_1.png")
            plot_r(women_df, "RNA: Women, Distribution", "/home/zacharyl/Desktop/RNA_0.png")
        else:
            men_emb["Source"] = "embeddings"
            men_text["Source"] = "Baseline"
            women_emb["Source"] = "embeddings"
            women_text["Source"] = "Baseline"
            men_emb = men_emb.loc[men_emb["P*"] < 0.05,:]
            women_emb = women_emb.loc[women_emb["P*"] < 0.05,:]
            men_emb = men_emb.loc[np.abs(men_emb["r"]) > 0.14]
            women_emb = women_emb.loc[np.abs(women_emb["r"]) > 0.12]
            men_df = pd.concat([men_emb, men_text.loc[men_emb.index.values,:]], axis = 0)
            women_df = pd.concat([women_emb, women_text.loc[women_emb.index.values,:]], axis = 0)
            plot_bars(men_df, "RNA: Men, Top Associations", "/home/zacharyl/Desktop/RNA_1.png")
            plot_bars(women_df, "RNA: Women, Top Asociations", "/home/zacharyl/Desktop/RNA_0.png")
    else:
        emb = read_res_file("embeddings/" + which + "_2.csv").dropna()
        text = read_res_file("ECGText/" + which + "_2.csv").dropna()

        emb = emb.loc[emb.index.get_level_values(1) == 1,:]
        text = text.loc[text.index.get_level_values(1) == 1,:]
        emb = emb.loc[emb.index.get_level_values(2) == 0.0,:].reset_index(["alpha", "weight"], drop = True)
        text = text.loc[text.index.get_level_values(2) == 0,:].reset_index(["alpha", "weight"], drop = True)
        emb["P*"] = multipletests(emb["P"].to_numpy().flatten(),method="fdr_bh", returnsorted=False)[1]
        text["P*"] = multipletests(text["P"].to_numpy().flatten(),method="fdr_bh", returnsorted=False)[1]
        if type == "dist":
            df = emb.merge(text, left_index=True, right_index=True, suffixes=["_emb", "_text"])
            plot_r(df, "Prediction of " + str(which), "/home/zacharyl/Desktop/" + str(which) + ".png")
        else:
            if which == "Retina":
                r_thresh = 0.21
            elif which == "Nightingale Metabolomics":
                r_thresh = 0.45
            elif which == "CGM":
                r_thresh = 0.14
            else:
                r_thresh = 0.1
            emb  = emb.loc[emb["P*"] < 0.05, :]
            emb  = emb.loc[np.abs(emb["r"]) > r_thresh, :]
            emb["Source"] = "embeddings"
            text["Source"] = "Baseline"
            df = pd.concat([emb, text.loc[emb.index.values,:]], axis = 0)
            plot_bars(df, str(which) + ": Top Associations", "/home/zacharyl/Desktop/" + str(which) + ".png")

def do_comparison():
    embeddings_one = pd.read_csv(
        "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/no_ft/all_no_linear.csv")
    embeddings_one_cvd = pd.read_csv(
        "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/no_ft/one_cvd_no_linear.csv")
    embeddings_one_cvd = embeddings_one_cvd.set_index("RegistrationCode")
    embeddings_one = embeddings_one.set_index("RegistrationCode")
    embeddings_one["Cohort"] = "Healthy"
    embeddings_one_cvd["Cohort"] = "Sick"
    emb_all = pd.concat([embeddings_one, embeddings_one_cvd])
    res = umap.UMAP(metric = "cosine", n_neighbors = 100, min_dist = 0, spread  = 4, init = "pca").fit_transform(emb_all.loc[:, list(map(lambda x: str(x), list(range(192))))].apply(lambda x: (x - x.mean())/ x.std()))
    res = pd.DataFrame(res)
    res["RegistrationCode"] = emb_all.index
    res = res.set_index("RegistrationCode")
    res["Cohort"] = emb_all["Cohort"]
    sns.scatterplot(x=res.iloc[:, 0], y=res.iloc[:, 1], hue=res["Cohort"], alpha = 0.6)
    plt.show() ##home segal lab

if __name__ == "__main__":
    np.random.seed(0)
    os.chdir("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/qp/")
    sethandlers()
    to_pred = ["CGM", "RNA", "DiseaseDiagnoses", "Sleep", "Retina", "GutMB", "Nightingale Metabolomics"]
    baseDatas = ["embeddings", "ECGText"]
    fileName = "no_ft/one_no_linear.csv"
    splits_path =  "/net/mraid20/export/jasmine/zach/cross_modal/splits/"
    files = list(pd.read_csv(splits_path + "files.csv")["0"])
    do_gender_strat_pred(to_pred, baseDatas, fileName)
