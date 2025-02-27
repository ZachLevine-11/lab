import pandas as pd
import numpy as np
from LabData.DataLoaders.MedicalConditionLoader import MedicalConditionLoader
from LabData.DataLoaders.ECGTextLoader import ECGTextLoader
from wandb_main import test_single_cond, predict_binary_phenotype, pred_age
from LabData.DataLoaders.SubjectLoader import SubjectLoader
import torch

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    splits_path =  "/net/mraid20/export/jasmine/zach/cross_modal/splits/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_people = list(pd.read_csv(splits_path + "train.csv")["0"])
    test_people = list(pd.read_csv(splits_path + "test.csv")["0"])
    validation_people = list(pd.read_csv(splits_path + "validation.csv")["0"])
    embeddings_one = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/no_ft/one_no_linear.csv").set_index(["RegistrationCode", "Date"])

    ecg_with_date = ECGTextLoader().get_data(study_ids=["10K"]).df.dropna(1).reset_index(["Date"], drop=False)
    ecg_with_date = ecg_with_date.loc[embeddings_one.index.get_level_values(0), :]
    ecg_with_date = ecg_with_date.loc[~ecg_with_date.index.get_level_values(0).duplicated(keep="first")]
    # some people for whom we only have the pheno data from one apt may have more than one apt in ECGText, causing duplicates here
    ##pick the first appointment for these people, which would be the one we have the raw data for
    ecg_with_date = ecg_with_date.reset_index().set_index(["RegistrationCode", "Date"])
    ecg_no_date = ecg_with_date.reset_index(["Date"], drop=True)
    b = MedicalConditionLoader().get_data(study_ids=["10K"]).df.reset_index()
    ecg_with_date_col = ecg_with_date.reset_index(["Date"], drop=False)
    counts = embeddings_one.reset_index("Date", drop=True).merge(b, left_index=True, right_on="RegistrationCode",
                                                                 how="inner").medical_condition.value_counts()
    ecg_with_date_col["ecg_collection_date"] = pd.to_datetime(
        ecg_with_date_col["Date"])  ##so its clear we are using the right date to compute age
    test_disease = False
    if test_disease:
        diseases = list(counts.index[counts > 30])
        res_hrv_xgb = {}
        res_hrv_mlp = {}
        res_hrv_elasticnet = {}
        for disease in diseases:
            res_disease_ecg_baseline_xgb = test_single_cond(disease, ecg_no_date, b, baseData="HRV", how="xgboost")
            res_disease_ecg_baseline_dl = test_single_cond(disease, ecg_no_date, b, baseData="HRV", how="dl")
            res_disease_ecg_baseline_elasticnet = test_single_cond(disease, ecg_no_date, b, baseData="HRV", how="linear")
            res_hrv_mlp[disease] = {"AUC_PRC": res_disease_ecg_baseline_dl[3][0],
                                    "AUC_ROC": res_disease_ecg_baseline_dl[3][1],
                                    "F1": res_disease_ecg_baseline_dl[4],
                                    "Sensitivity": res_disease_ecg_baseline_dl[5],
                                    "Specificity": res_disease_ecg_baseline_dl[6],
                                    "Cases Ratio": res_disease_ecg_baseline_dl[7]}
            res_hrv_xgb[disease] = {"AUC_PRC": res_disease_ecg_baseline_xgb[3][0],
                                    "AUC_ROC": res_disease_ecg_baseline_xgb[3][1],
                                    "F1": res_disease_ecg_baseline_xgb[4],
                                    "Sensitivity": res_disease_ecg_baseline_xgb[5],
                                    "Specificity": res_disease_ecg_baseline_xgb[6],
                                    "Cases Ratio": res_disease_ecg_baseline_xgb[7]}
            res_hrv_elasticnet[disease] = {"AUC_PRC": res_disease_ecg_baseline_elasticnet[3][0],
                                           "AUC_ROC": res_disease_ecg_baseline_elasticnet[3][1],
                                           "F1": res_disease_ecg_baseline_elasticnet[4],
                                           "Sensitivity": res_disease_ecg_baseline_elasticnet[5],
                                           "Specificity": res_disease_ecg_baseline_elasticnet[6],
                                           "Cases Ratio": res_disease_ecg_baseline_elasticnet[7]}
        res_hrv_xgb = pd.DataFrame(res_hrv_xgb).T.dropna()
        res_hrv_xgb.to_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/disease_pred/xgboost/res_hrv.csv")
        res_hrv_mlp = pd.DataFrame(res_hrv_mlp).T.dropna()
        res_hrv_mlp.to_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/disease_pred/mlp/res_hrv.csv")
        res_hrv_elasticnet = pd.DataFrame(res_hrv_elasticnet).T.dropna()
        res_hrv_elasticnet.to_csv(
            "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/disease_pred/linear/res_hrv.csv")

    do_age_hrv = True
    if do_age_hrv:
        subs = SubjectLoader().get_data(study_ids=["10K"]).df
        subs = subs.reset_index(["Date"], drop=True).dropna(subset=["yob", "month_of_birth"])
        subs_merged = subs.merge(ecg_with_date_col, left_index=True, right_index=True)
        age_computation = subs_merged.loc[:, ["month_of_birth", "yob"]]
        age_computation.columns = ["month", "year"]
        age_computation["day"] = 15
        subs_merged["birthdate"] = pd.to_datetime(age_computation)
        age_ecgtext = list(
            map(lambda x: (x[1] - x[0]).days, zip(subs_merged["birthdate"], ecg_with_date_col["ecg_collection_date"])))
        age_df_ecgtext = pd.DataFrame(age_ecgtext)
        age_df_ecgtext["RegistrationCode"] = list(subs_merged.index)
        age_df_ecgtext["ecg_collection_date"] = list(subs_merged["ecg_collection_date"])
        age_df_ecgtext = age_df_ecgtext.rename({0: "age", "ecg_collection_date": "Date"}, axis=1)
        age_df_ecgtext = age_df_ecgtext.set_index(["RegistrationCode", "Date"])
        pred_hrv, true_hrv, r_hrv, = pred_age(ecg_with_date, age_df_ecgtext, "xgboost", train_ids=train_people, test_ids = test_people, validation_ids=validation_people)
    do_gender = False
    if do_gender:
        subs = SubjectLoader().get_data(study_ids=["10K"]).df
        subs = subs.reset_index(["Date"], True).dropna(subset  = ["yob", "month_of_birth"])
        genderSeries = pd.DataFrame(subs.merge(ecg_no_date, left_index = True, right_index = True, how = "inner")["gender"]).dropna().astype(float)
        genderSeries.columns = ["has"]
        gender_pred_hrv = predict_binary_phenotype(ecg_no_date, genderSeries, baseData  = "DL", device = device, how="linear")