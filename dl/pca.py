import os

import pandas as pd
import numpy as np
from LabData.DataLoaders.MedicalConditionLoader import MedicalConditionLoader
from LabData.DataLoaders.SubjectLoader import SubjectLoader
from sklearn.decomposition import PCA
import torch

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    emb_one = pd.read_csv(
        "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/one_with_linear.csv")
    emb_one = emb_one.set_index(["RegistrationCode", "Date"])
    subs = SubjectLoader().get_data(study_ids=["10K"]).df
    subs = subs.reset_index(["Date"], True).dropna(subset=["gender"])
    subs = subs.loc[~subs.index.duplicated(), :]
    emb_w_gender = emb_one.merge(subs, how="inner", left_index=True, right_index=True)[
        list(emb_one.columns) + ["gender"]]
    emb_men = emb_w_gender.loc[emb_w_gender["gender"] == 1, :]  ##gender is 1 if man, 0 o.w
    emb_women = emb_w_gender.loc[emb_w_gender["gender"] == 0, :]  ##gender is 1 if man, 0 o.w
    emb_men = emb_men.drop(["gender"], axis=1)
    emb_women = emb_women.drop(["gender"], axis=1)
    pca_men = pd.DataFrame(PCA(n_components=35).fit_transform(emb_men))
    pca_women = pd.DataFrame(PCA(n_components=35).fit_transform(emb_women))
    pca_men["RegistrationCode"] = emb_men.index.get_level_values(0)
    pca_women["RegistrationCode"] = emb_women.index.get_level_values(0)
    pca_men = pca_men.set_index("RegistrationCode")
    pca_women = pca_women.set_index("RegistrationCode")
    ##check for corr with age
    ##check for assocs with diseases
    b = MedicalConditionLoader().get_data(study_ids=["10K"]).df.reset_index()
    counts = emb_one.reset_index("Date", drop=True).merge(b, left_index=True, right_on="RegistrationCode",
                                                          how="inner").medical_condition.value_counts()
    for code in list(counts.loc[counts > 2].index):
        temp = b.loc[b.medical_condition == code, :].set_index("RegistrationCode")
        temp["has"] = 1
        temp = pd.concat([temp, pd.DataFrame(
            {"RegistrationCode": list(set(pca_women.index.get_level_values(0)) - set(temp.index.values)),
             "has": 0}).set_index("RegistrationCode")])
        e_temp = pca_men.merge(temp, left_index=True, right_index=True)
        maybes = np.abs(e_temp.corr()["has"])
        maybes = list(maybes[maybes > 0.1].index)
        maybes = [x for x in maybes if x != "has"]
        if len(maybes) > 0:
            print(code)
            print(maybes)