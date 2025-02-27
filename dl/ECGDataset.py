import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from LabData.DataLoaders.ECGTextLoader import ECGTextLoader
import torch
import os
from datetime import datetime

class ecg_dataset_pheno(Dataset):
    def __init__(self, all_samples_ecg, fmap):
        self.phenopath = "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/Pheno/stas/ecg/waveform/clean/long_format/"
        self.all_samples = all_samples_ecg
        self.fmap = fmap

    def __len__(self):
        return len(self.files)

    def __getitem__(self, tenk_id, date, window_length):
        day = str(date.day)
        month = str(date.month)
        year = str(date.year)
        if len(day) == 1:
            day = "0" + day
        if len(month) == 1:
            month = "0" + month
        file_name = "ecg__10k__" + tenk_id.split("10K_")[-1] + "__" + year + month + day + ".parquet"
        temp = pd.read_parquet(self.phenopath+file_name)
        temp = temp.reset_index(list(filter(lambda x: x != "source", temp.index.names)), drop=True)
        temp = temp.loc[['MDC_ECG_LEAD_I_clean', ##force a specific order
                         'MDC_ECG_LEAD_II_clean',
                         'MDC_ECG_LEAD_III_clean',
                         'MDC_ECG_LEAD_aVR_clean',
                         'MDC_ECG_LEAD_aVL_clean',
                         'MDC_ECG_LEAD_aVF_clean',
                         'MDC_ECG_LEAD_V1_clean',
                         'MDC_ECG_LEAD_V2_clean',
                         'MDC_ECG_LEAD_V3_clean',
                         'MDC_ECG_LEAD_V4_clean',
                         'MDC_ECG_LEAD_V5_clean',
                         'MDC_ECG_LEAD_V6_clean'], :]
        temp = temp.to_numpy().reshape(12, 10000)
        temp = torch.Tensor(temp)
        return temp

    def make_date_obj(self, date_string):
        year = date_string[0:4]
        month = date_string[4:6]
        day = date_string[6:8]
        return datetime.strptime(year + "-" + month + "-" + day, '%Y-%m-%d').date()

    def get_date(self, tenk_id):
        matching_ecg_date_strings = self.fmap[tenk_id]
        matching_ecg_dates = [self.make_date_obj(x) for x in matching_ecg_date_strings]
        return list(np.sort(matching_ecg_dates))

def update_nastya_cache():
    basepath = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/ecg'
    g = pd.read_csv(basepath + "/ecg4.csv")
    all = [f for f in os.listdir(basepath + "/xml/")]

    files_simple = [basepath + "/xml/" + f for f in all if f.endswith(".xml")]
    ids_simple = list(map(lambda x: x.split(basepath + "/")[-1].split("_")[0], files_simple))
    dates_simple = list(map(lambda x: x.split("_")[-1].split(".")[0], files_simple))
    dict_simple = dict(zip(list(zip(ids_simple, dates_simple)), files_simple))


    subdirs = [f for f in all if not f.endswith(".xml")]
    keys = list(zip([f for f in subdirs if not f.endswith(".xml")], [os.listdir(basepath +"/xml/"+f) for f in subdirs if not f.endswith(".xml")],list(map(lambda x: x[0].split(".xml")[0], [os.listdir(basepath +"/xml/"+f+"/"+g) for f in subdirs for g in os.listdir(basepath +"/xml/"+ f) if not f.endswith(".xml")]))))
    vals = list(map(lambda x: basepath + "/xml/" + x[0] + "/" +x[1][0] + "/" + x[2] +".xml", keys))

    dict_complex = dict(zip(list(map(lambda key: (key[0], key[2]), keys)),vals))
    dict_complex.update(dict_simple)

    s = {}
    for f,g in dict_complex.items():
        with open(g, "r") as t:
            s[f] = t.read()