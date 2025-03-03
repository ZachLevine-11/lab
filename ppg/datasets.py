import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import os
from datetime import datetime
from LabData.DataLoaders.ECGTextLoader import ECGTextLoader
from LabData.DataLoaders.PRSLoader import PRSLoader
from LabData.DataLoaders.RetinaScanLoader import RetinaScanLoader
from LabData.DataLoaders.BloodTestsLoader import BloodTestsLoader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
from LabData.DataLoaders.UltrasoundLoader import UltrasoundLoader
from LabData.DataLoaders.ABILoader import ABILoader
from LabData.DataLoaders.DEXALoader import DEXALoader
from pyarrow.parquet import ParquetFile
import pyarrow as pa

min_ecg_val = -0.4
max_ecg_val = 1.2

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

class sleep_dataset_pheno(Dataset):
    def __init__(self, fmap):
        self.phenopath = "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/Pheno/v1.2/sleep/timeseries/"
        self.subdir = "night=1/source=pat_infra/"
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
        subject_dir_name = "sleep__10k__" + tenk_id.split("10K_")[-1] + "__" + year + month + day + "/"
        pf = ParquetFile(self.phenopath + subject_dir_name + self.subdir + os.listdir(self.phenopath + subject_dir_name + self.subdir)[0])
        first_n_rows = next(pf.iter_batches(batch_size=window_length))
        df = pa.Table.from_batches([first_n_rows]).to_pandas()
        return torch.Tensor(df.reset_index(list(filter(lambda x: x != "source", df.index.names)), drop=True).to_numpy()).reshape(1, -1)

    def make_date_obj(self, date_string):
        year = date_string[0:4]
        month = date_string[4:6]
        day = date_string[6:8]
        return datetime.strptime(year + "-" + month + "-" + day, '%Y-%m-%d').date()

    def get_date(self, tenk_id):
        matching_ecg_date_strings = self.fmap[tenk_id]
        matching_ecg_dates = [self.make_date_obj(x) for x in matching_ecg_date_strings]
        return list(np.sort(matching_ecg_dates))

class paired_dataset_bidmc(Dataset):
    def __init__(self):
        self.base_dir = "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/bidmc-ppg-and-respiration-dataset-1.0.0/bidmc_csv/"

    def __len__(self):
        return 53

    def __getitem__(self, sample_id_int):
        the_id =  str(sample_id_int)
        if len(the_id) == 1:
            the_id = "0" + the_id
        sample_signals = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/bidmc-ppg-and-respiration-dataset-1.0.0/bidmc_csv/" + "bidmc_" + the_id + "_Signals.csv")[[" II", " PLETH"]]
        sample_ecg = torch.Tensor(sample_signals[" II"].to_numpy())
        sample_ppg = torch.Tensor(sample_signals[" PLETH"].to_numpy())
        return sample_ecg, sample_ppg

class tenk_tabular_dataset(Dataset):
    def __init__(self, dataloader):
        thresh_dict = {BodyMeasuresLoader: (None, None),
                       UltrasoundLoader: (None, None),
                       ABILoader: (None, None),
                       DEXALoader: (None, None),
                       PRSLoader: (None, 500)}
        if dataloader not in ["RNA", PRSLoader, RetinaScanLoader, "iglu"]:
            if dataloader != UltrasoundLoader:
                self.df = dataloader().get_data(research_stage = "baseline", study_ids = list(range(100)) + list(range(1000, 1011, 1))).df
            else: ##doesn't work for some reason
                self.df = dataloader().get_data(study_ids = list(range(100)) + list(range(1000, 1011, 1))).df
            self.df = self.df.reset_index().set_index(["RegistrationCode"])
            self.df = self.df.sort_values(by='Date')
            self.df = self.df.loc[~self.df.index.duplicated(keep = "first"),:]
            self.df = self.df.dropna(axis = 1, thresh = (len(self.df)*(3/4))//1).iloc[:, 0:thresh_dict[dataloader][1]]
        elif dataloader == "RNA":
            raise ValueError
        elif dataloader == PRSLoader:
            self.p_meta = PRSLoader().get_data().df_columns_metadata
            self.df = PRSLoader().get_data().df
            self.df = self.df.loc[:, list(set(self.p_meta.loc[self.p_meta.h2_confidence == "high", :].index.values).intersection(self.df.columns))]
            self.df = self.df.loc[:, ~self.df.columns.duplicated()]
            self.df = self.df.iloc[:, 0:thresh_dict[dataloader][1]]
        elif dataloader == ECGTextLoader:
            raise ValueError
        elif dataloader == "iglu":
            self.df = pd.read_csv('/net/mraid20/export/genie/LabData/Data/10K/RiskFactors/cgm_features.csv', index_col=0)
        elif dataloader == RetinaScanLoader:
            self.df = RetinaScanLoader().get_data(research_stage = "baseline", study_ids = list(range(100)) + list(range(1000, 1011, 1))).df.copy().unstack()  ##Don't groupby reg here because the extra index (right vs left eye) will mess things up
            self.df.columns = list(map(lambda col_tuple: str(col_tuple[0]) + "_" + str(col_tuple[1]), self.df.columns))
            self.df = self.df.reset_index().set_index("RegistrationCode")
            self.df = self.df.drop(['participant_id_l_eye', "participant_id_r_eye"], axis = 1)
        if "Date" in self.df.columns:
            self.df = self.df.sort_values(by='Date')
            self.df = self.df.drop(["Date"], axis = 1)
        self.df = self.df.loc[~self.df.index.duplicated(keep="first"), :]
        if dataloader == BodyMeasuresLoader:
            self.df = self.df.loc[:, self.df.dtypes[self.df.dtypes == "float64"].index]
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        self.df = self.df.select_dtypes(include=numerics)
        self.dataloader = dataloader

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, tenk_id):
        return torch.nan_to_num(torch.Tensor(self.df.loc[tenk_id].to_numpy().reshape(1, -1)))


min_ecg_val = -0.4
max_ecg_val = 1.2

def min_max_scaling(data, dim, a=min_ecg_val, b=max_ecg_val, how="minmax"):
    if how == "minmax":
        min_val = data.min(axis=dim, keepdim=True)[0]
        max_val = data.max(axis=dim, keepdim=True)[0]
        scaled_data = (data - min_val) / (max_val - min_val) * (b - a) + a
    else:
        scaled_data = (data - data.mean(dim=dim, keepdim=True)) / data.std(dim=dim, keepdim=True)
    return scaled_data


def preprocess(signal, length):
    """Slice out a specific window from the signal for demonstration."""
    start = 1000
    end = 1000 + length
    # signal shape is (12, 10000) typically.
    # We'll return shape (length, 12) below.
    return signal[:, start:end].T


class ECGTabularDataset(Dataset):
    """
    A dataset that pulls:
      - ECG data from `ds_ecg` (ecg_dataset_pheno)
      - Tabular data from a single domain (e.g., 10K tabular dataset)
    """

    def __init__(self, ids_list, ds_ecg, ds_tabular, length_window, length_sim):
        """
        Args:
            ids_list (list): List of TenK IDs for train/eval/test split.
            ds_ecg (ecg_dataset_pheno): The ECG dataset object.
            ds_tabular (tenk_tabular_dataset): The single-domain tabular dataset.
            length_window (int): Window size to slice from raw ECG (e.g. 1000).
            length_sim (int): Final length to interpolate to (e.g. 300).
        """
        super().__init__()
        self.ids_list = ids_list
        self.ds_ecg = ds_ecg
        self.ds_tabular = ds_tabular
        self.length_window = length_window
        self.length_sim = length_sim

    def __len__(self):
        return len(self.ids_list)

    def __getitem__(self, index):
        """
        Returns a dictionary with:
            {
              "tenk_id": str,
              "ecg_no_interp": (12, length_window) tensor,
              "ecg_true_interp": (12, length_sim) tensor,
              "tabular_feats": (N_features,) tensor
            }
        or None if data not found.
        """
        tenk_id = self.ids_list[index]
        try:
            # 1) Grab ECG date(s)
            dates = self.ds_ecg.get_date(tenk_id)
            if not dates:
                return None
            # Use the first date or random date  up to you
            date = dates[0]
            # 2) Load raw ECG -> shape: (12, 10,000) typically
            ecg_signal_12 = self.ds_ecg.__getitem__(tenk_id, date, self.length_window)

            # 3) Slice down to shape (length_window, 12)
            ecg_slice = preprocess(ecg_signal_12, self.length_window)  # shape: (length_window, 12)
            ecg_slice = torch.nan_to_num(ecg_slice)  # remove NaNs
            ecg_slice = ecg_slice.T  # -> shape: (12, length_window)

            # 4) Interpolate to length_sim
            #   for F.interpolate we need shape (B, C, L), so wrap ecg_slice in batch dim
            ecg_slice_b = ecg_slice.unsqueeze(0)  # (1, 12, length_window)
            ecg_true_interp = F.interpolate(
                ecg_slice_b.unsqueeze(-1),  # (1,12,length_window,1)
                size=(self.length_sim, 1),
                mode="bilinear",
                align_corners=False
            ).squeeze(-1)  # shape -> (1,12,length_sim)

            ecg_no_interp = min_max_scaling(ecg_slice_b, dim=-1)  # (1, 12, length_window)
            ecg_true_interp = min_max_scaling(ecg_true_interp, dim=-1)  # (1,12,length_sim)

            ecg_no_interp = ecg_no_interp[0]  # (12, length_window)
            ecg_true_interp = ecg_true_interp[0]  # (12, length_sim)

            # 5) Fetch tabular data
            tabular_feats = self.ds_tabular.__getitem__(tenk_id)  # shape: (1, N_features)
            tabular_feats = tabular_feats.squeeze()  # -> shape: (N_features,)

        except (FileNotFoundError, KeyError, IndexError):
            return None

        return {
            "tenk_id": tenk_id,
            "ecg_no_interp": ecg_no_interp,  # (12, length_window)
            "ecg_true_interp": ecg_true_interp,  # (12, length_sim)
            "tabular_feats": tabular_feats  # (N_features,)
        }


def ecg_tabular_collate_fn(batch):
    """
    Custom collate function that:
    - filters out None samples
    - stacks everything into a single batch
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    tenk_ids = [sample["tenk_id"] for sample in batch]
    ecg_no_interp = torch.stack([sample["ecg_no_interp"] for sample in batch], dim=0)
    ecg_true_interp = torch.stack([sample["ecg_true_interp"] for sample in batch], dim=0)
    tabular_feats = torch.stack([sample["tabular_feats"] for sample in batch], dim=0)

    return {
        "tenk_id": tenk_ids,
        "ecg_no_interp": ecg_no_interp,
        "ecg_true_interp": ecg_true_interp,
        "tabular_feats": tabular_feats
    }


def get_domain_dataloaders(
        train_people,
        eval_people,
        ds_ecg,
        tabular_datasets,  # dict {domain_label: tenk_tabular_dataset}
        config
):
    """
    Returns:
      train_loaders = { domain_label: DataLoader }
      eval_loaders  = { domain_label: DataLoader }
    """
    train_loaders = {}
    eval_loaders = {}

    for domain_label, ds_tabular in tabular_datasets.items():
        # Build the train dataset & loader
        try:
            train_dataset = ECGTabularDataset(
                ids_list=train_people,
                ds_ecg=ds_ecg,
                ds_tabular=ds_tabular,
                length_window=config.length_window,
                length_sim=config.length_sim
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=6,  # parallel workers
                collate_fn=ecg_tabular_collate_fn,
                drop_last=False
            )
        except AttributeError:
            train_dataset = ECGTabularDataset(
                ids_list=train_people,
                ds_ecg=ds_ecg,
                ds_tabular=ds_tabular,
                length_window=config["length_window"],
                length_sim=config["length_sim"] ##outside of wandb runs, this is a simple dict
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=6,  # parallel workers
                collate_fn=ecg_tabular_collate_fn,
                drop_last=False
            )

        train_loaders[domain_label] = train_loader
        try:
        # Build the eval dataset & loader
            eval_dataset = ECGTabularDataset(
                ids_list=eval_people,
                ds_ecg=ds_ecg,
                ds_tabular=ds_tabular,
                length_window=config.length_window,
                length_sim=config.length_sim
            )
            eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=6,
            collate_fn=ecg_tabular_collate_fn,
            drop_last=False
            )
        except AttributeError:
            eval_dataset = ECGTabularDataset(
                ids_list=eval_people,
                ds_ecg=ds_ecg,
                ds_tabular=ds_tabular,
                length_window=config["length_window"],
                length_sim=config["length_sim"]
                )
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=6,
                collate_fn=ecg_tabular_collate_fn,
                drop_last=False
            )
        eval_loaders[domain_label] = eval_loader

    return train_loaders, eval_loaders
