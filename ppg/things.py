import torch
from datasets import ecg_dataset_pheno, sleep_dataset_pheno, baseline_dataset, paired_dataset_bidmc
import os
import numpy as np
import torch.nn as nn
from models import PPGtoECG
from matplotlib import pyplot as plt
import pandas as pd
from tslearn.metrics import SoftDTWLossPyTorch
from LabData.DataLoaders.SubjectLoader import SubjectLoader
from LabData.DataLoaders.ECGTextLoader import ECGTextLoader
import statsmodels.api as sm

def make_age_df(emb_df):
    subs = SubjectLoader().get_data(study_ids=list(range(100)) + list(range(1000, 1011, 1))).df
    ecgtext =  ECGTextLoader().get_data(study_ids=list(range(100)) + list(range(1000, 1011, 1))).df.reset_index().set_index(["RegistrationCode"])
    ecgtext = ecgtext.loc[~ecgtext.index.duplicated(keep = "first"),:]
    emb_df = emb_df.merge(ecgtext["Date"], left_index = True, right_index = True).set_index(["Date"], append = True)
    subs = subs.reset_index(["Date"], drop=True).dropna(subset=["yob", "month_of_birth"])
    subs_merged_raw = emb_df.merge(subs, left_index=True, right_index=True)
    age_computation = subs_merged_raw.loc[:, ["month_of_birth", "yob"]]
    age_computation.columns = ["month", "year"]
    age_computation["day"] = 15
    subs_merged_raw["birthdate"] = pd.to_datetime(age_computation)
    subs_merged_raw = subs_merged_raw.reset_index(["Date"], drop = False)
    subs_merged_raw["Date"] = pd.to_datetime(subs_merged_raw["Date"])
    age_raw = list(map(lambda x: (x[1] - x[0]).days, zip(subs_merged_raw["birthdate"], subs_merged_raw["Date"])))
    age_df_raw = pd.DataFrame(age_raw)
    age_df_raw["RegistrationCode"] = list(subs_merged_raw.index)
    age_df_raw["Date"] = list(subs_merged_raw.Date)
    age_df_raw = age_df_raw.rename({0: "age"}, axis=1)
    age_df_raw = age_df_raw.set_index(["RegistrationCode", "Date"])
    return age_df_raw

##just train off one lead of the ecg
def preprocess(signal, length):
    store = []
    cuts = [(1000, 1000 + length)]
    for cut in cuts:
        sig = signal[:, cut[0]:cut[1], ].T
        store.append(sig)
    return store

def get_available_pairs(tenk_id, ds, length):
    dates = ds.get_date(tenk_id)
    ecg_one = ds.__getitem__(tenk_id, dates[0])
    peaks = preprocess(ecg_one, length)
    return peaks

def get_available_pairs_baseline(tenk_id, ds):
    baseline_one = ds.__getitem__(tenk_id)
    return baseline_one

def batch_manager(ids_tenk, ds_ecg, ds_ppg, ds_baseline, device, length_window, num_segments = 5):
    ecgs, ppgs, baselines = [], [], [],
    endpoints = [length_window*x for x in range(60000//length_window)]
    cuts = list(zip(endpoints[:-1], endpoints[1:]))
    ##randomly order the segments to avoid bias, but keep the ecg and ppg alligned
    cuts = [cuts[i] for i in list(np.random.choice(list(range(len(cuts))), size = min(num_segments, len(cuts)), replace = False).astype(int))]
    if ds_ppg is None or ds_baseline is None:
        for id in ids_tenk:
            for low, high in cuts:
                ecg, ppg = ds_ecg.__getitem__(id)
                ecgs += [ecg[low:high]]
                ppgs += [ppg[low:high]]
        ecgs = torch.nn.utils.rnn.pad_sequence(ecgs, True).to(device)
        ppgs = torch.nn.utils.rnn.pad_sequence(ppgs, True).to(device)
        return ecgs, ppgs, None
    for i in range(len(ids_tenk)):
        try: ##some sleep directories are empty, not sure why
            ##make sure all exist before adding to batch
            a = get_available_pairs(ids_tenk[i], ds_ppg, length_window//10)
            b = get_available_pairs_baseline(ids_tenk[i], ds_baseline)
            c = get_available_pairs(ids_tenk[i], ds_ecg, length_window)
            ##
            ppgs += a
            baselines += b
            ecgs += c
        except (FileNotFoundError, KeyError):
            pass
    if len(ecgs) == 0:
        return None, None, None
    baselines = torch.nn.utils.rnn.pad_sequence(baselines, True).to(device)
    ecgs = torch.nn.utils.rnn.pad_sequence(ecgs, True).permute(0, 2, 1).to(device)
    ppgs = torch.nn.utils.rnn.pad_sequence(ppgs, True).to(device)
    return ecgs, ppgs, baselines

def train_ppg_to_ecg(model,
                    length_sim,
                    ds_ecg,
                    ds_ppg,
                    ds_baseline,
                    device,
                    people,
                    lr,
                    input = "ecg",
                    store_after = None):
    print("Using " + str(device))
    torch.cuda.empty_cache()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    train_batches = np.array_split(np.random.choice(people, replace = False, size = len(people)), 20)
    loss_fn = SoftDTWLossPyTorch(gamma = 0.1, normalize = True)
    mse_eval = nn.MSELoss()
    mae_eval = nn.L1Loss()
    i = 0
    emb_stores = {}
    model.training = True
    model.train()
    for epoch in range(100):
        for batch in train_batches:
            i += 1
            length_window = np.random.randint(low = 200, high = 1536)
            ecgs, ppgs, baselines = batch_manager(batch, ds_ecg, ds_ppg, ds_baseline, device, length_window)
            if ecgs is not None:
                ppgs, ecgs = torch.nan_to_num(ppgs), torch.nan_to_num(ecgs) #do first to keep the nans out of all computations
                ecg_true_interp = nn.functional.interpolate(ecgs.reshape(ecgs.shape[0], 1, -1, 1),
                                                            size=(length_sim, 1)).reshape(-1, length_sim)
                ecg_true_interp = (ecg_true_interp-ecg_true_interp.mean(dim = 1, keepdim = True))/ecg_true_interp.std(dim = 1, keepdim = True)
                ppgs_true_interp = nn.functional.interpolate(ppgs.reshape(ppgs.shape[0], 1, -1, 1),
                                                            size=(length_sim, 1)).reshape(-1, length_sim)
                ppgs_true_interp = (ppgs_true_interp - ppgs_true_interp.mean(dim=1, keepdim=True)) / ppgs_true_interp.std(dim=1,                                                                                                keepdim=True)
                if input == "ECG" or input == "ecg":
                    if store_after is None:
                        [output, nonseq_q] = model(ecg_true_interp.reshape(ecg_true_interp.shape[0], -1, 1), baselines, length_sim)
                    else:
                        if i >= store_after:
                            model.training = False
                            model.eval()
                            emb_stores.update(zip(batch, model(ecg_true_interp.reshape(ecg_true_interp.shape[0], -1, 1), baselines, length_sim, onlyz = True)))
                            continue
                        else:
                            [output, nonseq_q] = model(ecg_true_interp.reshape(ecg_true_interp.shape[0], -1, 1), baselines,
                                                       length_sim)
                elif input == "PPG" or input == "ppg":
                    if store_after is None:
                        [output, nonseq_q] = model(ppgs_true_interp.reshape(ppgs_true_interp.shape[0], -1, 1), baselines, length_sim)
                    else:
                        if i >= store_after:
                            model.training = False
                            model.eval()
                            emb_stores.update(zip(batch, model(ppgs_true_interp.reshape(ppgs_true_interp.shape[0], -1, 1),
                                                             baselines, length_sim, onlyz=True)))
                            continue
                        else:
                            [output, nonseq_q] = model(ppgs_true_interp.reshape(ppgs_true_interp.shape[0], -1, 1), baselines,
                                                       length_sim)
                else:
                    if store_after is None:
                        [output, nonseq_q] = model(None, baselines, length_sim)
                    else:
                        if i >= store_after:
                            model.training = False
                            model.eval()
                            emb_stores.update(zip(batch, model(None, baselines, length_sim, onlyz = True)))
                            continue
                        else:
                            [output, nonseq_q] = model(None, baselines, length_sim)
                ecg_pred = output[:, :, 2].reshape(ecg_true_interp.shape)
                ecg_pred = (ecg_pred - ecg_pred.mean(dim=1, keepdim=True)) / ecg_pred.std(dim=1, keepdim=True)
                dtw = loss_fn(torch.nan_to_num(ecg_pred).reshape(ecg_pred.shape[0], 1, -1), ecg_true_interp.reshape(ecg_pred.shape[0], 1, -1)).mean()
                mse = mse_eval(torch.nan_to_num(ecg_pred).reshape(ecg_pred.shape[0], 1, -1),
                               ecg_true_interp.reshape(ecg_pred.shape[0], 1, -1))
                with torch.no_grad():
                    mae = mae_eval(torch.nan_to_num(ecg_pred).reshape(ecg_pred.shape[0], 1, -1), ecg_true_interp.reshape(ecg_pred.shape[0], 1, -1)).item()
                    r2 = torch.corrcoef(torch.stack([torch.nan_to_num(ecg_pred)[0, :], ecg_true_interp[0, :]]))[0, 1]
                    print("mae: ", mae)
                    print("RMSE: ", torch.sqrt(mse).item())
                    print("r2: ", r2)
                print("Time DTW: ", dtw)
                loss = dtw
                loss.backward()
                if input is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1, norm_type = 2.0) ##help deal with exploding gradients
                optimizer.step()
                optimizer.zero_grad()
                print("loss: ", loss.item())
                if i % 1 == 0:
                    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)  # 3 rows, 1 column

                    axs[0].plot(list(range(length_sim)), ecg_true_interp[0, :].squeeze().cpu(), c = "black")
                    axs[0].set_title("True ECG signal")
                    axs[0].set_ylabel("Voltage (mV)")
                    axs[1].set_xlabel('x')
                    axs[1].set_xlabel('time (t)')

                    axs[1].plot(list(range(length_sim)), ppgs_true_interp[0, :].squeeze().cpu(), c = "blue",)
                    axs[1].set_title('PPG Signal')
                    axs[1].set_xlabel('time (t)')

                    axs[2].plot(list(range(length_sim)), ecg_pred[0, :].detach().cpu(), c = "green")
                    axs[2].set_title("Predicted ECG Signal")
                    axs[2].set_ylabel("Voltage (mV)")
                    axs[1].set_xlabel('time (t)')
                    plt.tight_layout()
                    plt.show()
    if store_after is not None:
        embs = pd.DataFrame(torch.stack(list(emb_stores.values())).detach().cpu().numpy().astype(np.float64)).astype("float64")
        embs.index = list(emb_stores.keys())
        embs.index.name = "RegistrationCode"
    else:
        embs = None
    return model, embs


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_bidmc = True
    if train_bidmc:
        ds_both = paired_dataset_bidmc()
        people = list(range(1, 54))
        length_sim = 750
        model = PPGtoECG(64, proj_size = 0, num_layers=2, length_sim=length_sim, num_baseline_features=0, encoder_dropout_prob=0,
                         use_time_series=True, use_baselines=False).to(device)
        lr = 5e-5
        model, embs = train_ppg_to_ecg(model, length_sim, ds_both, None, None, device, lr = lr, people = people, input="ppg", store_after=None)
    train_hpp = False
    if train_hpp:
        ds_baseline = baseline_dataset(include_baselines=True)
        ds_ecg =  ecg_dataset_pheno(files = os.listdir("/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/Pheno/stas/ecg/waveform/clean/long_format/"))
        ds_sleep = sleep_dataset_pheno(files = os.listdir( "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/Pheno/v1.2/sleep/timeseries/"))
        people_with_all = list(set(ds_ecg.all_samples.index).intersection(set(ds_sleep.all_samples.index)).intersection(set(ds_baseline.rna.index.values)).intersection(set(ds_baseline.p.index.values)).intersection(set(ds_baseline.b.index.get_level_values(0))).intersection(set(ds_baseline.d.index.get_level_values(0))))
        torch.cuda.empty_cache()
        ##ECG-ECG
        length_sim = 1000
        model = PPGtoECG(128, 2, length_sim = length_sim, num_baseline_features=6549, encoder_dropout_prob=0, use_time_series = True, use_baselines = True).to(device)
        lr = 1e-4
        model, embs =  train_ppg_to_ecg(model, length_sim, ds_ecg, ds_sleep, ds_baseline, device, list(np.random.choice(people_with_all, 1000, replace = False)), lr = lr, input = "ppg",  store_after=200)

        age_df = make_age_df(embs).reset_index(["Date"], drop = True)
        c = embs.merge(age_df, left_index=True, right_index=True)
        c = c.loc[~c.index.duplicated(), :]
        batches = np.array_split(c.index.values, 5)
        test_preds = {}
        for test_indices in batches:
            train_indices = list(set(list(c.index)) - set(test_indices))
            j = c.iloc[:, :31].loc[train_indices,:]
            k = c.loc[train_indices, "age"].astype(float).values
            model_age = sm.OLS(k, j).fit()
            n = c.iloc[:, :31].loc[test_indices]
            test_preds.update(zip(test_indices, model_age.predict(n)))
        test_preds = pd.Series(test_preds)
        l = c.loc[:, "age"].loc[test_preds.index].astype(float)
        np.corrcoef(l, test_preds)[0,1]**2
