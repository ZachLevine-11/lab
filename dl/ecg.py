import numpy as np
import pandas as pd
import os
import torch
from torch import optim
from tcn import both
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn as nn
from torch.nn import functional as F
import wandb
import pytorch_warmup as warmup

from scipy.stats import pearsonr

def guy_metric_across(d, k):
    correct = 0
    for i in range(d.shape[1]):
        ind = np.argpartition(d[:, i], -k)[-k:]
        if i in ind:
            correct += 1
    return correct/d.shape[0]

def guy_metric_within(d, k):
    correct = 0
    for i in range(d.shape[1]-1):
        if i % 2  == 0:
            other_window_int_pos = i+1
        else:
            other_window_int_pos = i-1
        ind = np.argpartition(d[:, i], -(k+1))[-(k+1):]
        if other_window_int_pos in ind:
            correct += 1
    return correct/d.shape[0]

def big_guy_metric(emb_matrix_appointment_one, emb_matrix_appointment_two, type, ids_one, ids_two):
    e1 = emb_matrix_appointment_one.cpu().detach().numpy()
    e2 = emb_matrix_appointment_two.cpu().detach().numpy()
    ##for cross apt metric
    e1_df = pd.DataFrame(e1)
    e2_df = pd.DataFrame(e2)
    e1_df["id"], e2_df["id"] = ids_one, ids_two
    e1_mean, e2_mean = e1_df.set_index("id").groupby("id").mean(), e2_df.set_index("id").groupby("id").mean()
    e1_only_has_second = e1_mean.loc[list(e2_mean.index.values),:]
    corrmat_across_apt = e1_only_has_second.to_numpy() @ e2_mean.to_numpy().T
    corrmat_across_apt = corrmat_across_apt / corrmat_across_apt.max()
    print("Number of across appointment people: ", corrmat_across_apt.shape)
    ##for within apt one metric
    corrmat_within_apt_one = e1 @ e1.T
    corrmat_within_apt_one = corrmat_within_apt_one / corrmat_within_apt_one.max()
    print("Number of within first appointment windows: ", corrmat_within_apt_one.shape)
    ##for within apt two metric
    corrmat_within_apt_two = e2 @ e2.T
    corrmat_within_apt_two = corrmat_within_apt_two / corrmat_within_apt_two.max()
    print("Number of within second appointment windows: ", corrmat_within_apt_two.shape)
    returnval = 0
    for k in [1,5, 10]:
        if k < corrmat_across_apt.shape[0]:
            guy_metric_ans_across_apt = guy_metric_across(corrmat_across_apt, k)
            wandb.log({"Across_Apt_Multiplier_" + type + "_" + str(k) : guy_metric_ans_across_apt/(k/corrmat_across_apt.shape[0])})
            wandb.log({"Across_Apt_Acc_" + type + "_" + str(k) : guy_metric_ans_across_apt})
            guy_metric_ans_within_apt_one = guy_metric_within(corrmat_within_apt_one, k)
            wandb.log({"Within_First_Apt_Multiplier_" + type + "_" + str(k) : guy_metric_ans_within_apt_one/(k/corrmat_within_apt_one.shape[0])})
            wandb.log({"Within_First_Apt_Acc_" + type + "_" + str(k) : guy_metric_ans_within_apt_one})
            guy_metric_ans_within_apt_two = guy_metric_within(corrmat_within_apt_two, k)
            wandb.log({"Within_Second_Apt_Multiplier_" + type + "_" + str(k) : guy_metric_ans_within_apt_two/(k/corrmat_within_apt_two.shape[0])})
            wandb.log({"Within_Second_Apt_Acc_" + type + "_" + str(k): guy_metric_ans_within_apt_two})
            if k == 1:
                returnval = guy_metric_ans_across_apt
    return returnval

def fasterContrast(emb_matrix_appointment_one, emb_matrix_appointment_two, temp, device, ids_one, ids_two):
    first_window_embs = emb_matrix_appointment_one[0::2, :]
    second_window_embs =  emb_matrix_appointment_one[1::2, :]
    logits_only_one = (first_window_embs @ second_window_embs.t()) * torch.exp(temp)
    loss_i_only_one = F.cross_entropy(logits_only_one, torch.arange(logits_only_one.shape[0], device = device))
    loss_t_only_one = F.cross_entropy(logits_only_one.T, torch.arange(logits_only_one.shape[0], device = device))
    dist_only_one = (loss_i_only_one + loss_t_only_one) / 2
    if emb_matrix_appointment_two is None:
        return dist_only_one
    else:
        first_window_embs_two = emb_matrix_appointment_two[0::2, :]
        second_window_embs_two = emb_matrix_appointment_two[1::2, :]
        logits_only_two = (first_window_embs_two @ second_window_embs_two.t()) * torch.exp(temp)
        loss_i_only_two = F.cross_entropy(logits_only_two, torch.arange(logits_only_two.shape[0], device=device))
        loss_t_only_two = F.cross_entropy(logits_only_two.T, torch.arange(logits_only_two.shape[0], device=device))
        dist_only_two = (loss_i_only_two + loss_t_only_two) / 2
        no_apt_two = list(set(ids_one) - set(ids_two))
        ilocs_one_and_two = [i for i in range(len(ids_one)) if ids_one[i] not in no_apt_two]
        ids_one_and_two = [ids_one[i] for i in range(len(ids_one)) if ids_one[i] not in no_apt_two]
        assert ids_one_and_two == ids_two
        logits_comparing_appointments = (F.avg_pool1d(emb_matrix_appointment_one[ilocs_one_and_two,:].permute(1,0), kernel_size = 2).permute(1,0) @ F.avg_pool1d(emb_matrix_appointment_two.permute(1,0), kernel_size = 2).permute(1,0).t()) * torch.exp(temp)
        loss_i_comparing_appointments =  F.cross_entropy(logits_comparing_appointments,torch.arange(logits_comparing_appointments.shape[0], device=device))
        loss_t_comparing_appointments = F.cross_entropy(logits_comparing_appointments.T,torch.arange(logits_comparing_appointments.shape[0], device=device))
        dist_comparing_appointments = (loss_i_comparing_appointments + loss_t_comparing_appointments)/2
        return dist_comparing_appointments + (dist_only_one + dist_only_two)/10

def normalize(x):
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)

def preprocess(signal, length):
    store = []
    cuts = [(1000, 1000 + length), (1000 + length, 1000 + 2 * length)]
    for cut in cuts:
        sig = signal[:, cut[0]:cut[1]].T
        store.append(sig)
    return store

def chunk_signal_by_peaks(signal, length, dir, remake_windows_from_scratch, tenk_id):
    store = []
    if remake_windows_from_scratch:
        store = preprocess(signal, length)
    else:
        saved_window_files = os.listdir(dir + str(tenk_id) + "/")
        for window_file in saved_window_files:
            if window_file != "":
                sig = torch.tensor(pd.read_csv(dir + str(tenk_id) + "/" + window_file).iloc[:, 1:].to_numpy().astype(np.float64)) ##drop the artificial index created when saving
                store.append(sig)
    return store

def get_available_pairs(tenk_id, ds, ones_dir, twos_dir, remake_windows_from_scratch, length):
    dates = ds.get_date(tenk_id)
    ecg_one = ds.__getitem__(tenk_id, dates[0], window_length = 4000)
    peaks_one = chunk_signal_by_peaks(ecg_one, length, ones_dir, remake_windows_from_scratch, tenk_id)
    if len(dates) > 1:
        ecg_two = ds.__getitem__(tenk_id, dates[1], window_length = 4000)
        peaks_two =  chunk_signal_by_peaks(ecg_two, length, twos_dir, remake_windows_from_scratch, tenk_id)
        min_num_windows = min(len(peaks_one), len(peaks_two))
    else:
        peaks_two = []
        min_num_windows = len(peaks_one)
    return peaks_one[0:min_num_windows], peaks_two[0:min_num_windows]

def batch_manager(ids_tenk, ds_train_ecg, device, ones_dir, twos_dir, remake_windows_from_scratch, length = 4000):
    ids_one = []
    ids_two = []
    one = []
    two = []
    for i in range(len(ids_tenk)):
        pair = get_available_pairs(ids_tenk[i], ds_train_ecg, ones_dir, twos_dir, remake_windows_from_scratch, length)
        one += pair[0]
        two += pair[1]
        ids_one += [ids_tenk[i] for j in range(len(pair[0]))]
        ids_two += [ids_tenk[i] for j in range(len(pair[1]))]
    ones = torch.nn.utils.rnn.pad_sequence(one, True).permute(0, 2, 1).to(device)
    if len(two) > 0:
        twos = torch.nn.utils.rnn.pad_sequence(two, True).permute(0, 2, 1).to(device)
    else:
        twos = None
    return ones, twos, ids_one, ids_two

class EarlyStopperContrastive:
    def __init__(self, min_loss, patience=1, min_delta=0, saveName = "checkpoint.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = min_loss
        self.saveName = saveName
        self.saved_model_path = "/net/mraid20/export/jasmine/zach/cross_modal/saved_models/"

    def early_stop(self, curr_val_loss, pretrained_ecgnet, optimizer, temp):
        if curr_val_loss < self.min_loss:
            self.min_loss = curr_val_loss
            self.counter = 0
            checkpoint = {
                'model': pretrained_ecgnet,
                'optimizer': optimizer,
                 "min_loss": curr_val_loss,
                "temp": temp}
            torch.save(checkpoint, self.saved_model_path + self.saveName)
        elif curr_val_loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class EarlyStopperContrastiveMaximize:
    def __init__(self, max_acc = 0, patience=1, min_delta=0, saveName = "checkpoint.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_acc = max_acc
        self.saveName = saveName
        self.saved_model_path = "/net/mraid20/export/jasmine/zach/cross_modal/saved_models/"

    def early_stop(self, curr_val_acc, pretrained_ecgnet, optimizer, temp):
        if curr_val_acc > self.max_acc:
            self.max_acc = curr_val_acc
            self.counter = 0
            checkpoint = {
                'model': pretrained_ecgnet,
                'optimizer': optimizer,
                 "min_loss": curr_val_acc,
                "temp": temp}
            torch.save(checkpoint, self.saved_model_path + self.saveName)
        elif curr_val_acc < (self.max_acc + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def do_test(test_people, batch_size, ds_train_ecg, device, ones_dir, twos_dir,remake_windows_from_scratch, temp, saveName):
    print("reading in saved model")
    saved_model_path = "/net/mraid20/export/jasmine/zach/cross_modal/saved_models/"
    checkpoint = torch.load(saved_model_path + saveName)
    checkpointed_ecgnet = checkpoint['model']
    checkpointed_ecgnet.eval()
    with torch.no_grad():
        emb_stores_apt_one = []
        emb_stores_apt_two = []
        ids_all_one = []
        ids_all_two = []
        int_ids_batched_test = np.array_split(np.random.choice(test_people, size=len(test_people), replace=False), max((len(test_people)) // batch_size, 1))
        epoch_test_loss = 0
        for test_batch in int_ids_batched_test:
            batched_ecgs_raw_apt_one_test, batched_ecgs_raw_apt_two_test, ids_one_test, ids_two_test = batch_manager(
                test_batch, ds_train_ecg, device, ones_dir, twos_dir,
                remake_windows_from_scratch)
            print("test: ", batched_ecgs_raw_apt_one_test.shape, " from first visits")
            ecg_embeddings_apt_one_test = checkpointed_ecgnet(batched_ecgs_raw_apt_one_test)
            if batched_ecgs_raw_apt_two_test is not None:
                ecg_embeddings_apt_two_test = checkpointed_ecgnet(batched_ecgs_raw_apt_two_test)
                emb_stores_apt_two.append(ecg_embeddings_apt_two_test)
                ids_all_two += ids_two_test
                print("test: ", batched_ecgs_raw_apt_two_test.shape, " from second visits")
            else:
                ecg_embeddings_apt_two_test = None
            emb_stores_apt_one.append(ecg_embeddings_apt_one_test)
            ids_all_one += ids_one_test
            test_loss = fasterContrast(ecg_embeddings_apt_one_test, ecg_embeddings_apt_two_test, temp, device,
                                       ids_one_test, ids_two_test)
            epoch_test_loss += float(np.real(test_loss)) / len(int_ids_batched_test)
        print("test loss per epoch: ", epoch_test_loss)
        wandb.log({"epoch_test_loss": epoch_test_loss})
        if len(emb_stores_apt_two) > 0:
            big_guy_metric(torch.cat(emb_stores_apt_one), torch.cat(emb_stores_apt_two), "test", ids_all_one,
                           ids_all_two)

def do_eval(pretrained_ecgnet, validation_people, ds_train_ecg, device, ones_dir, twos_dir, remake_windows_from_scratch, batch_size, temp):
    emb_stores_apt_one = []
    emb_stores_apt_two = []
    ids_all_one = []
    ids_all_two = []
    with torch.no_grad():
        pretrained_ecgnet.eval()
        int_ids_batched_validation = np.array_split(
            np.random.choice(validation_people, size=len(validation_people), replace=False), max(len(validation_people)//batch_size, 1))
        epoch_validation_loss = 0
        for validation_batch in int_ids_batched_validation:
            batched_ecgs_raw_apt_one_validation, batched_ecgs_raw_apt_two_validation, ids_one_validation, ids_two_validation = batch_manager(
                validation_batch, ds_train_ecg, device, ones_dir, twos_dir,
                remake_windows_from_scratch)
            print("validation: ", batched_ecgs_raw_apt_one_validation.shape, " from first visits")
            ecg_embeddings_apt_one_validation = pretrained_ecgnet(batched_ecgs_raw_apt_one_validation)
            if batched_ecgs_raw_apt_two_validation is not None:
                ecg_embeddings_apt_two_validation = pretrained_ecgnet(batched_ecgs_raw_apt_two_validation)
                emb_stores_apt_two.append(ecg_embeddings_apt_two_validation)
                ids_all_two += ids_two_validation
                print("validation: ", batched_ecgs_raw_apt_two_validation.shape, " from second visits")
            else:
                ecg_embeddings_apt_two_validation = None
            emb_stores_apt_one.append(ecg_embeddings_apt_one_validation)
            ids_all_one += ids_one_validation
            validation_loss = fasterContrast(ecg_embeddings_apt_one_validation, ecg_embeddings_apt_two_validation, temp,
                                             device, ids_one_validation, ids_two_validation)
            epoch_validation_loss += float(np.real(validation_loss)) / len(int_ids_batched_validation)
        print("validation loss per epoch: ", epoch_validation_loss)
        wandb.log({"validation_loss": epoch_validation_loss})
        if len(emb_stores_apt_two) > 0:
            across_one_acc = big_guy_metric(torch.cat(emb_stores_apt_one), torch.cat(emb_stores_apt_two), "validation", ids_all_one,
                           ids_all_two)
        return epoch_validation_loss, across_one_acc

def train_ecg_contrastive(restart,
                      config,
                      ds_train_ecg,
                      device,
                    train_people, test_people, validation_people, ones_dir, twos_dir, remake_windows_from_scratch, saveName, to_load_name = "SSL/NoFT_600.pth"):
    print("Using " + str(device))
    torch.cuda.empty_cache()
    epochs = config["epochs"]
    lr = config["learning_rate"]
    temp_0 = config["temp_0"]
    batch_size = config["batch_size"]
    optimizer = config["optim"]
    skipLinear = config["skipLinear"]
    warm_up_steps = config["warm_up_steps"]
    if restart:
        temp = torch.nn.Parameter(torch.Tensor([temp_0]).to(device), requires_grad = False)
        pretrained_ecgnet = both(device).to(device)
        optimizer = optimizer(list(pretrained_ecgnet.parameters()), lr=lr)
    else:
        print("reading in saved model")
        saved_model_path = "/net/mraid20/export/jasmine/zach/cross_modal/saved_models/"
        checkpoint = torch.load(saved_model_path + to_load_name)
        pretrained_ecgnet = checkpoint['model']
        optimizer = checkpoint['optimizer']
        temp = checkpoint["temp"]
    wandb.watch(pretrained_ecgnet, log_freq = 10, idx = 0)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            pretrained_ecgnet = nn.DataParallel(pretrained_ecgnet)
    ##only decoder parameters of both networks
    ##eran wants the epoch 0 "before-training" validation loss, so log it
    with torch.no_grad():
        pretrained_ecgnet.eval()
        eval_loss, max_acc = do_eval(pretrained_ecgnet, validation_people, ds_train_ecg, device, ones_dir, twos_dir, remake_windows_from_scratch, batch_size, temp)
    early_stopper = EarlyStopperContrastiveMaximize(max_acc = max_acc, patience = 25, saveName = saveName)
    for epoch in range(epochs):
        print("Starting epoch: ", epoch)
        int_ids_batched = np.array_split(np.random.choice(train_people, size = len(train_people), replace = False), len(train_people)//batch_size)
        pretrained_ecgnet.train()
        for batch in int_ids_batched:
            batched_ecgs_raw_apt_one, batched_ecgs_raw_apt_two, ids_one, ids_two = batch_manager(batch, ds_train_ecg, device,
                                                                                                 ones_dir, twos_dir, remake_windows_from_scratch)

            print(batched_ecgs_raw_apt_one.shape, " from first visits")
            ecg_embeddings_apt_one = pretrained_ecgnet(batched_ecgs_raw_apt_one, skipLinear)
            if batched_ecgs_raw_apt_two is not None:
                ecg_embeddings_apt_two = pretrained_ecgnet(batched_ecgs_raw_apt_two, skipLinear)
                print(batched_ecgs_raw_apt_two.shape, " from second visits")
            else:
                ecg_embeddings_apt_two = None
            train_loss = fasterContrast(ecg_embeddings_apt_one,
                                            ecg_embeddings_apt_two,
                                            temp,
                                            device, ids_one, ids_two)
            print(train_loss.item())
            wandb.log({"train_loss_mini_batch": float(train_loss)}),
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        epoch_validation_loss, across_one_acc = do_eval(pretrained_ecgnet, validation_people, ds_train_ecg, device, ones_dir, twos_dir, remake_windows_from_scratch, batch_size, temp)
        if early_stopper.early_stop(across_one_acc, pretrained_ecgnet, optimizer, temp):
            do_test(test_people, batch_size, ds_train_ecg, device, ones_dir, twos_dir, remake_windows_from_scratch,temp, saveName)
            return pretrained_ecgnet
