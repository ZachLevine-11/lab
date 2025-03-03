import torch
from datasets import ecg_dataset_pheno, get_domain_dataloaders, tenk_tabular_dataset, preprocess, min_max_scaling, max_ecg_val, min_ecg_val
import os
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import torch.nn as nn
from tslearn.metrics import SoftDTWLossPyTorch
import matplotlib.pyplot as plt
from models import PPGtoECG, BlackBoxAutoencoder, EarlyStopper
from LabData.DataLoaders.PRSLoader import PRSLoader
from LabData.DataLoaders.RetinaScanLoader import RetinaScanLoader
from LabData.DataLoaders.BloodTestsLoader import BloodTestsLoader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
from LabData.DataLoaders.UltrasoundLoader import UltrasoundLoader
from LabData.DataLoaders.ABILoader import ABILoader
from LabData.DataLoaders.DEXALoader import DEXALoader
from LabData.DataLoaders.ECGTextLoader import ECGTextLoader
import wandb
import torch.nn.functional as F
from scipy.signal import find_peaks
import pickle
from torch.autograd import functional


loader_set_dict = {1: ["ABI.pth", ],
                   2: ["DEXA.pth", ],
                   3: ["iglu.pth"],
                   4: ["RetinaScan.pth"],
                   5: ["PRS.pth"],
                   6: ["Ultrasound.pth"],
                   7: ["BodyMeasures.pth"],
                   "all": ["ABI.pth",
                           "Ultrasound.pth",
                           "BodyMeasures.pth",
                           "DEXA.pth",
                           "iglu.pth",
                           "RetinaScan.pth",
                           "PRS.pth"]
                   }

def extract_domain_name(k):
    k = k.split(".pth")[0]
    k = k.split("black_box_")[-1] if "black_box_" in k else k
    if k == "iglu" or k == "RNA":
        return k
    else:
        return eval(k + "Loader")

def align_ecg(true_peaks, ecg_pred):
    pred_peaks = find_peaks(ecg_pred, height=3)[0]
    a = min(true_peaks.shape[0], pred_peaks.shape[0])
    peaks_diff = pred_peaks[:a] - true_peaks[:a]
    if len(peaks_diff) > 0:
        avg_shift = peaks_diff[0]
        avg_shift = np.nan_to_num(avg_shift, 1, nan=1)
    else:
        avg_shift = 1
    if avg_shift > 0:
        ecg_pred = ecg_pred[int(avg_shift // 1):]
    else:
        ecg_pred = ecg_pred[:int(avg_shift // 1)]
    return ecg_pred


def ecg_plot(sim_results, ecg_true_eval, i=0):
    """
    Plots the ground truth and predicted ECG signals for all leads (channels).

    Parameters:
      sim_results (dict): A dictionary where each key maps to a dict containing
                          keys "pred_tabular" and "pred_time", each holding a tensor.
      ecg_true_eval (tensor): A tensor of shape (batch, leads, time) with ground truth ECG.
      i (int): An index appended to the saved file name.
    """
    n_leads = ecg_true_eval.shape[1]
    # Create a subplot for each lead. Adjust figsize as needed.
    fig, axes = plt.subplots(n_leads, 1, figsize=(40, 5 * n_leads), sharex=True)
    # If there's only one lead, wrap axes in a list for consistency.
    if n_leads == 1:
        axes = [axes]
    time_range = range(ecg_true_eval.shape[2])
    for lead in range(n_leads):
        ax = axes[lead]
        # Plot ground truth for the current lead.
        ecg_true = ecg_true_eval[0, lead, :].detach().cpu().flatten()
        ax.plot(time_range, ecg_true, label="Ground truth ECG", c="black", lw=2)

        # Plot predictions from each simulation result.
        for label, res in sim_results.items():
            # Plot "pred_tabular" if available.
            pred_tab = res.get("pred_tabular")
            if pred_tab is not None:
                ecg_pred = pred_tab[0, lead, :].detach().cpu().flatten()
                ax.plot(range(ecg_pred.shape[0]), ecg_pred,
                        label="Predicted from " + label)

            # Plot "pred_time"
            pred_time = res.get("pred_time")
            if pred_time is not None:
                ecg_pred2 = pred_time[0, lead, :].detach().cpu().flatten()
                ax.plot(range(ecg_pred2.shape[0]), ecg_pred2,
                        label="AE Output " + label)

        ax.set_ylabel("Voltage (mV)")
        ax.set_title(f"Lead {lead}")
        ax.legend()

    axes[-1].set_xlabel("Time (t)")
    plt.tight_layout()
    plt.savefig("/home/zacharyl/Desktop/ecg/eval_" + label + "_" + str(i) + ".jpg")

def do_forward(config, model, input, length_sim, length_window, ecg_no_interp, ecg_true_interp, mechanistic = True, do_output_first = True):
    do_output_first = mechanistic
    if config.do_integration:
        z_first, dist_first = model(None, input, length_sim, onlyz=True, use_time_series=False, use_baselines=True,
                                length_window=length_window)
    else:
        z_first, dist_first = None, None
    ##do not change!!!
    z_second, dist_second = model(ecg_no_interp, None, onlyz=True, use_time_series=True, use_baselines=False,
                                  length_sim=length_sim, length_window=length_window)
    output_second, jac_second = model(None, None, length_sim=length_sim, fromz=True, z_in=z_second, length_window=length_window)
    if mechanistic:
        if not do_output_first or not config.do_integration:
            output_first = None
        else:
            output_first, jac = model(None, None, length_sim=length_sim, fromz=True, z_in=z_first, length_window=length_window)
    else:
        output_first = None
    output_second = output_second.permute(1, 2, 0)
    output_second = min_max_scaling(output_second, dim = -1)
    if output_first is not None:
        output_first = output_first.permute(1, 2, 0)
        output_first = min_max_scaling(output_first, dim = -1)
    return output_first, output_second, z_first, z_second, dist_first, dist_second, jac_second

def do_loss(model, output_first, output_second, dist_first, dist_second, ecg_true_interp, config, label, mode, jac, metric_time_matters = nn.MSELoss()):
    metric_time_invariant = SoftDTWLossPyTorch(gamma=config.gamma, normalize=config.normalize_dtw_loss)
    if config.first_to_second_alpha > 0:
        time_first_second = config.first_to_second_alpha*metric_time_invariant(output_first.reshape(output_first.shape[0], -1, 3), output_second.reshape(output_first.shape[0], -1, 3)).mean()
        wandb.log({label + "_" + "first to second " + mode: time_first_second})
    else:
        time_first_second = 0
    if config.alpha_dtw > 0:
        dtw = config.alpha_dtw*metric_time_invariant(output_second.reshape(output_second.shape[0], -1, 1), ecg_true_interp.reshape(output_second.shape[0], -1, 1)).mean()
        wandb.log({label + "_" + "dtw second " + mode: dtw})
    else:
        dtw = 0
    if config.mse_alpha > 0:
        mse = config.mse_alpha*metric_time_matters(output_second, ecg_true_interp)
        wandb.log({label + "_" + "MSE second " + mode: mse})
    else:
        mse = 0
    print(mse)
    if config.jac_alpha > 0:
        jac_loss = jac*config.jac_alpha
        wandb.log({label + "_" + "jac " + mode: jac * config.jac_alpha})
    else:
        jac_loss = 0
    time_second_real = dtw + mse + jac_loss
    wandb.log({label + "_" + "time second " + mode: time_second_real})
    if config.kl_alpha1 > 0 and config.do_integration:
        kl = torch.distributions.kl.kl_divergence(dist_first, dist_second).sum((0, 1)) * config.kl_alpha1
    else:
        kl = 0
    if config.kl_alpha2 > 0:
        kl2 = torch.distributions.kl.kl_divergence(dist_second, model.kl_dist).sum((0, 1)) * config.kl_alpha2
        wandb.log({label + "_" + "kl2  " + mode: kl2})
    else:
        kl2 = 0
    wandb.log({label + " _" + "kl " + mode: kl})
    loss = time_second_real + kl + time_first_second + kl2
    wandb.log({label + " _" + "total " + mode: loss})
    return time_first_second, time_second_real, loss, kl, mse

def do_eval(config, model, ecg_true_interp_eval, ecg_no_interp_eval, input, label, length_sim, length_window):
    model.training = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        output_first_eval, output_second_eval, __, __, dist_first_eval, dist_second_eval, jac = do_forward(config,
                                                                                                         model, input, length_sim, length_window,
                                                                                                        ecg_no_interp_eval,
                                                                                                        ecg_true_interp_eval,
                                                                                                        mechanistic = config.mechanistic,
                                                                                                        do_output_first = True)
        time_first_second, time_second_real, loss, kl, mse = do_loss(model, output_first_eval, output_second_eval, dist_first_eval, dist_second_eval, ecg_true_interp_eval, config, label, mode = "eval", jac = jac)
    return output_first_eval, output_second_eval, ecg_true_interp_eval, loss, mse

def make_ecg_fmap(tenk_id_list, files, from_cache):
    if from_cache:
        with open('/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/caches/ecg_fmap.pickle', 'rb') as handle:
            fmap = pickle.load(handle)
        return fmap
    else:
        fmap = {}
        for tenk_id in tenk_id_list:
            matching_ecg_files = list(filter(lambda x: x.startswith("ecg__10k__" + tenk_id.split("10K_")[-1]), files))
            fmap[tenk_id] = list(map(lambda x: x.split("__")[-1].split(".parquet")[0], matching_ecg_files))
        with open("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/caches/ecg_fmap.pickle", 'wb') as handle:
            pickle.dump(fmap, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return fmap

def fasterContrast(emb_matrix_appointment_one, emb_matrix_appointment_two, device = "cuda"):
    emb_matrix_appointment_one, emb_matrix_appointment_two = F.normalize(emb_matrix_appointment_one), F.normalize(emb_matrix_appointment_two)
    logits_only_one = (emb_matrix_appointment_one @ emb_matrix_appointment_two.t()) * torch.exp(torch.FloatTensor([0.07]).to(device))
    loss_i_only_one = F.cross_entropy(logits_only_one, torch.arange(logits_only_one.shape[0], device = device))
    loss_t_only_one = F.cross_entropy(logits_only_one.T, torch.arange(logits_only_one.shape[0], device = device))
    dist_only_one = (loss_i_only_one + loss_t_only_one) / 2
    return dist_only_one

def train(config,
          model,
          optimizer,
          batch_data,   # <-- Instead of separate ecg/input, we accept a single batch dict.
          device,
          label,
          scheduler):
    """
    Minimal-changes version of your train() function that uses the already-prepared batch_data from a DataLoader.
    """
    torch.cuda.empty_cache()
    model.train()
    model.training = True

    # 1) Extract from batch
    ecg_no_interp   = batch_data["ecg_no_interp"].to(device)   # (B, 12, length_window)
    ecg_true_interp = batch_data["ecg_true_interp"].to(device) # (B, 12, length_sim)
    input_feats     = batch_data["tabular_feats"].to(device)   # (B, N_features)

    # 2) Forward pass
    output_first, output_second, z_first, z_second, dist_first, dist_second, jac = do_forward(
        config,
        model,
        input_feats,
        config.length_sim,
        config.length_window,
        ecg_no_interp,
        ecg_true_interp,
        mechanistic=config.mechanistic
    )

    # 3) Compute loss
    time_first_second, time_second_real, loss, kl, mse = do_loss(
        model,
        output_first,
        output_second,
        dist_first,
        dist_second,
        ecg_true_interp,
        config,
        label,
        mode="train",
        jac=jac
    )

    # 4) Backprop + step
    if not torch.isnan(loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    else:
        print("nan loss encountered in batch")

    # Return the scalar loss for logging
    return loss.item()

def get_label(ds_input):
    label = str(ds_input.dataloader).split(".")[-1].split("'")[0]
    if "Loader" in label:
        label = label.split("Loader")[0]
    return label

def do_eval_pass(config,
                 model,
                 eval_loader,
                 device,
                 label):
    """
    Minimal-changes version that includes the original "plot saving" logic.
    We'll do a simple once-per-batch approach if config.plot is True.
    """
    model.eval()
    model.training = False
    total_mse = 0.0
    num_batches = 0

    for batch_idx, batch_data in enumerate(eval_loader):
        if batch_data is None:
            continue

        ecg_no_interp_eval   = batch_data["ecg_no_interp"].to(device)
        ecg_true_interp_eval = batch_data["ecg_true_interp"].to(device)
        input_feats_eval     = batch_data["tabular_feats"].to(device)

        # Forward pass
        output_first_eval, output_second_eval, _, _, dist_first_eval, dist_second_eval, jac_eval = do_forward(
            config,
            model,
            input_feats_eval,
            config.length_sim,
            config.length_window,
            ecg_no_interp_eval,
            ecg_true_interp_eval,
            mechanistic=config.mechanistic
        )

        # Compute loss or MSE
        _, _, loss_eval, _, mse_eval = do_loss(
            model,
            output_first_eval,
            output_second_eval,
            dist_first_eval,
            dist_second_eval,
            ecg_true_interp_eval,
            config,
            label,
            mode="eval",
            jac=jac_eval
        )
        total_mse += mse_eval.item()
        num_batches += 1
        if config.plot and (batch_idx % config.plot_every == 0):
            sim_results = {
                label: {
                    "pred_tabular": output_first_eval,
                    "pred_time":    output_second_eval
                }
            }
            ecg_plot(sim_results, ecg_true_interp_eval, i=batch_idx)

    if num_batches == 0:
        return 0.0

    return total_mse / num_batches


def save(models, optimizers, dir):
    for k, model in models.items():
        label = get_label(k)
        checkpoint = {
            'model': model,
            'optimizer': optimizers[model]}
        torch.save(checkpoint, dir + label + ".pth")

def make_and_save_split(people_with_all, config, loader_set):
    test_people = list(np.random.choice(people_with_all, config.total_test_n, replace=False))
    people_remaining = list(set(people_with_all) - set(test_people))
    with open(f"/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/splits/loader_set_{loader_set}_test_people.pickle", 'wb') as handle:
        pickle.dump(test_people, handle, protocol=pickle.HIGHEST_PROTOCOL)
    eval_people = list(np.random.choice(people_remaining, config.total_eval_n, replace=False))
    with open(f"/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/splits/loader_set_{loader_set}_eval_people.pickle", 'wb') as handle:
        pickle.dump(eval_people, handle, protocol=pickle.HIGHEST_PROTOCOL)
    train_people = list(set(people_remaining) - set(eval_people))
    with open(f"/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/splits/loader_set_{loader_set}_train_people.pickle", 'wb') as handle:
        pickle.dump(train_people, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return train_people, eval_people, test_people

def load_split(loader_set):
    with open(f"/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/splits/loader_set_{loader_set}_test_people.pickle", 'rb') as handle:
        test_people = pickle.load(handle)
    with open(f"/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/splits/loader_set_{loader_set}_train_people.pickle", 'rb') as handle:
        train_people = pickle.load(handle)
    with open(f"/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/splits/loader_set_{loader_set}_eval_people.pickle", 'rb') as handle:
        eval_people = pickle.load(handle)
    return train_people, eval_people, test_people

def warmup_lambda(step, config):
        if step < config.warmup_steps:
            return (step + 1) / config.warmup_steps
        return 1.0

def do(config):
    np.random.seed(0)
    torch.manual_seed(0)
    ecg_files = os.listdir("/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/Pheno/stas/ecg/waveform/clean/long_format/")
    all_samples_ecg_pheno = pd.Series(list(map(lambda x: "10K_" + x.split("ecg__10k__")[-1].split("__")[0], ecg_files))).reset_index().set_index(0).index.values
    all_samples_ecg_tenk = set(ECGTextLoader().get_data(study_ids=list(range(100)) + list(range(1000, 1011, 1))).df.reset_index().RegistrationCode)  # 10K cohort samples
    all_samples_ecg = list(set(all_samples_ecg_pheno).intersection(all_samples_ecg_tenk))  # keep pheno samples that are 10K cohort samples only
    ecg_fmap = make_ecg_fmap(tenk_id_list=all_samples_ecg, files=ecg_files, from_cache=True)
    ds_ecg = ecg_dataset_pheno(all_samples_ecg=all_samples_ecg, fmap=ecg_fmap)
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using " + str(device))
    lr = config.lr
    if config.weight_sharing:
        encoder_lstm = nn.LSTM(input_size=12, proj_size=0, hidden_size=config.hidden_size, num_layers=config.num_layers,
                               batch_first=False, dropout=config.encoder_dropout_prob, bidirectional=True)
        decoder_lstm = nn.LSTM(input_size=3, proj_size=0,
                               hidden_size=config.hidden_size, num_layers=config.num_layers,
                               batch_first=False, dropout=config.encoder_dropout_prob, bidirectional=False)
    else:
        encoder_lstm = None
        decoder_lstm = None

    if config.load_models:
        saved_model_dir = "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/saved_models/"
        saved_models = loader_set_dict[config.loader_set]
        if not config.mechanistic:
            saved_models = list(map(lambda x: "black_box_" + x, saved_models))
        checkpoints = dict(zip(saved_models, [torch.load(saved_model_dir + x, map_location=device) for x in saved_models]))
        tabular_domains = [extract_domain_name(k) for k in checkpoints]
        tabular_datasets = [tenk_tabular_dataset(x) for x in tabular_domains]
        datasets_to_checkpoints = {dict(zip(tabular_domains, tabular_datasets))[extract_domain_name(k)]: v
                                   for k, v in checkpoints.items()}
        labels = list(map(lambda x: x if x == "RNA" or x == "iglu" else str(x).split(".")[2].split("Loader")[0], tabular_domains))
        tabular_domain_labels = dict(zip(labels, tabular_datasets))
        models = {k: v["model"] for k, v in datasets_to_checkpoints.items()}
        optimizers = {v["model"]: v["optimizer"] for k, v in datasets_to_checkpoints.items()}
    else:
        tabular_domains = list(map(
            lambda x: eval(x.split(".pth")[0] + "Loader") if x != "iglu.pth" and x != "RNA.pth" else x.split(".pth")[0],
            loader_set_dict[config.loader_set]
        ))
        tabular_datasets = [tenk_tabular_dataset(x) for x in tabular_domains]
        labels = list(map(
            lambda x: x if x == "RNA" or x == "iglu" else str(x).split(".")[2].split("Loader")[0],
            tabular_domains
        ))
        tabular_domain_labels = dict(zip(labels, tabular_datasets))

        if config.mechanistic:
            models = dict(zip(
                tabular_datasets,
                [
                    PPGtoECG(hidden_size=config.hidden_size,
                             num_layers=config.num_layers,
                             encoder_dropout_prob=config.encoder_dropout_prob,
                             dt=config.dt,
                             encoder_lstm=encoder_lstm,
                             decoder_lstm=decoder_lstm,
                             num_baseline_features=len(ds_input.df.columns),
                             temperature=config.temperature).to(device)
                    for ds_input in tabular_datasets
                ]
            ))
        else:
            models = dict(zip(
                tabular_datasets,
                [
                    BlackBoxAutoencoder(encoding_size=50,
                                        encoder_input_size=30,
                                        hidden_size=config.hidden_size,
                                        num_layers=config.num_layers,
                                        encoder_dropout_prob=config.encoder_dropout_prob,
                                        num_baseline_features=len(ds_input.df.columns),
                                        decoder_dropout_prob=config.encoder_dropout_prob,
                                        decoder_input_size=1).to(device)
                    for ds_input in tabular_datasets
                ]
            ))

        optimizers = dict(zip(
            models.values(),
            [torch.optim.Adam(model.parameters(), lr=lr) for model in models.values()]
        ))

    schedulers = dict(zip(
        optimizers.values(),
        [LambdaLR(optimizer, lr_lambda=lambda x: warmup_lambda(x, config)) for optimizer in optimizers.values()]
    ))

    people_with_all = list(set(all_samples_ecg).intersection(
        set.intersection(*[set(x.df.index.values) for x in tabular_datasets])
    ))
    early_stopper = EarlyStopper(patience=3)

    if config.make_new_split:
        train_people, eval_people, test_people = make_and_save_split(people_with_all, config, config.loader_set)
    else:
        train_people, eval_people, test_people = load_split(config.loader_set)

    # Build DataLoaders
    train_loaders, eval_loaders = get_domain_dataloaders(
        train_people,
        eval_people,
        ds_ecg,
        tabular_domain_labels,  # {label: dataset}
        config
    )

    for epoch in range(config.epochs):
        # ----------------- TRAINING PASS -----------------
        print(f"Epoch {epoch + 1}/{config.epochs}")
        epoch_train_loss = 0.0
        randomized_labels = list(np.random.permutation(list(tabular_domain_labels.keys())))
        random_order = dict(zip(
            randomized_labels,
            map(lambda x: tabular_domain_labels.get(x), randomized_labels)
        ))
        i = 0
        for domain_label, dataset in random_order.items():
            model = models[tabular_domain_labels[domain_label]]
            optimizer = optimizers[model]
            scheduler = schedulers[optimizer]
            train_loader = train_loaders[domain_label]

            for batch_idx, batch_data in enumerate(train_loader):
                if batch_data is None:
                    continue
                single_batch_loss = train(config, model, optimizer, batch_data, device, domain_label, scheduler)
                epoch_train_loss += single_batch_loss
                i += 1
        avg_train_loss = epoch_train_loss / i
        wandb.log({"epoch_train_loss": avg_train_loss})
        print(f"Epoch {epoch + 1} complete. Avg training loss = {avg_train_loss:.4f}")
        epoch_eval_loss_mse = 0.0
        if config.do_eval:
            for domain_label, dataset in tabular_domain_labels.items():
                model = models[tabular_domain_labels[domain_label]]
                eval_loader = eval_loaders[domain_label]

                domain_eval_mse = do_eval_pass(
                    config,
                    model,
                    eval_loader,
                    device,
                    domain_label
                )
                epoch_eval_loss_mse += domain_eval_mse
            avg_eval_mse = epoch_eval_loss_mse / len(tabular_domains)
            early_stopper(avg_eval_mse)
            if early_stopper.early_stop:
                print("Early stopping triggered. Stopping training.")
                break
            if avg_eval_mse <= early_stopper.best_score:
                save(
                    models,
                    optimizers,
                    dir=(
                        f"/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/saved_models/"
                        f"{'integrated_' if config.do_integration else ''}"
                    )
                )
    print("Training loop completed.")