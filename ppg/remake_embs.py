import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
import pandas as pd
import torch
from all_to_ecg import loader_set_dict, do_forward, get_label, make_ecg_fmap, load_split, extract_domain_name
import numpy as np
import pandas as pd
from collections import defaultdict
import os
from datasets import ecg_dataset_pheno, get_domain_dataloaders, tenk_tabular_dataset, preprocess, min_max_scaling, max_ecg_val, min_ecg_val
from single_run_wandb import sweep_configuration as config
from LabData.DataLoaders.ECGTextLoader import ECGTextLoader

saved_emb_dir = '/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/embs/'
saved_model_dir = "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/saved_models/"
inverse_load_set_dict = {tuple(v):k for k,v in loader_set_dict.items()}

def make_par_labels():
    pars = []
    for wave in ["P", "Q", "R", "S", "T_plus", "T_iminus"]:
        pars.append(wave + "_a")
        pars.append(wave + "_theta")
        pars.append(wave + "_b")
    for rr_par in ["R1", "R2"]:
        pars.append(rr_par + "_mean")
        pars.append(rr_par + "_sd")
    pars.append("LF_HF_ratio")
    return pars + ["theta_0", "z0_0"]

def make_embs(z_dict):
    for label, embs_dict in z_dict.items():
        arr = pd.DataFrame(np.concatenate(list(embs_dict.values())))
        arr.index = np.concatenate(list(embs_dict.keys())).astype(str)
        arr.columns = make_par_labels()
        z_dict[label] = arr
    return z_dict

def extract_dist_all(ds_ecg, device, config, tabular_domains, models, tabular_datasets, tabular_domain_labels, do_integrated = False, return_dist = False, save = True):
    length_sim = config["length_sim"]
    length_window = config["length_window"]
    for i in range(len(tabular_datasets)):
        z_first_all, z_second_all, dist_first_all, dist_second_all = defaultdict(dict), defaultdict(dict), defaultdict(
            dict), defaultdict(dict)
        ds_input = tabular_datasets[i]
        label = f"{get_label(ds_input)}{'_integrated' if do_integrated else ''}"
        model = models[ds_input]
        if do_integrated:
            operative_label_for_split = f"{label.split('_integrated')[0]}"
        else:
            operative_label_for_split = label
        labelkey = inverse_load_set_dict[(f"{operative_label_for_split}.pth",)]
        train_people, eval_people, test_people = load_split(loader_set=labelkey)
        __, test_loaders = get_domain_dataloaders( ##replace eval set with test set, a set on which the model was neither trained nor used as eval in the model (as a stopping condition)
            train_people,
            test_people,
            ds_ecg,
            tabular_domain_labels,  # {label: dataset}
            config
        )
        eval_loader = test_loaders[label.split("_integrated")[0]]
        for batch_idx, batch_data in enumerate(eval_loader):
            if batch_data is None:
                continue
            ecg_no_interp_eval = batch_data["ecg_no_interp"].to(device)
            ecg_true_interp_eval = batch_data["ecg_true_interp"].to(device)
            input_feats_eval = batch_data["tabular_feats"].to(device)
            tenk_ids = batch_data["tenk_id"]
            z_first_all[label][tuple(tenk_ids)], z_second_all[label][tuple(tenk_ids)], dist_first_all[label][tuple(tenk_ids)], dist_second_all[label][tuple(tenk_ids)] = extract_dist(config, model,
                                                                      ecg_true_interp_eval,
                                                                      ecg_no_interp_eval,
                                                                      input_feats_eval, length_sim, length_window)

        z_first_all, z_second_all = make_embs(z_first_all), make_embs(z_second_all)
    if save:
        for k in z_second_all.keys():
            z_second_all[k].to_csv(f"{saved_emb_dir}{k}_time.csv")
            z_first_all[k].to_csv(f"{saved_emb_dir}{k}.csv")
    if return_dist:
        return dist_first_all, dist_second_all
    else:
        return 1, 1

def constrain_pars(model, z):
    param, state = z[..., :model.param_size], z[..., model.param_size:]
    param = model.constrain(param, min=model.param_lims[:, 0], max=model.param_lims[:, 1])
    state = model.constrain(state, min=model.state_lims[:, 0], max=model.state_lims[:, 1])
    return torch.cat([param, state], dim=-1)

##to get the forward pass to work without redefining everything
class pseudo_config(dict):
    def __init__(self):
        super().__init__()
        self.do_integration = True

def extract_dist(config, model, ecg_true_interp_eval, ecg_no_interp_eval, input, length_sim, length_window):
    model.training = False
    model.eval()
    with torch.no_grad():
        output_first, output_second, z_first, z_second,  dist_first, dist_second, jac = do_forward(pseudo_config(),
                                                                                                        model, input,
                                                                                                        length_sim, length_window,
                                                                                                        ecg_no_interp_eval,
                                                                                                        ecg_true_interp_eval,
                                                                                                        do_output_first = True)
        z_first, z_second = constrain_pars(model, z_first),  constrain_pars(model, z_second)
    return z_first.detach().cpu(), z_second.detach().cpu(), dist_first, dist_second

def load_things(saved_model_dir, do_integrated):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    if do_integrated:
        saved_models = [x for x in os.listdir(saved_model_dir) if x.endswith(".pth") and x.startswith("integrated_")]
    else:
        saved_models = [x for x in os.listdir(saved_model_dir) if x.endswith(".pth") and not x.startswith("integrated_")]
    checkpoints = dict(zip(saved_models, [torch.load(saved_model_dir + x, map_location=device) for x in saved_models]))
    tabular_domains = [extract_domain_name(k.split("integrated_")[-1] if "integrated_" in k else k) for k in checkpoints]
    tabular_datasets = [tenk_tabular_dataset(x) for x in tabular_domains]
    labels = list(
        map(lambda x: x if x == "RNA" or x == "iglu" else str(x).split(".")[2].split("Loader")[0], tabular_domains))
    tabular_domain_labels = dict(zip(labels, tabular_datasets))
    datasets_to_checkpoints = {dict(zip(tabular_domains, tabular_datasets))[
                                   extract_domain_name(k.split("integrated_")[-1] if "integrated_" in k else k)]: v for
                               k, v
                               in checkpoints.items()}
    return tabular_datasets, tabular_domains, tabular_domain_labels, checkpoints, datasets_to_checkpoints

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(0)
    torch.manual_seed(0)
    ecg_files = os.listdir(
        "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/Pheno/stas/ecg/waveform/clean/long_format/")
    all_samples_ecg_pheno = pd.Series(
        list(map(lambda x: "10K_" + x.split("ecg__10k__")[-1].split("__")[0], ecg_files))).reset_index().set_index(
        0).index.values
    all_samples_ecg_tenk = set(ECGTextLoader().get_data(
        study_ids=list(range(100)) + list(range(1000, 1011, 1))).df.reset_index().RegistrationCode)  # 10K cohort samples
    all_samples_ecg = list(set(all_samples_ecg_pheno).intersection(
        all_samples_ecg_tenk))  ##keep pheno samples that are 10K cohort samples only
    ecg_fmap = make_ecg_fmap(tenk_id_list=all_samples_ecg, files=ecg_files, from_cache=True)
    ds_ecg = ecg_dataset_pheno(all_samples_ecg=all_samples_ecg, fmap=ecg_fmap)
    print("Using " + str(device))
    do_integrated = True
    do_z_gen = True
    config = {k:v["values"][0] for k, v in config["parameters"].items()}
    if do_z_gen:
        tabular_datasets, tabular_domains, tabular_domain_labels, checkpoints, datasets_to_checkpoints = load_things(saved_model_dir, do_integrated)
        models = {k: v["model"].to(device) for k, v in datasets_to_checkpoints.items()}
        _, __ = extract_dist_all(ds_ecg, device, config,
                             tabular_domains, models,
                             tabular_datasets,
                             tabular_domain_labels,
                             do_integrated = do_integrated,
                             return_dist = False,
                            save = True)
    do_dist_gen = False
    if do_dist_gen:
        ##for the dist computations
        specfic_loader_list = ["BodyMeasures.pth"] ##set to None for black box generation
        tabular_datasets, tabular_domains, tabular_domain_labels, checkpoints, datasets_to_checkpoints = load_things(saved_model_dir, do_integrated)
        models = {k: v["model"].to(device) for k, v in datasets_to_checkpoints.items()}
        d1, d2 = extract_dist_all(ds_ecg, device, config,
                             tabular_domains, models,
                             tabular_datasets,
                             tabular_domain_labels,
                             do_integrated = do_integrated,
                            return_dist=True,
                            save = False)
        dist_one_extracted_loc = {}
        dist_one_extracted_scale = {}
        for k, v in d1["BodyMeasures"].items():
            dist_one_extracted_loc[k] = v.loc.detach().cpu().numpy()
            dist_one_extracted_scale[k] = v.scale.detach().cpu().numpy()
        arr_one_loc = pd.DataFrame(np.concatenate(list(dist_one_extracted_loc.values())))
        arr_one_loc.index = np.concatenate(list(dist_one_extracted_loc.keys())).astype(str)
        arr_one_loc.columns = make_par_labels()
        arr_one_scale = pd.DataFrame(np.concatenate(list(dist_one_extracted_scale.values())))
        arr_one_scale.index = np.concatenate(list(dist_one_extracted_scale.keys())).astype(str)
        arr_one_scale.columns = make_par_labels()
        arr_one_loc.to_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/dist/loc_one.csv")
        arr_one_scale.to_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/dist/scale_one.csv")
        dist_two_extracted_loc = {}
        dist_two_extracted_scale = {}
        for k, v in d2["BodyMeasures"].items():
            dist_two_extracted_loc[k] = v.loc.detach().cpu().numpy()
            dist_two_extracted_scale[k] = v.scale.detach().cpu().numpy()
        arr_two_loc = pd.DataFrame(np.concatenate(list(dist_two_extracted_loc.values())))
        arr_two_loc.index = np.concatenate(list(dist_two_extracted_loc.keys())).astype(str)
        arr_two_loc.columns = make_par_labels()
        arr_two_scale = pd.DataFrame(np.concatenate(list(dist_two_extracted_scale.values())))
        arr_two_scale.index = np.concatenate(list(dist_two_extracted_scale.keys())).astype(str)
        arr_two_scale.columns = make_par_labels()
        arr_two_loc.to_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/dist/loc_two.csv")
        arr_two_scale.to_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/dist/scale_two.csv")