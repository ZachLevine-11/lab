import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import itertools
from sklearn.metrics import roc_auc_score
import seaborn as sns
from all_to_ecg import loader_set_dict, do_forward, get_label, batch_director, add_input, make_ecg_fmap, make_ppg_fmap, load_split, extract_domain_name
import numpy as np
import pandas as pd
from collections import defaultdict
from datasets import ecg_dataset_pheno
from single_run_wandb import sweep_configuration as config
from LabData.DataLoaders.ECGTextLoader import ECGTextLoader
from remake_embs import saved_model_dir, inverse_load_set_dict, make_par_labels, saved_emb_dir
from make_figures import make_age_df, cv, make_diseases_df, read_load_embs
from matplotlib import pyplot as plt
from datasets import tenk_tabular_dataset
from sklearn.decomposition import PCA
from torch.distributions import Normal

def do_counterfactual(ds_ecg, device, config, tabular_domains, models, tabular_datasets, tabular_domain_labels, quantiles, counterfactual_col, dataset_to_use):
    length_sim = config["length_sim"]
    length_window = config["length_window"]
    for j, ds_input in tabular_datasets.items():
        z_first_all, dist_first_all = defaultdict(lambda: defaultdict(dict)), defaultdict(lambda: defaultdict(dict))
        label = get_label(ds_input)
        model = models[ds_input]
        operative_label_for_split = label
        if config["loader_set"] != "all":
            labelkey = inverse_load_set_dict[(f"{operative_label_for_split}.pth",)]
        else:
            labelkey = "all"
        train_people, eval_people, test_people = load_split(loader_set=labelkey)
        all_people = list(set.union(*[set(train_people), set(test_people), set(eval_people)]))
        eval_batches = np.array_split(np.random.choice(all_people, size=len(all_people), replace=False),
                                      max(len(all_people) // 200, 1))
        counter = 0
        for batch_eval in eval_batches:
            print(counter)
            ecg_true_interp_eval, ecg_no_interp_eval, ids_tenk_existing_data_eval = batch_director(
                batch_eval, ds_ecg, device, config, length_sim, length_window)
            ecg_true_interp_eval, inputs_eval = add_input(ids_tenk_existing_data_eval, ds_input, ecg_true_interp_eval, length_sim, device)
            counterfactual_idx  =list(dataset_to_use.columns).index(counterfactual_col)
            simulated_inputs = []
            for quantile in quantiles:
                temp = inputs_eval.clone()
                temp[:, counterfactual_idx] = quantile*torch.ones_like(temp[:, counterfactual_idx])
                simulated_inputs.append(temp.view(inputs_eval.shape[0], -1))
            ##batch dim, quantile, feature
            sim_tensor = torch.stack(simulated_inputs, dim = 1)
            ##to make sure the tensor is being filled with the counterfactual values properly
           # print(sim_tensor[:, 10, counterfactual_idx])
           # print(sim_tensor[:, 5, counterfactual_idx])
            for i in range(sim_tensor.shape[1]):##all evalute each quantile at a time
                z_first, dist_first = model(None, sim_tensor[:, i, :], length_sim, onlyz=True, use_time_series=False, use_baselines=True,
                                                length_window=length_window)
                z_first_all[label][tuple(ids_tenk_existing_data_eval)][i] = z_first.detach().cpu()
                dist_first_all[label][tuple(ids_tenk_existing_data_eval)][i] = dist_first
            counter += 1
    return z_first_all, dist_first_all

def constrain_pars(model, z):
    param, state = z[..., :model.param_size], z[..., model.param_size:]
    param = model.constrain(param, min=model.param_lims[:, 0], max=model.param_lims[:, 1])
    state = model.constrain(state, min=model.state_lims[:, 0], max=model.state_lims[:, 1])
    return torch.cat([param, state], dim=-1)

def extract_dist(config, model, ecg_true_interp_eval, ecg_no_interp_eval, input, length_sim, length_window, do_blackbox):
    model.training = False
    model.eval()
    with torch.no_grad():
        output_first, output_second, z_first, z_second,  dist_first, dist_second, jac = do_forward(config,
                                                                                                        model, input,
                                                                                                        length_sim, length_window,
                                                                                                        ecg_no_interp_eval,
                                                                                                        ecg_true_interp_eval,
                                                                                                        do_output_first = True)
    if not do_blackbox:
        z_first, z_second = constrain_pars(model, z_first),  constrain_pars(model, z_second)
    return z_first.detach().cpu(), z_second.detach().cpu(), dist_first, dist_second

def load_things(saved_model_dir, do_blackbox, specific_models = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    if specific_models is not None:
        saved_models = specific_models
    else:
        if do_blackbox:
            saved_models = [x for x in os.listdir(saved_model_dir) if x.endswith(".pth") and x.startswith("black_box")]
        else:
            saved_models = [x for x in os.listdir(saved_model_dir) if x.endswith(".pth") and not x.startswith("black_box")]
    checkpoints = dict(zip(saved_models, [torch.load(saved_model_dir + x, map_location=device) for x in saved_models]))
    tabular_domains = [extract_domain_name(k.split("black_box_")[-1] if "black_box_" in k else k) for k in checkpoints]
    tabular_domain_labels = list(map(lambda x: x if x == "RNA" or x == "iglu" else str(x).split(".")[2].split("Loader")[0], tabular_domains))
    tabular_datasets = dict(zip(tabular_domain_labels, [tenk_tabular_dataset(x) for x in tabular_domains]))
    datasets_to_checkpoints = {dict(zip(tabular_domains, list(tabular_datasets.values())))[
                                   extract_domain_name(k.split("black_box_")[-1] if "black_box_" in k else k)]: v for
                               k, v
                               in checkpoints.items()}
    return tabular_datasets, tabular_domains, tabular_domain_labels, checkpoints, datasets_to_checkpoints

def two_feature_diseases_iterator(df, features, what = "5B81", top_i = 3, how = "ols", task = "r"):
    res = {}
    for i in range(1, top_i):
        cols = list(itertools.permutations(df.columns, i))
        for col_set in cols:
            all_preds, _ = cv(df[list(col_set)], features, what = what, how = "ols", task = "c")
            all_T = roc_auc_score(all_preds["true"], all_preds["preds"])
            res[(tuple(col_set), i)] = all_T
    res = pd.DataFrame(res.values(), index = res.keys()).sort_values(ascending = False, by = 0)
    return res

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(0)
    torch.manual_seed(0)
    config = {k:v["values"][0] for k,v in config["parameters"].items()}
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
    loader = "ABI.pth"
    specfic_loader_list = [loader]
    tabular_datasets, tabular_domains, tabular_domain_labels, checkpoints, datasets_to_checkpoints = load_things(saved_model_dir, do_blackbox = False, specific_models=specfic_loader_list)
    models = {k: v["model"].to(device) for k, v in datasets_to_checkpoints.items()}
    train_people, eval_people, test_people = load_split(loader_set="all")
    all_people = list(set.union(*[set(train_people), set(test_people), set(eval_people)]))
    num_quantiles = 100
    ##must match loader
    matching_dataset_name = loader.split(".pth")[0]
    dataset_to_use = tabular_datasets[matching_dataset_name].df
    print("using dataset with first five cols as: ", dataset_to_use.columns[0:5])
    print(f"using loader {loader}")
    print("if these two don't agree, stop")
    existing_counterfactual = os.listdir("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/counterfactual")
    orig_cols = list(dataset_to_use.loc[all_people].columns)
    remaining = list(set(orig_cols) - set(existing_counterfactual))
    print(f"ignoring {len(existing_counterfactual)}, leaving {len(remaining)} from {len(orig_cols)}")
   ## remaining =  ["sitting_blood_pressure_systolic", "hand_grip_left", "bmi"] save time when writing the paper
    for counterfactual_col in remaining:
        quantiles =pd.qcut(dataset_to_use.loc[all_people][counterfactual_col], q=num_quantiles, duplicates='drop',
                retbins=True)[1]
        z_ones, dists_one = do_counterfactual(ds_ecg, device, config,
                                              tabular_domains, models,
                                              tabular_datasets,
                                              tabular_domain_labels,
                                              quantiles,
                                              counterfactual_col=counterfactual_col,
                                              dataset_to_use=dataset_to_use)
        ##the first pass is the one utilizing the prior
        scale1 = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/dist/scale_one.csv").set_index(
            "Unnamed: 0")
        loc1 = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/dist/loc_one.csv").set_index(
            "Unnamed: 0")
        counter = 0
        prebatched_samples = list(dists_one[matching_dataset_name].keys())
        sample_set = [y for x in prebatched_samples for y in x]
        total_samples = len(sample_set)
        kl_pairwise = np.zeros((total_samples, len(quantiles), 25))
        lower_window = 0
        print(f"distributions generated for {total_samples} people")
        ecg_pars = make_par_labels()
        place = f"/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/counterfactual/{counterfactual_col}"
        if not os.path.isdir(place):
            os.makedirs(place)
        for i, person_chunk1 in enumerate(prebatched_samples):
            slice_size = len(person_chunk1)
            true_dist = Normal(torch.from_numpy(loc1.loc[person_chunk1, :].values).to(device), torch.from_numpy(scale1.loc[person_chunk1, :].values).to(device))
            ##this works through broadcasting
            for quantile, dist in dists_one[matching_dataset_name][person_chunk1].items():
                kl_pairwise[lower_window:(lower_window + slice_size), quantile, :] = torch.distributions.kl.kl_divergence(dist, true_dist).cpu().detach().numpy()
                counter += len(person_chunk1)
                print(100*counter/(total_samples*len(quantiles)))
            lower_window += slice_size
        corresponding_diff = np.zeros((total_samples, len(quantiles)))
        for i, person in enumerate(sample_set):
            true_val = tabular_datasets[matching_dataset_name].df.loc[person, counterfactual_col]
            for j, counterfactual_val in enumerate(quantiles):
                corresponding_diff[i, j] = counterfactual_val - true_val
    ##fragility computationre
        pres = {}
        for j, ecg_par in enumerate(ecg_pars):
            for i in range(len(kl_pairwise)):
                pres[i] = np.corrcoef(kl_pairwise[i, :, j], corresponding_diff[i, :])[0, 1]
            fragility = pd.DataFrame(pd.Series(pres))
            fragility["RegistrationCode"] = sample_set
            fragility = pd.DataFrame(fragility)
            fragility = fragility.set_index("RegistrationCode").dropna()
            fragility.to_csv(f"{place}/{ecg_par}.csv")
    fragility_figures= False
    if fragility_figures:
        old_fragility = False
        if old_fragility:
            ##for the counterfactual figures
            fragility = pd.concat([pd.read_csv(
                "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/counterfactual/summed_emb_dim/" + x).set_index(
                "RegistrationCode").rename(mapper={"0": x.split(".csv")[0]}, axis=1) for x in
                                   os.listdir("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/counterfactual/")],
                                  join="outer", axis=1)
            fragility = fragility.astype(np.float64)
            specific_fragility = pd.concat(list(map(lambda i: fragility[list(set(list(fragility.columns)).intersection(
                set(list(map(lambda x: x + "_fragility", tabular_datasets[i].df.columns)))))].mean(axis=1),
                                                    list(range(len(tabular_datasets))))), axis=1)
            specific_fragility.columns = list(
                map(lambda x: str(x).split(".")[-1].split("'")[0].split("Loader")[0] if x != "iglu" else str(x),
                    tabular_domain_labels))
            common_people = list(set(list(comparator.index)).intersection(list(fragility.index)))
            ##use only the 20 most corelated vars to age
            ##
            corr = fragility.merge(features_df["age"], left_index=True, right_index=True).corr()[
                "age"].sort_values().dropna()
            corr.hist(bins=100)
            plt.title("individual fragility features vary with age")
            plt.xlabel("rho")
            plt.show()
            features_positive = list(corr.loc[corr >= 0.1].index.values)
            features_positive.remove("age")
            features_negative = list(corr.loc[corr <= -0.1].index.values)
            features_neither = list(corr.loc[np.abs(corr) < 0.1].index.values)
            features_positive_base = list(map(lambda x: x.split("_fragility")[0], features_positive))
            features_negative_base = list(map(lambda x: x.split("_fragility")[0], features_negative))

        new_fragility = False
        if new_fragility:
            dirs = os.listdir("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/counterfactual/")
            dirs.remove("summed_emb_dim")
            frags = {}
            for y in dirs:
                frag_temp = pd.concat([pd.read_csv(
                    f"/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/counterfactual/{y}/" + x).set_index(
                    "RegistrationCode").rename(mapper={"0": f"{x.split('.csv')[0]}"}, axis=1) for x in
                                       os.listdir(
                                           f"/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/counterfactual/{y}/")],
                                      join="outer", axis=1)
                frag_temp = frag_temp.astype(np.float64)
                frags[y] = (frag_temp)
        total_frags = frags
        more_fragility = False
        if more_fragility:
            reses = {}
            for disease in ["CA23"]:
                could_be_better = {}
                than = {}
                i = 0
                for k, v in frags.items():
                    try:
                        same_people = list(set(list(v.index)).intersection(list(comparator.index)))
                        frag = v.loc[same_people]
                        medical_data = pd.DataFrame(comparator.loc[same_people, k])
                        both = medical_data.merge(frag, left_index=True, right_index=True)
                        alone = make_disease_pred(medical_data, disease)
                        together = make_disease_pred(both, disease)
                        print(alone, together)
                        than[k] = alone
                        could_be_better[k] = together
                    except Exception:
                        pass
                pred_res = pd.concat([pd.Series(than).rename({0: "baseline"}),
                                      pd.Series(could_be_better).rename({0: "baseline_fragility_together"})], axis=1)
                pred_res.columns = ["baseline", "ours"]
                reses[disease] = pred_res
            agg_wins = pd.Series({k: (v["ours"] > v["baseline"]).sum() / len(v) for k, v in reses.items()}).sort_values(
                ascending=False)
            top_diseases = list(set(list(agg_wins.loc[agg_wins > 0.95].index) + ["5A00"]))

            ##finding the best predictor for each comparator and disease
            same_people_for_this = list(
                set(list(z_first_all_mech_ecg["BodyMeasures_time"].index)).intersection(list(comparator.index)))
            print(f"Are they the same? {set(same_people_for_this) == set(same_people)}")
            best_predictors = {}
            for a_disease in ["CA23"]:
                print(a_disease)
                total_res = {}
                i = 0
                for a_feature in comparator.columns:
                    print(100 * i / len(comparator.columns))
                    i += 1
                    try:
                        medical_data = pd.DataFrame(comparator.loc[same_people, a_feature])
                        total_res[a_feature] = make_disease_pred(medical_data, a_disease)
                    except np.linalg.LinAlgError:
                        pass
                best_pred = pd.Series(total_res).sort_values(ascending=False)
                best_predictors[a_disease] = [best_pred.index.values[0], best_pred.iloc[0]]

            r_fragility = {}
            r_baseline = {}
            r_both = {}
            for k, v in frags.items():
                if k in ["femur_rig_upper_neck_bmd", "cgm_mod", "cgm_range", "femur_lower_neck_mean_bmd",
                         "hand_grip_right", "spine_l1_l4_bmd", "r_ankle_pressure", "cgm_auc", "weight",
                         "standing_one_min_blood_pressure_diastolic"]:
                    try:
                        same_people = list(set(list(v.index)).intersection(list(comparator.index)))
                        r_fragility[k] = cv(v.loc[same_people], features_df, "age")[0].corr()["preds"]["true"]
                        medical_data = pd.DataFrame(comparator.loc[same_people, k])
                        both = medical_data.merge(v.loc[same_people], left_index=True, right_index=True)
                        r_baseline[k] = cv(medical_data, features_df, "age")[0].corr()["preds"]["true"]
                        r_both[k] = cv(both, features_df, "age")[0].corr()["preds"]["true"]
                    except ValueError:
                        pass
            rs_fragility = pd.Series(r_fragility)
            rs_baseline = pd.Series(r_baseline)
            rs_both = pd.Series(r_both)
            all_r = pd.concat([rs_baseline, rs_fragility, rs_both], axis=1).dropna()
            all_r = all_r.loc[(all_r > 0.35).any(axis=1), :]
            all_r.columns = ["feature_only", "fragility_only", "feature+fragility"]
            all_r_plot = pd.melt(all_r.reset_index(), id_vars="index").sort_values("value", ascending=False)
            sns.set(font_scale=2)
            fig, ax = plt.subplots(1, 1, figsize=(15, 15))
            sns.barplot(data=all_r_plot, x="index", y="value", hue="variable", ax=ax,
                        palette=sns.color_palette(["#003f5c", "#7a5195", "#000000"]))
            ax.set_xticklabels(rotation=45, ha='right',
                               labels=["bmd", "ankle_BP", "spine_bmd", "cgm_auc", "grip", "cgm_range", "standing_bp_d"])
            ax.set_xlabel("")
            ax.set_ylabel("Pearson r of age prediction")
            ax.set_title("Fragility outpredicts age relative to clinical features")
            plt.show()