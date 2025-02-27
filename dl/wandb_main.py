import numpy as np
import pandas as pd
import torch
import wandb
from ECGDataset import ecg_dataset_pheno
from ecg import train_ecg_contrastive, batch_manager
import os
import pickle
from LabData.DataLoaders.SubjectLoader import SubjectLoader
from LabData.DataLoaders.ECGTextLoader import ECGTextLoader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
import seaborn as sns
from matplotlib import pyplot as plt

##give these the test data
def generate_eye_embeddings(eye_dataset, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using " + str(device))
    if device == "cuda":
        torch.cuda.empty_cache()
    print("Using " + str(device))
    torch.cuda.empty_cache()
    embeddings = {}
    i = 0
    for tenk_id in eye_dataset.ids:
        embeddings[tenk_id] = pd.Series(*model(eye_dataset.get_image(tenk_id).to(device)[None, :, :, :], onlyEmbeddings = True, latent_shape= (115, 74), orig_shape=(3000,3000)).detach().cpu())
        print(i)
        i += 1
    embeddings = pd.DataFrame(embeddings).T
    embeddings.index.name = "RegistrationCode"
    return embeddings

def do_dating(df, dates):
    df.index.name = "RegistrationCode"
    df = df.groupby("RegistrationCode").mean()
    df["Date"] = dates
    df = df.set_index("Date", append = True)
    return df

def generate_ecg_embeddings(ecg_dataset, test_people, model, device, skipLinear = False, remake_windows_from_scratch = True):
    print("Using " + str(device))
    if device == "cuda":
        torch.cuda.empty_cache()
    print("Using " + str(device))
    torch.cuda.empty_cache()
    embeddings_one = {}
    embeddings_two = {}
    dates_one = []
    dates_two = []
    i = 0
    model.eval()
    for test_person in test_people:
        x1, x2, ids_one_batch, ids_two_batch = batch_manager([test_person], ecg_dataset, device, ones_dir=ones_dir, twos_dir = twos_dir, remake_windows_from_scratch = remake_windows_from_scratch)
        x1 = x1.to(device)
        dates_one.append(ecg_dataset.get_date(test_person)[0])
        embeddings_one[test_person] = pd.Series(model(x1, skipLinear=skipLinear).detach().cpu().mean(axis=0))
        print(i)
        i += 1
        if x2 is not None:
            x2 = x2.to(device)
            dates_two.append(ecg_dataset.get_date(test_person)[1])
            embeddings_two[test_person] = pd.Series(model(x2, skipLinear=skipLinear).detach().cpu().mean(axis=0))
    embeddings_one = do_dating(pd.DataFrame(embeddings_one).T, dates_one)
    embeddings_two = do_dating(pd.DataFrame(embeddings_two).T, dates_two)
    embeddings_all = pd.concat([embeddings_one, embeddings_two])
    newDate = list(pd.to_datetime(embeddings_all.index.get_level_values(1)))
    embeddings_all = embeddings_all.reset_index(["Date"], drop=True)
    embeddings_all["Date"] = newDate
    embeddings_all = embeddings_all.set_index(["Date"], append = True)
    #windows are averaged per person per visit, but not across visits
    return embeddings_all, embeddings_one, embeddings_two

def make_age_df(emb_df, ds):
    subs = SubjectLoader().get_data(study_ids=ds.id_list).df
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

def load_model(saved_model_path, fname):
    ##reload models from disk
    checkpoint = torch.load(saved_model_path + fname)
    e = checkpoint['model']
    if torch.cuda.device_count() < 2:
        try:
            e = e.module  ##if we don't have multiple GPUs, strip away the DataParallel from the saved SSL model if there is one
        except AttributeError:
            ##then there is non data parallel object on the saved SSL model
            pass
    return e

def make_ecgtext_table(ds):
    res = []
    ecg_all = ECGTextLoader().get_data(study_ids=ds.id_list).df
    backup_id = ecg_all.index.get_level_values(0).copy()
    for var in ecg_all.columns:
        try:
            ecg_all[var] =  ecg_all[var].astype(np.float64)
        except ValueError:
            ecg_all.drop(var, axis = 1, inplace = True)
    imputer_X = KNNImputer().fit(ecg_all)
    ecg_all = imputer_X.transform(ecg_all)
    ecg_all = pd.DataFrame(ecg_all)
    ecg_all["RegistrationCode"] = backup_id
    ecg_all.set_index("RegistrationCode", inplace = True)
    ecg_all = ecg_all.loc[~ecg_all.index.get_level_values(0).duplicated(keep = "first"),:]
    subs = SubjectLoader().get_data(study_ids=ds.id_list).df
    subs = subs.loc[~subs.index.duplicated(keep = "first"),:]
    for study_id in ds.id_list:
        ecg = ecg_all.loc[list(set(subs.loc[subs["StudyTypeID"] == study_id].index.get_level_values(0)).intersection(set(ecg_all.index.get_level_values(0)))),:]
        if len(ecg) > 0:
            ecg = ecg
            means = ecg.mean()
            sds = ecg.std()
            table = pd.Series(list(map(lambda i: str(np.round(means[i], decimals = 2)) + " " + "(" + str(np.round(sds[i], decimals = 2)) + ")", list(range(len(sds))))))
            table = pd.DataFrame(table)
            table["feature"] = means.index
            table = table.set_index("feature")
            table.columns = [study_id]
            res.append(table)
    res_concat  = pd.concat(res, axis = 1)
    res_concat["feature"] = means.index
    res_concat.set_index("feature", inplace = True)
    return res_concat

def make_baseline_table(ids = [10, 1001, 1005, 1006, 1007, 1008, 1009, 1010]):
    '''
    - Age at baseline (months)
    - Waist Circumference (cm)
    - BMI
    - Height
    - Systolic Sitting BP
    - Diastolic Sitting BP
    '''
    subs = SubjectLoader().get_data(study_ids = ids).df
    subs = subs.loc[~subs.index.get_level_values(0).duplicated(keep = "first"),:]
    body = BodyMeasuresLoader().get_data(study_ids = ids).df
    body = body.loc[~body.index.get_level_values(0).duplicated(keep = "first"),:]
    res = []
    for study_id in ids:
        body_temp = body.loc[list(set(subs.loc[subs["StudyTypeID"] == study_id].index.get_level_values(0)).intersection(set(body.index.get_level_values(0)))),:]
        body_temp = body_temp[["bmi", "waist", "height", "sitting_blood_pressure_systolic", "sitting_blood_pressure_diastolic"]]
        body_temp = body_temp
        means = body_temp.mean()
        sds = body_temp.std()
        table = pd.Series(list(map(lambda i: str(np.round(means[i], decimals = 2)) + " " + "(" + str(np.round(sds[i], decimals = 2)) + ")", list(range(len(sds))))))
        table = pd.DataFrame(table)
        table["feature"] = means.index
        table = table.set_index("feature")
        table.columns = [study_id]
        res.append(table)
    res_concat  = pd.concat(res, axis = 1)
    return res_concat


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

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    ##os environment variables help with debugging but break nn.DataParallel, which allows us to use split over multiple gpus
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #for debugging
    #os.environ["TORCH_USE_CUDA_DSA"] = "1" #for debugging
    #https://discuss.pytorch.org/t/training-on-the-multi-gpus-but-stuck-in-loss-backward/174554/9

    saved_model_path = "/net/mraid20/export/jasmine/zach/cross_modal/saved_models/"
    splits_path =  "/net/mraid20/export/jasmine/zach/cross_modal/splits/"
    base_dir = "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/heartbeats/"
    ones_dir = base_dir + "ones/"
    twos_dir = base_dir + "twos/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 ## now index, but keep the duplicates in the dataset
    print("Using " + str(device))
    if device == "cuda":
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
   # wandb.login()
    train_config = dict(
        epochs=500,
        temp_0 = 0.07,
        batch_size=100,
        learning_rate=1e-5,
        optim = torch.optim.AdamW,
        warm_up_steps = 10,
        skipLinear = False)
    remake_split = False
    files =  list(pd.read_csv(splits_path + "files.csv")["0"])
    np.random.seed(0)
    torch.manual_seed(0)
    ecg_files = os.listdir(
        "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/Pheno/stas/ecg/waveform/clean/long_format/")
    all_samples_ecg_pheno = pd.Series(
        list(map(lambda x: "10K_" + x.split("ecg__10k__")[-1].split("__")[0], ecg_files))).reset_index().set_index(
        0).index.values
    all_samples_ecg_tenk = set(ECGTextLoader().get_data(study_ids=list(range(100)) + list(
        range(1000, 1011, 1))).df.reset_index().RegistrationCode)  # 10K cohort samples
    all_samples_ecg = list(set(all_samples_ecg_pheno).intersection(
        all_samples_ecg_tenk))  ##keep pheno samples that are 10K cohort samples only
    ecg_fmap = make_ecg_fmap(tenk_id_list=all_samples_ecg, files=ecg_files, from_cache=True)
    ds = ecg_dataset_pheno(all_samples_ecg=all_samples_ecg, fmap=ecg_fmap)
    if remake_split:
        print("generating new splits and saving to file")
        ##genereate and save a split to file
        train_people = list(np.random.choice(list(list(ds.fmap.keys())), int(len(list(list(ds.fmap.keys())))*0.9), replace=False))
        validation_people = list(np.random.choice(list(set(list(list(ds.fmap.keys()))) - set(train_people)), size = 557, replace = False))
        test_people = list(set(list(list(ds.fmap.keys()))) - set(train_people) - set(validation_people))
        ##save the splits to disk so we can resume training later with the same split
        pd.Series(train_people).to_csv(splits_path + "train.csv")
        pd.Series(test_people).to_csv(splits_path + "test.csv")
        pd.Series(validation_people).to_csv(splits_path + "validation.csv")
    else:
        ##reload splits from disk
        print("read in splits saved previously")
        train_people = list(pd.read_csv(splits_path + "train.csv")["0"])
        test_people = list(pd.read_csv(splits_path + "test.csv")["0"])
        validation_people = list(pd.read_csv(splits_path + "validation.csv")["0"])
    ##after one epoch of true on True, can set to False to avoid repeating the same computation of window size
    ##however because loading the files in from disk also takes a long time, this is actually slower with big batches than remaking the windows from scratch each tome
    recount = False
    if recount:
        counts_train = 0
        counts_test = 0
        counts_validation = 0
        for i in range(len(train_people)):
            if len(ds.get_date(train_people[i])) > 1:
                counts_train += 1
        for i in range(len(test_people)):
            if len(ds.get_date(test_people[i])) > 1:
                counts_test += 1
        for i in range(len(validation_people)):
            if len(ds.get_date(validation_people[i])) > 1:
                counts_validation += 1
        print("train set set (", len(train_people) ,") includes ", counts_train, " repeat visits")
        print("test set set (", len(test_people) ,") includes ", counts_test, " repeat visits")
        print("validation set (", len(validation_people) ,") includes ", counts_validation, " repeat visits")
    retrain = True
    remake_windows_from_scratch = True
    saved_model_name = "SSL/new.pth"
    if retrain:
        wandb.init()
        from_scratch = True
        train_ecg_contrastive(from_scratch,
                              train_config,
                              ds,
                              device,
                              train_people, test_people, validation_people, ones_dir, twos_dir, remake_windows_from_scratch, saveName = saved_model_name, to_load_name = saved_model_name)
    make_emb_df = True
    if make_emb_df:
        e = load_model(saved_model_path, saved_model_name)
        embeddings_all, embeddings_one, embeddings_two = generate_ecg_embeddings(ds, set(ds.all_samples.index), e,
                                                                                 device,
                                                                                 skipLinear=True,
                                                                                 remake_windows_from_scratch=remake_windows_from_scratch)

        embeddings_all.to_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/no_ft/all_no_linear.csv")
        embeddings_one.to_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/no_ft/one_no_linear.csv")
        embeddings_two.to_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/no_ft/two_no_linear.csv")
