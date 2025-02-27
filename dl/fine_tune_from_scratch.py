import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from tcn import SmallPredictor, both
from torch import nn as nn
import wandb
from ecg import batch_manager
from wandb_main import make_age_df
from ECGDataset import ecg_dataset_pheno
from LabData.DataLoaders.SubjectLoader import SubjectLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib import pyplot as plt
from general_pred import make_diagnoses_df
import cv2
from wandb_main import plot_ecg

def get_true_vals(ids, df, device):
    temp = df.loc[ids]
    return torch.Tensor(temp.to_numpy()).to(device)

class EarlyStopperFineTuning:
    def __init__(self, max_r, patience=1, min_delta=0, saveName_prefix = "checkpoint_finetuning"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_r = max_r
        self.saveName_prefix = saveName_prefix
        self.saved_model_path = "/net/mraid20/export/jasmine/zach/cross_modal/saved_models/"

    def early_stop(self, curr_val_r, pretrained_ecgnet, proj_net, optimizer_ecgnet, optimizer_mlp):
        if curr_val_r > self.max_r:
            self.max_r = curr_val_r
            self.counter = 0
            checkpoint_ecg = {
                'model': pretrained_ecgnet,
                'optimizer': optimizer_ecgnet,
                "max_r": curr_val_r}
            torch.save(checkpoint_ecg, self.saved_model_path + self.saveName_prefix + "_ecg.pth")
            checkpoint_proj = {'model': proj_net,
                               'optimizer': optimizer_mlp}
            torch.save(checkpoint_proj, self.saved_model_path + self.saveName_prefix + "_mlp.pth")
        elif curr_val_r < (self.max_r - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_ecg_supervised(proj_net,
                use_checkpoint,
                checkpoint_name,
                ds_train_ecg,
                device,
                train_people, validation_people, test_people, ones_dir, twos_dir, remake_windows_from_scratch,
                loss,
                config,
                df,
                skipLinear,
                doFT,
                saveName_prefix,
                mode = "r",
                visualize = None,
                restore_lr = True):
    print("Using " + str(device))
    torch.cuda.empty_cache()
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    optimizer = config["optim"]
    lr_mlp = config["learning_rate_if_from_scratch"]
    early_stopper = EarlyStopperFineTuning(max_r = 0, patience = 5, saveName_prefix= saveName_prefix)
    if use_checkpoint:
        print("reading in saved model")
        saved_model_path = "/net/mraid20/export/jasmine/zach/cross_modal/saved_models/"
        checkpoint = torch.load(saved_model_path + checkpoint_name)
        pretrained_ecgnet = checkpoint['model']
        if torch.cuda.device_count() < 2:
            try:
                pretrained_ecgnet = pretrained_ecgnet.module ##if we don't have multiple GPUs, strip away the DataParallel from the saved SSL model if there is one
            except AttributeError:
                ##then there is nonel data parallel object on the saved SSL model
                pass
    else:
        pretrained_ecgnet = both().to(device)
    if restore_lr:
        lr_ecg = checkpoint["optimizer"].param_groups[0]['lr']/3
    else:
        lr_ecg = lr_mlp
    optimizer_ecgnet = optimizer(pretrained_ecgnet.parameters(), lr=lr_ecg)
    optimizer_mlp = optimizer(proj_net.parameters(), lr=lr_mlp)
    scheduler_ecg = ReduceLROnPlateau(optimizer_ecgnet, patience=2)
    scheduler_mlp = ReduceLROnPlateau(optimizer_mlp, patience=2)
    for epoch in range(epochs):
        emb_stores_apt_one = []
        true_vals_stores_apt_one = []
        ids_all_one = []
        print("Starting epoch: ", epoch)
        int_ids_batched = np.array_split(np.random.choice(train_people, size = len(train_people), replace = False), len(train_people)//batch_size)
        pretrained_ecgnet.train()
        proj_net.train()
        i = 0
        for batch in int_ids_batched:
            batched_ecgs_raw_apt_one, batched_ecgs_raw_apt_two, ids_one, ids_two = batch_manager(batch, ds_train_ecg, device,
                                                                                                 ones_dir, twos_dir, remake_windows_from_scratch)
            true_vals = get_true_vals(ids_one, df, device)
            ecg_embeddings_apt_one = proj_net(pretrained_ecgnet(batched_ecgs_raw_apt_one, skipLinear))
            train_loss = loss(ecg_embeddings_apt_one.squeeze(), true_vals.squeeze())
            wandb.log({"train_loss_mini_batch": float(train_loss)}),
            train_loss.backward()
            optimizer_ecgnet.step()
            optimizer_ecgnet.zero_grad()
            optimizer_mlp.step()
            optimizer_mlp.zero_grad()
            print(100 * i/len(int_ids_batched), "% finished epoch, train", sep = "")
            i += 1
        j = 0
        if visualize == "grad_cam":
            gradients = pretrained_ecgnet.get_activations_gradient()
            gradients = torch.mean(gradients,dim=2)  ##global average pool the channels to get weights for each channel in the activation map
            activations = pretrained_ecgnet.get_activations(batched_ecgs_raw_apt_one).detach()
            for i in range(activations.shape[1]):
                activations[:, i, :] *= gradients[:, i].reshape(-1, 1).expand(-1, activations[:, i, :].shape[1]) ##weight all points for each of 192 channels by the gradient weight
            heatmap = torch.mean(activations,dim=1).cpu().squeeze()  ##now average all channels to get a single dimensional gradient heatmap for the channels
            heatmap = np.maximum(heatmap, 0)
            for j in range(4):
                temp = heatmap[j, :, ].reshape(1, 1, -1)
                temp = temp/torch.max(temp)
                big_heatmap = nn.Upsample((batched_ecgs_raw_apt_one.shape[2]), mode="linear")(temp).flatten().numpy()
                plot_ecg(pd.DataFrame(batched_ecgs_raw_apt_one[j, :, :].cpu().numpy()),
                         fname="/home/zacharyl/Desktop/" + saveName + + str(epoch) + "_" + str(j) + ".jpg",
                         colours=big_heatmap)
        elif visualize == "attention":
            prev = pretrained_ecgnet(batched_ecgs_raw_apt_one, stop_after_eleven = True)
            output, scores = pretrained_ecgnet.transformer.encoder_stack_twelve.layers[0].self_attn(prev, prev, prev)
            for j in range(4):
                temp = scores[j, 0, 1:].squeeze().cpu().detach().reshape(1, 1, 58)
                big_heatmap = nn.Upsample((batched_ecgs_raw_apt_one.shape[2]), mode="linear")(temp).flatten().numpy()

                plot_ecg(pd.DataFrame(batched_ecgs_raw_apt_one[j, :, :].cpu().numpy()),
                     fname="/home/zacharyl/Desktop/" + saveName + str(epoch) + "_" + str(j) + ".jpg",
                     colours=big_heatmap)
        else:
            pass
        with torch.no_grad():
            pretrained_ecgnet.eval()
            proj_net.eval()
            int_ids_batched_eval = np.array_split(
                np.random.choice(validation_people, size=len(validation_people), replace=False), max((4*len(validation_people)) // batch_size, 1))
            epoch_eval_loss = 0
            for validation_batch in int_ids_batched_eval:
                batched_ecgs_raw_apt_one_eval, batched_ecgs_raw_apt_two_eval, ids_one_eval, ids_two_eval = batch_manager(
                            validation_batch, ds_train_ecg, device, ones_dir, twos_dir,
                remake_windows_from_scratch)
                ecg_embeddings_apt_one_eval = proj_net(pretrained_ecgnet(batched_ecgs_raw_apt_one_eval, skipLinear))
                emb_stores_apt_one.append(ecg_embeddings_apt_one_eval.detach().cpu())
                ids_all_one += ids_one_eval
                true_vals_eval = get_true_vals(ids_one_eval, df, device)
                true_vals_stores_apt_one.append(true_vals_eval.cpu())
                eval_loss = loss(ecg_embeddings_apt_one_eval.squeeze(), true_vals_eval.squeeze())
                epoch_eval_loss += float(np.real(eval_loss)) / len(int_ids_batched_eval)
                print(100 * j / len(int_ids_batched_eval), "% finished epoch, validation", sep = "")
                j += 1
            ##pearson r calculation over whole eval set
            if mode == "r":
                metric = pearsonr(torch.cat(emb_stores_apt_one).numpy().flatten(), torch.cat(true_vals_stores_apt_one).numpy().flatten())
                wandb.log({"eval_r": metric[0]})
                wandb.log({"eval_P": metric[1]})
            elif mode == "c":
                metric = [roc_auc_score(torch.cat(true_vals_stores_apt_one).numpy().flatten(), torch.cat(emb_stores_apt_one).numpy().flatten())]
                wandb.log({"eval_AUC": metric[0]})
            print("eval loss per epoch: ", epoch_eval_loss)
            wandb.log({"eval_loss": epoch_eval_loss})
            scheduler_ecg.step(epoch_eval_loss)
            scheduler_mlp.step(epoch_eval_loss)
            if early_stopper.early_stop(metric[0], pretrained_ecgnet, proj_net, optimizer_ecgnet, optimizer_mlp):
                j = 0
                print("reading in saved model")
                saved_model_path = "/net/mraid20/export/jasmine/zach/cross_modal/saved_models/"
                checkpoint_ecg = torch.load(saved_model_path + saveName_prefix + "_ecg.pth")
                checkpointed_ecgnet = checkpoint_ecg['model']
                checkpoint_proj = torch.load(saved_model_path + saveName_prefix + "_mlp.pth")
                checkpoint_proj_net = checkpoint_proj["model"]
                checkpointed_ecgnet.eval()
                checkpoint_proj_net.eval()
                emb_stores_apt_one = []
                true_vals_stores_apt_one = []
                ids_all_one = []
                int_ids_batched_test = np.array_split(
                    np.random.choice(test_people, size=len(test_people), replace=False),
                    max((4 * len(test_people)) // batch_size, 1))
                epoch_test_loss = 0
                for test_batch in int_ids_batched_test:
                    batched_ecgs_raw_apt_one_test, batched_ecgs_raw_apt_two_test, ids_one_test, ids_two_test = batch_manager(
                        test_batch, ds_train_ecg, device, ones_dir, twos_dir,
                        remake_windows_from_scratch)
                    ecg_embeddings_apt_one_test = checkpoint_proj_net(checkpointed_ecgnet(batched_ecgs_raw_apt_one_test, skipLinear))
                    emb_stores_apt_one.append(ecg_embeddings_apt_one_test.detach().cpu())
                    ids_all_one += ids_one_test
                    true_vals_test = get_true_vals(ids_one_test, df, device)
                    true_vals_stores_apt_one.append(true_vals_test.cpu())
                    test_loss = loss(ecg_embeddings_apt_one_test.squeeze(), true_vals_test.squeeze())
                    epoch_test_loss += float(np.real(test_loss)) / len(int_ids_batched_eval)
                    print(100 * j / len(int_ids_batched), "% finished epoch, test", sep="")
                    j += 1
                ##pearson r calculation over whole test set
                if mode == "r":
                    metric = pearsonr(torch.cat(emb_stores_apt_one).numpy().flatten(),
                                        torch.cat(true_vals_stores_apt_one).numpy().flatten())
                    wandb.log({"test_r": metric[0]})
                    wandb.log({"test_P": metric[1]})
                elif mode == "c":
                    metric = [roc_auc_score(torch.cat(true_vals_stores_apt_one).numpy().flatten(), torch.cat(emb_stores_apt_one).numpy().flatten())]
                    wandb.log({"test_AUC": metric[0]})
                print("test loss per epoch: ", epoch_test_loss)
                wandb.log({"test_loss": epoch_test_loss})
                return None

def make_fine_tune_age_gender_dfs(splits_path, emb_df, ds):
    train_people = list(pd.read_csv(splits_path + "train.csv")["0"])
    test_people = list(pd.read_csv(splits_path + "test.csv")["0"])
    validation_people = list(pd.read_csv(splits_path + "validation.csv")["0"])
    subs = SubjectLoader().get_data(study_ids=ds.id_list).df
    subs = subs.reset_index(["Date"], True).dropna(subset=["yob", "month_of_birth"])
    genderSeries = pd.DataFrame(subs["gender"]).dropna().astype(float)
    genderSeries = genderSeries.loc[~genderSeries.index.duplicated(keep="first"), :]
    ##some people have NA gender, we dropped them from the prediction df, so drop them from the train/test split too
    train_people = list(set(train_people).intersection(set(genderSeries.index.values)))
    validation_people = list(set(validation_people).intersection(set(genderSeries.index.values)))
    test_people = list(set(test_people).intersection(set(genderSeries.index.values)))
    age_df_raw = make_age_df(emb_df, ds)
    age_df_here = age_df_raw.reset_index("Date", drop=False).sort_values("Date").drop(["Date"], axis=1)
    age_df_here = age_df_here.loc[~age_df_here.index.duplicated(keep="first"), :]
    scaler_Y = StandardScaler().fit(np.array(age_df_here.loc[train_people]).reshape(-1, 1))
    age_normalized = pd.DataFrame(scaler_Y.transform(np.array(age_df_here).reshape(-1, 1)).reshape(-1))
    age_normalized["RegistrationCode"] = age_df_here.index
    age_normalized = age_normalized.set_index("RegistrationCode")
    return genderSeries, age_normalized, train_people, validation_people, test_people


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    base_dir = "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/heartbeats/"
    ones_dir = base_dir + "ones/"
    twos_dir = base_dir + "twos/"
    splits_path = "/net/mraid20/export/jasmine/zach/cross_modal/splits/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = list(pd.read_csv(splits_path + "files.csv")["0"])
    ds = ecg_dataset_pheno(files=files)
    emb_df = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/no_ft/one_no_linear.csv").set_index(["RegistrationCode", "Date"])
    genderSeries, age_normalized, train_people, validation_people, test_people = make_fine_tune_age_gender_dfs(splits_path, emb_df, ds)
    config_here = dict(
        epochs=500,
        temp_0=0.07,
        batch_size=100,
        learning_rate_if_from_scratch=5e-5,
        optim=torch.optim.AdamW,
        stop_after=None,
        max_test_steps=None,
        numTestTotal=305)
    ## grad_cam can't be run from shell, only in python console (i.e pycharm)
    ##for some reason, grad cam not work at all in the shell (interpolation packages don't load for some reason outside of pycharm)
    ##so run within the python console
    checkpoint = False
    if checkpoint:
        saveName = "ssl_"
    else:
        saveName = "scratch_"
    wandb.init()
    age = False
    if age:
        train_ecg_supervised(SmallPredictor(768).to(device),
                         use_checkpoint=checkpoint,
                         checkpoint_name="SSL/cur.pth",
                         ds_train_ecg=ds,
                         device=device,
                         train_people=train_people, validation_people=validation_people, test_people=test_people,
                         ones_dir=ones_dir, twos_dir=twos_dir,
                         remake_windows_from_scratch=True,
                         loss=nn.CrossEntropyLoss(),
                         config=config_here,
                         df=genderSeries,
                         skipLinear=True,
                         doFT=False,
                         saveName_prefix="supervised/" + saveName + "gender",
                         mode="c",
                         visualize=None,
                         restore_lr=False)
    gender = False
    if gender:
        train_ecg_supervised(SmallPredictor(768).to(device),
                         use_checkpoint=checkpoint,
                         checkpoint_name="SSL/cur.pth",
                         ds_train_ecg=ds,
                         device=device,
                         train_people=train_people, validation_people=validation_people, test_people=test_people,
                         ones_dir=ones_dir, twos_dir=twos_dir,
                         remake_windows_from_scratch=True,
                         loss=nn.MSELoss(),
                         config=config_here,
                         df=age_normalized,
                         skipLinear=True,
                         doFT=False,
                         saveName_prefix="supervised/" + saveName + "age",
                         mode="r",
                         visualize=None,
                         restore_lr=False)
    other = True
    if other:
        df_diag = make_diagnoses_df(pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_embeddings/no_ft/one_no_linear.csv").set_index(["RegistrationCode", "Date"]), ds.id_list)
        df_obesity = df_diag.loc[:, "BA00"]
        train_ecg_supervised(SmallPredictor(768).to(device),
                         use_checkpoint=checkpoint,
                         checkpoint_name="SSL/cur.pth",
                         ds_train_ecg=ds,
                         device=device,
                         train_people=train_people, validation_people=validation_people, test_people=test_people,
                         ones_dir=ones_dir, twos_dir=twos_dir,
                         remake_windows_from_scratch=True,
                         loss=nn.CrossEntropyLoss(),
                         config=config_here,
                         df=df_obesity,
                         skipLinear=True,
                         doFT=False,
                         saveName_prefix="supervised/" + saveName + "hypertension",
                         mode="c",
                         visualize= "grad_cam",
                         restore_lr=False)

