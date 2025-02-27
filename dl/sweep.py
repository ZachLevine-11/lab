import torch
from tcn import SmallPredictor
from torch import nn as nn
import wandb
from fine_tune_from_scratch import train_ecg_supervised, make_fine_tune_age_gender_dfs
import pandas as pd
from ECGDataset import ecg_dataset_pheno

def dispatch(df, mode, name):
    wandb.init()
    config_here = dict(
        epochs=500,
        temp_0=0.07,
        batch_size=16,
        learning_rate=wandb.config.learning_rate,
        optim=torch.optim.Adam,
        stop_after=None,
        max_test_steps=None,
        numTestTotal=305)
    base_dir = "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/heartbeats/"
    ones_dir = base_dir + "ones/"
    twos_dir = base_dir + "twos/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "r":
        loss = nn.MSELoss()
    else:
        loss = nn.CrossEntropyLoss()
    train_ecg_supervised(SmallPredictor(192).to(device),
                         use_checkpoint=True,
                         checkpoint_name="SSL/NoFT_100_scheduler-1e-5_adamw.pth",
                         ds_train_ecg=ds,
                         device=device,
                         train_people=train_people, validation_people=validation_people,
                         test_people=test_people, ones_dir=ones_dir, twos_dir=twos_dir,
                         remake_windows_from_scratch=True,
                         loss=loss,
                         config=config_here,
                         df=df,
                         skipLinear=True,
                         doFT=False,
                         saveName_prefix="supervised/sweep_" + name + "_",
                         mode=mode)


splits_path = "/net/mraid20/export/jasmine/zach/cross_modal/splits/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
genderSeries, age_normalized, train_people, validation_people, test_people = make_fine_tune_age_gender_dfs(splits_path)

files = list(pd.read_csv(splits_path + "files.csv")["0"])
ds = ecg_dataset_pheno(files=files)

sweep_config = {
    'method': 'random',
    'metric': {
            'name': 'test_r',
            'goal': 'maximize'
    },
        'parameters': {
            "learning_rate": {
                "values": [5e-6]
            }
        }
    }

sweep_id = wandb.sweep(sweep_config, project="SSL-age-AdamW-1e-5-pretrained")
wandb.agent(sweep_id, function = lambda: dispatch(df = age_normalized, mode = "r", name = "age"))
