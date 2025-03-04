import wandb
from all_to_ecg import do
import os
import numpy as np

os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"]= "5000"

# 2: Define the search space
sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "minimize", "name": "batch_eval_loss"},
    "parameters": {
        "mechanistic": {"values": [True]},
        "do_integration": {"values": [True]},
        "do_eval": {"values": [True]},
        "load_models": {"values": [False]},
        "plot_every": {"values": [1]},
        "make_new_split": {"values": [True]},
        "weight_sharing": {"values": [False]},
        "hidden_size": {"values": [3]},
        "num_layers": {"values": [1]},
        "dt": {"values": [np.log(1)]},
        "warmup_steps": {"values": [0]},
        "encoder_dropout_prob": {"values": [0]},
        "temperature": {"values": [1]},
        "alpha_dtw": {"values": [0]},
        "mse_alpha": {"values": [100]},
        "jac_alpha": {"values": [2e-2]},
        "eval_every": {"values": [250]},
        "normalize_dtw_loss": {"values": [False]},
        "total_eval_n": {"values": [200]},
        "total_test_n": {"values": [7580]},
        "length_sim": {"values": [1000]},
        "length_window": {"values": [4000]},
        "plot": {"values": [True]},
        "lr": {"values": [5e-2]},
        "batch_size": {"values": [100]},
        "first_to_second_alpha": {"values": [0]},
        "gamma": {"values": [0.1]},
        "epochs": {"values": [250]},
        "kl_alpha1": {"values": [0]},
        "kl_alpha2": {"values": [1.5]},
        "loader_set": {"values": [7]},
    }
}

def do_sweep():
    wandb.init(project="mech-interp")
    do(wandb.config)

if __name__ == "__main__":
    wandb.login()
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="mech-interp")
    wandb.agent(sweep_id=sweep_id, function= do_sweep, project="mech-interp")