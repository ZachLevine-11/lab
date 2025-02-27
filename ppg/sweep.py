import wandb
from all_to_ecg import do
import os

os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"]= "5000"

# 2: Define the search space
sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "minimize", "name": "batch_eval_loss"},
    "parameters": {
        "do_eval": {"values": [True]},
        "weight_sharing": {"values": [False]},
        "hidden_size": {"values": [60]},
        "num_layers": {"values": [2]},
        "encoder_dropout_prob": {"values": [0]},
        "temperature": {"values": [1]},
        "alpha_dtw": {"values": [1/2]},
        "mse_alpha": {"values": [100, 0]},
        "eval_every": {"values": [75]},
        "normalize_dtw_loss": {"values": [False, True]},
        "total_train_n": {"values": [2000]},
        "total_eval_n": {"values": [250]},
        "plot": {"values": [False]},
        "lr": {"values": [1e-3]},
        "batch_size": {"values": [10]},
        "first_to_second_alpha": {"values": [0, 1/2]},
        "length_sim_range": {"values": [[400, 401]]},
        "length_window_range": {"values": [[4000, 4001]]},
        "gamma": {"values": [0.1]},
        "epochs": {"values": [10]},
        "kl_alpha": {"values": [20]},
    }
}

def do_sweep():
    wandb.init(project="mech-interp")
    do(wandb.config)

wandb.login()
#sweep_id = wandb.sweep(sweep=sweep_configuration, project="mech-interp")
wandb.agent(sweep_id="dq34k1fx", function= do_sweep, project="mech-interp")