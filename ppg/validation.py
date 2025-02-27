#!/usr/bin/env python

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import functional
from tslearn.metrics import SoftDTWLossPyTorch

##############################################################################
# (1) Your model definitions (Mechanistic + BlackBox) - adapted from your code
##############################################################################

class ConvLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, channel_last=True):
        super().__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1, bias=bias)
        self.channel_last = channel_last

    def forward(self, x):
        if self.channel_last:
            x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
            x = self.conv(x)
            x = x.transpose(1, 2)
        else:
            x = self.conv(x)
        return x

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric):
        score = -metric  # We assume lower is better (like a loss)
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopper: No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class BlackBoxAutoencoder(nn.Module):
    def __init__(
        self,
        encoding_size,
        encoder_input_size,
        hidden_size,
        num_layers,
        encoder_dropout_prob,
        num_baseline_features,
        decoder_dropout_prob,
        decoder_input_size
    ):
        super().__init__()
        self.drop = nn.Dropout1d(p=encoder_dropout_prob)
        self.encoding_size = encoding_size
        self.encoder_lstm = nn.LSTM(
            input_size=12,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            dropout=encoder_dropout_prob,
            bidirectional=True
        )
        self.non_seq_proj = nn.Linear(hidden_size * num_layers * 2, self.encoding_size)
        self.decoder_hidden_size = hidden_size
        self.decoder_model = nn.LSTM(
            input_size=self.decoder_hidden_size,
            hidden_size=self.decoder_hidden_size,
            num_layers=1,
            batch_first=False,
            dropout=decoder_dropout_prob,
            bidirectional=False
        )
        self.num_baseline_features = num_baseline_features
        self.training = True
        self.decoder_proj = ConvLinear(
            in_features=self.decoder_hidden_size,
            out_features=12,
            channel_last=True
        )
        self.baseline_proj0 = nn.Linear(self.num_baseline_features, self.encoding_size)
        self.baseline_proj1 = nn.Linear(self.encoding_size, self.encoding_size)
        self.cat_act = nn.GELU()
        self.output_proj = nn.Linear(25, self.decoder_hidden_size)
        self.input_proj = nn.Linear(1, self.decoder_hidden_size)

    def decode(self, nonseq_encoding, length_sim, length_window, device, output):
        output, _ = self.decoder_model(output)
        decoding = self.decoder_proj(output)
        return decoding, None, None

    def forward(
        self,
        x,
        baseline_df,
        length_sim,
        length_window,
        onlyz=False,
        fromz=False,
        use_baselines=False,
        z_in=None,
        use_time_series=True
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not fromz:
            if use_baselines:
                baseline_z = self.baseline_proj0(baseline_df)
                baseline_z = self.cat_act(baseline_z)
                baseline_z = self.baseline_proj1(baseline_z)
                z = baseline_z
            elif use_time_series:
                output, (_, cells) = self.encoder_lstm(x.permute(2, 0, 1))
                z = cells.transpose(0, 1).flatten(-2)
                z = self.non_seq_proj(z)

            # Split into mean/std
            mean, std = torch.chunk(z, 2, dim=-1)
            std = torch.exp(0.5 * std)
            if self.training:
                z_sample = torch.randn_like(mean).to(device) * std + mean
            else:
                z_sample = mean
            if onlyz:
                from torch.distributions import Normal
                return z_sample, Normal(mean, std)
        else:
            z_sample = z_in

        # Decode
        states, _, _ = self.decode(
            z_sample, length_sim, length_window, device,
            output=None
        )
        return states, None

##########################################################################
# Mechanistic model (example: PPGtoECG), with ODE + NN synergy
##########################################################################

class PPGtoECG(nn.Module):
    """
    Mechanistic ODE + Neural synergy model.
    (Truncated from your provided code to focus on relevant parts.)
    """
    def __init__(
        self,
        hidden_size,
        num_layers,
        encoder_dropout_prob,
        encoder_lstm=None,
        decoder_lstm=None,
        num_baseline_features=49,
        temperature=1.0,
        dt=0.1
    ):
        super().__init__()
        self.encoding_size = 50
        self.temperature = nn.Parameter(
            torch.FloatTensor([temperature]),
            requires_grad=True
        )
        self.training = True
        self.dt = nn.Parameter(torch.FloatTensor([dt]), requires_grad=True)

        # Example parameter for ODE constraints
        self.register_buffer(
            "param_lims",
            torch.FloatTensor([[-1.7, 1.7],
                               # ... omitted for brevity ...
                               [0.4, 0.6]])
        )
        # ... More buffer definitions for state_lims, etc.

        if encoder_lstm is None:
            self.encoder_lstm = nn.LSTM(
                input_size=12,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=False,
                dropout=encoder_dropout_prob,
                bidirectional=True
            )
        else:
            self.encoder_lstm = encoder_lstm

        self.decoder_hidden_size = hidden_size
        self.decoder_num_layers = num_layers
        if decoder_lstm is None:
            self.decoder_model = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=False,
                dropout=encoder_dropout_prob,
                bidirectional=False
            )
        else:
            self.decoder_model = decoder_lstm

        self.norm_in = nn.BatchNorm1d(1)
        self.norm_out = nn.BatchNorm1d(1)

        self.non_seq_proj = nn.Linear(hidden_size * num_layers * 2, self.encoding_size)
        self.baseline_proj0 = nn.Linear(num_baseline_features, self.encoding_size)
        self.baseline_proj1 = nn.Linear(self.encoding_size, self.encoding_size)
        self.cat_act = nn.GELU()
        self.input_proj = nn.Linear(3, hidden_size)  # 3 for state
        self.decoder_proj = ConvLinear(in_features=hidden_size, out_features=12, channel_last=True)
        self.mixture_weights = nn.Parameter(torch.ones(3) / 3.0, requires_grad=True)

    def ecg_dynamics(self, params, state, Tn):
        """
        ODE-based ECG dynamics (truncated).
        """
        # ...
        return torch.zeros_like(state), torch.zeros(state.size(0), dtype=torch.bool)

    def decode(self, nonseq_encoding, length_sim, window_length, device):
        """
        Merges ODE-based forward simulation with LSTM-based decoding.
        """
        # ...
        # Provide a typical mechanistic decode or forward integration
        # ...
        states = torch.zeros(nonseq_encoding.size(0), length_sim, 3).to(device)
        param = nonseq_encoding[:, :23]  # or however your param chunk is
        # ...
        # LSTM decode:
        decoder_input = self.input_proj(states).permute(1, 0, 2)
        output, _ = self.decoder_model(decoder_input)
        decoding = self.decoder_proj(output)
        return decoding, param, None, 0.0  # return extra stuff as needed

    @property
    def kl_dist(self):
        """
        Example distribution for regularizing
        """
        from torch.distributions import Normal
        # ...
        return Normal(loc=0, scale=1)  # placeholder

    def forward(
        self,
        ecg_no_interp,
        baseline_df,
        length_sim,
        length_window,
        onlyz=False,
        fromz=False,
        use_baselines=False,
        z_in=None,
        use_time_series=True
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ...
        if not fromz:
            # NN-based encode
            # ...
            pass
        else:
            pass

        # decode ODE + RNN
        states, param, init_state, jac = self.decode(
            torch.zeros(ecg_no_interp.size(0), 25).to(device),
            length_sim, length_window, device
        )
        # ...
        # final shape is [time, batch, channels]; re-permute as needed
        # For demonstration, return (batch, channels, time)
        return states.permute(1, 2, 0), jac

##########################################################################
# (2) Functions to separate ODE vs. NN parameters, track grad norms, partial rec
##########################################################################

def separate_params_ode_nn(model: nn.Module):
    """
    Based on name checks or any other logic, classify parameters as ODE vs. NN.
    Adjust if needed for your param naming scheme.
    """
    ode_params = []
    nn_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Heuristic: if certain substrings appear, classify as ODE
        # (Adapt to your actual naming)
        if any(key in name.lower()
               for key in ["ecg_dynamics", "param_lims", "temperature", "mixture_weights", "dt"]):
            ode_params.append(param)
        else:
            nn_params.append(param)
    return ode_params, nn_params

def compute_grad_norm(param_list):
    grad_sq = 0.0
    for p in param_list:
        if p.grad is not None:
            grad_sq += p.grad.norm(2).item()**2
    return grad_sq**0.5

def track_gradient_norms(model):
    ode_params, nn_params = separate_params_ode_nn(model)
    gn_ode = compute_grad_norm(ode_params)
    gn_nn  = compute_grad_norm(nn_params)
    return gn_ode, gn_nn

@torch.no_grad()
def partial_reconstruction(model,
                           ecg_input,
                           baseline_input,
                           length_sim,
                           length_window,
                           freeze_ode=False,
                           freeze_nn=False):
    # Identify param subsets
    ode_params, nn_params = separate_params_ode_nn(model)
    orig_ode_flags = [p.requires_grad for p in ode_params]
    orig_nn_flags  = [p.requires_grad for p in nn_params]

    if freeze_ode:
        for p in ode_params:
            p.requires_grad_(False)
    if freeze_nn:
        for p in nn_params:
            p.requires_grad_(False)

    model.eval()
    out, _ = model(
        ecg_no_interp=ecg_input,
        baseline_df=baseline_input,
        length_sim=length_sim,
        length_window=length_window,
        onlyz=False,
        fromz=False,
        use_baselines=(baseline_input is not None),
        use_time_series=(ecg_input is not None)
    )

    # Restore
    for p, f in zip(ode_params, orig_ode_flags):
        p.requires_grad_(f)
    for p, f in zip(nn_params, orig_nn_flags):
        p.requires_grad_(f)

    return out  # shape: (batch, channels, time)

def demonstrate_partial_reconstruction(model,
                                       ecg_true,
                                       ecg_input,
                                       baseline_input,
                                       length_sim,
                                       length_window,
                                       idx_plot=0,
                                       save_path=None):
    rec_full     = partial_reconstruction(model, ecg_input, baseline_input, length_sim, length_window, False, False)
    rec_ode_only = partial_reconstruction(model, ecg_input, baseline_input, length_sim, length_window, False, True)
    rec_nn_only  = partial_reconstruction(model, ecg_input, baseline_input, length_sim, length_window, True,  False)

    # Convert to numpy for plotting
    rec_full_np     = rec_full[idx_plot].detach().cpu().numpy()     # shape [channels, time]
    rec_ode_only_np = rec_ode_only[idx_plot].detach().cpu().numpy()
    rec_nn_only_np  = rec_nn_only[idx_plot].detach().cpu().numpy()
    gt_np           = ecg_true[idx_plot].detach().cpu().numpy()

    # Plot (channel 0 for illustration)
    ch = 0
    plt.figure(figsize=(12,5))
    plt.plot(gt_np[ch], label="Ground Truth", color="black", linewidth=2)
    plt.plot(rec_full_np[ch], label="Full Model")
    plt.plot(rec_ode_only_np[ch], label="ODE-Only (NN Frozen)")
    plt.plot(rec_nn_only_np[ch], label="NN-Only (ODE Frozen)")
    plt.title("Partial Reconstruction (Channel 0)")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

##########################################################################
# (3) Example training code hooking in gradient-norm tracking & partial recon
##########################################################################

def train_step(model, optimizer, data_batch, config):
    """
    Simplified example of a single training step:
    data_batch = (ecg_input, baseline_input, ecg_true, etc.)
    """
    model.train()
    optimizer.zero_grad()

    ecg_input, baseline_input, ecg_true = data_batch
    output, _ = model(
        ecg_no_interp=ecg_input,
        baseline_df=baseline_input,
        length_sim=config.length_sim,
        length_window=config.length_window
    )

    # Example loss
    # MSE vs ground truth, shape: [batch, channels, time]
    # ecg_true is also [batch, channels, time]
    criterion = nn.MSELoss()
    loss = criterion(output, ecg_true)

    # Backprop
    loss.backward()

    # Track gradient norms
    gn_ode, gn_nn = track_gradient_norms(model)
    wandb.log({"grad_norm_ODE": gn_ode, "grad_norm_NN": gn_nn})

    # Clip or step
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    return loss.item()

##########################################################################
# (4) Main "do()" driver adapted from your code, hooking in synergy checks
##########################################################################

def do(config):
    wandb.init(project="mech-interp-synergy", config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Build or load model
    if config["mechanistic"]:
        model = PPGtoECG(
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            encoder_dropout_prob=config["encoder_dropout_prob"],
            dt=config["dt"],
            temperature=config["temperature"]
        ).to(device)
    else:
        model = BlackBoxAutoencoder(
            encoding_size=50,
            encoder_input_size=30,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            encoder_dropout_prob=config["encoder_dropout_prob"],
            num_baseline_features=49,
            decoder_dropout_prob=config["encoder_dropout_prob"],
            decoder_input_size=1
        ).to(device)

    # 2) Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)  # or your warmup lambda

    # 3) Example data placeholders
    # Replace with your real Datasets (e.g., ds_ecg, ds_input, etc.)
    # Suppose each item in data is (ecg_input, baseline_input, ecg_true)
    train_data = [
        # (ecg_in, baseline_in, ecg_true) ...
    ]
    eval_data = [
        # ...
    ]

    # 4) EarlyStop / Train loop
    early_stopper = EarlyStopper(patience=3)
    for epoch in range(config["epochs"]):
        # --- Training ---
        np.random.shuffle(train_data)
        total_loss = 0.0
        for i, batch in enumerate(train_data):
            batch = tuple(x.to(device) for x in batch)
            loss_val = train_step(model, optimizer, batch, config)
            total_loss += loss_val
            scheduler.step()

        avg_train_loss = total_loss / max(len(train_data), 1)
        wandb.log({"epoch_train_loss": avg_train_loss}, step=epoch)
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f}")

        # --- Evaluation & Early Stopping ---
        if len(eval_data) > 0:
            model.eval()
            with torch.no_grad():
                eval_loss = 0.0
                for batch_eval in eval_data:
                    batch_eval = tuple(x.to(device) for x in batch_eval)
                    ecg_input, baseline_input, ecg_true = batch_eval
                    out_eval, _ = model(
                        ecg_no_interp=ecg_input,
                        baseline_df=baseline_input,
                        length_sim=config["length_sim"],
                        length_window=config["length_window"]
                    )
                    eval_loss += F.mse_loss(out_eval, ecg_true).item()
                avg_eval_loss = eval_loss / len(eval_data)
                wandb.log({"epoch_eval_loss": avg_eval_loss}, step=epoch)
                print(f"[Epoch {epoch}] Eval Loss: {avg_eval_loss:.4f}")

            early_stopper(avg_eval_loss)
            if early_stopper.early_stop:
                print("Early stopping triggered.")
                break

    # 5) Partial reconstruction demo on final model
    # Grab a small batch from eval_data
    if len(eval_data) > 0:
        example_batch = eval_data[0]
        example_batch = tuple(x.to(device) for x in example_batch)
        ecg_in, base_in, ecg_true = example_batch
        demonstrate_partial_reconstruction(
            model, ecg_true, ecg_in, base_in,
            config["length_sim"], config["length_window"],
            idx_plot=0,
            save_path="partial_recon_demo.png"
        )
        print("Partial reconstruction plot saved as partial_recon_demo.png")

    # Save final model
    torch.save(model.state_dict(), "final_model.pt")
    print("Model saved as final_model.pt")

##########################################################################
# (5) If run as script, parse args or define config, then call do()
##########################################################################

if __name__ == "__main__":
    # Example minimal config
    config = {
        "mechanistic": True,
        "hidden_size": 64,
        "num_layers": 1,
        "encoder_dropout_prob": 0.0,
        "temperature": 1.0,
        "dt": 1.0,
        "lr": 1e-4,
        "epochs": 5,
        "length_sim": 400,
        "length_window": 4000,
    }
    do(config)

