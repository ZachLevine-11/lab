import random

import torch
from torch import nn as nn
from torch.distributions import Normal
from torch.autograd import functional

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0, verbose=False):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement.
            min_delta (float): Minimum improvement to be considered as an improvement.
            verbose (bool): Whether to print messages when training is stopped early.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric):
        """
        Args:
            metric (float): Current value of the monitored metric (e.g., validation loss).
        """
        if self.best_score is None:
            self.best_score = metric
        elif metric >= self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopper: No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = metric
            self.counter = 0

class ConvLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, channel_last=True):
        super().__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1, bias=bias)
        self.channel_last = channel_last

    def forward(self, x):
        assert x.ndim == 3, "Expected input to be (batch_size, seq_len, input_size) or (batch_size, input_size, seq_len)"
        if self.channel_last:
            x = x.transpose(1, 2)
            x = self.conv(x)
            x = x.transpose(1, 2)
        else:
            x = self.conv(x)
        return x

class BlackBoxAutoencoder(nn.Module):
    def __init__(self, encoding_size, encoder_input_size, hidden_size, num_layers, encoder_dropout_prob, num_baseline_features, decoder_dropout_prob, decoder_input_size):
        super().__init__()
        self.drop = nn.Dropout1d(p = encoder_dropout_prob)
        self.encoding_size = encoding_size
        self.encoder_lstm = nn.LSTM(input_size=12, hidden_size=hidden_size, num_layers=num_layers,
                                    batch_first=False, dropout=encoder_dropout_prob, bidirectional=True)
        self.non_seq_proj = nn.Linear(hidden_size * num_layers * 2,
                  out_features=self.encoding_size)
        self.decoder_hidden_size = hidden_size
        self.decoder_model = nn.LSTM(input_size=self.decoder_hidden_size, hidden_size=self.decoder_hidden_size, num_layers=1,
                                     batch_first=False, dropout=decoder_dropout_prob, bidirectional=False)

        self.num_baseline_features = num_baseline_features
        self.encoding_size = encoding_size
        self.training = True
        self.decoder_proj = ConvLinear(in_features=self.decoder_hidden_size, out_features=12, channel_last=True)

        self.baseline_proj0 = nn.Linear(in_features = self.num_baseline_features,
                                      out_features=self.encoding_size)
        self.baseline_proj1 = nn.Linear(in_features = self.encoding_size,
                                      out_features= self.encoding_size)
        self.cat_act = nn.GELU()
        self.output_proj = nn.Linear(25, self.decoder_hidden_size)
        self.input_proj = nn.Linear(1, self.decoder_hidden_size)

    def decode(self, nonseq_encoding, length_sim, length_window, device, output):
        output, (_, _) = self.decoder_model(output)
        decoding = self.decoder_proj(output)
        return decoding, None, None

    def forward(self, x, baseline_df, length_sim, length_window, onlyz = False, fromz = False, use_baselines = False, z_in = None, use_time_series = True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output = None
        if not fromz:
            if use_baselines:
                baseline_z = self.baseline_proj0(baseline_df)
                baseline_z  = self.cat_act(baseline_z)
                baseline_z = self.baseline_proj1(baseline_z)
                z = baseline_z
            elif use_time_series:
                output, (_, cells) = self.encoder_lstm(x.permute(2, 0, 1))
                z = cells.transpose(0, 1).flatten(-2)
                ##important for stability if you want to use a fast learning rate (i.e 1e-2)
                z =  self.non_seq_proj(z)
            nonseq_encoding_mean, nonseq_encoding_std = torch.chunk(z, 2, dim=-1)
            nonseq_encoding_std = torch.exp(0.5*nonseq_encoding_std)
            if self.training:
                nonseq_encoding_sample = torch.randn_like(nonseq_encoding_mean).to(
                    device) * nonseq_encoding_std + nonseq_encoding_mean
            else:
                nonseq_encoding_sample = nonseq_encoding_mean
            if onlyz:
                return nonseq_encoding_sample, Normal(nonseq_encoding_mean, nonseq_encoding_std)
        else:
            nonseq_encoding_sample = z_in
        states, param, init_state = self.decode(nonseq_encoding_sample, length_sim, length_window, device, output)
        return states, None

class PPGtoECG(torch.nn.Module):
    def __init__(self, hidden_size, num_layers, encoder_dropout_prob, encoder_lstm, decoder_lstm, num_baseline_features = 49, temperature = 1.0, dt = 0.1):
        super().__init__()
        self.encoding_size = 50
        self.temperature = nn.Parameter(torch.FloatTensor([temperature]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), requires_grad = False)
        self.training = True
        self.dt = nn.Parameter(torch.FloatTensor([dt]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), requires_grad = False)
        self.non_seq_proj1 = nn.Linear(hidden_size * num_layers *2,  # times 2 for bidirectional
                  out_features=self.encoding_size)
        self.drop = nn.Dropout1d(p = encoder_dropout_prob)
        self.state_size = 3
        self.num_baseline_features = num_baseline_features
        if encoder_lstm is None:
            self.encoder_lstm = nn.LSTM(input_size=12, proj_size=0, hidden_size=hidden_size, num_layers=num_layers,
                                    batch_first=False, dropout=encoder_dropout_prob, bidirectional=True)
        else:
            self.encoder_lstm = encoder_lstm
        ##the complexity of the decoder network acts as a regularizer on the neural networks
        self.decoder_hidden_size = hidden_size
        self.decoder_num_layers = num_layers
        if decoder_lstm is None:
            self.decoder_model = nn.LSTM(input_size=self.state_size, proj_size=0, hidden_size=self.decoder_hidden_size, num_layers=self.decoder_num_layers,
                                    batch_first=False, dropout=encoder_dropout_prob, bidirectional=False)
        else:
            self.decoder_model = decoder_lstm
        self.norm_in = torch.nn.BatchNorm1d(1)
        self.norm_out = torch.nn.BatchNorm1d(1)
        self.param_size = 23
        self.register_buffer("param_lims", torch.FloatTensor(
            ##P
            [[-1.7, 1.7000e+00],
             [-7.0000e+01*torch.pi/180, -5.0000e+01*torch.pi/180],
             [2.4000e-01, 3.5000e-01],
             ##Q
             [-6.5000e+00, -1.5000e+00],
             [-25*torch.pi/180, -5.0000e+00*torch.pi/180],
             [8.0000e-02, 1.2000e-01],
             ##R
             [1.5000e+01, 5.5000e+01],
             [-2.0000e+01*torch.pi/180, 2.0000e+01*torch.pi/180],
             [9.0000e-02, 1.1000e-01],
             ##S
             [-0.8, -0.3],
             [5*torch.pi/180, 25*torch.pi/180],
             [1e-16, 1.0000e-01],
             #Tplus
             [5.0000e-01, 9.0000e-01],
             [8.0000e+01*torch.pi/180, 1.2000e+02*torch.pi/180],
             [3.0000e-01, 5.0000e-01],
             #Tmimus
             [2.0000e-01, 9.0000e-01],
             [1.3000e+02*torch.pi/180, 1.5000e+02*torch.pi/180],
             [1.5000e-01, 2.5000e-01],
             #RR1
             [0.08, 0.12],
             [0.008, 0.012],
             #RR2
             [0.23, 0.27],
             [0.008, 0.012],
             ##ratio
             [0.4, 0.6]
             ]))
        self.register_buffer("state_lims", torch.FloatTensor(
            ##theta
            [[-torch.pi, torch.pi],
             #z0
             [-4e-6, 4e-6]]))
        self.non_seq_proj = nn.Linear(hidden_size * num_layers * 2,
                  out_features=self.encoding_size)
        self.baseline_proj0 = nn.Linear(in_features = self.num_baseline_features,
                                      out_features=self.encoding_size)
        self.baseline_proj1 = nn.Linear(in_features = self.encoding_size,
                                      out_features= self.encoding_size)
        self.cat_act = nn.GELU()
        self.mixture_weights = nn.Parameter(torch.ones(3) / 3.0, requires_grad = True)
        self.decoder_proj = ConvLinear(in_features=self.decoder_hidden_size, out_features=12, channel_last=True)
        self.norm = nn.LayerNorm(self.num_baseline_features)

    ## as close as we can get to the constrained scale
    def unconstrain(self, y, min, max, EPS: float = 1e-8):
        numerator = y - min
        denominator = max - min
        return self.temperature * torch.logit(numerator / denominator, eps=EPS)

    @property
    def kl_dist(self):
        ##use the unconstrained param lims on the unconstrained values, so that the constrained scale is close to the constrained true value
        means_constrained = torch.cat([self.param_lims.mean(dim = 1), self.state_lims.mean(dim = 1)])
        lower_lims = torch.cat([self.param_lims[:, 0], self.state_lims[:, 0]])
        upper_lims = torch.cat([self.param_lims[:, 1], self.state_lims[:, 1]])
        means_unconstrained = self.unconstrain(means_constrained, lower_lims, upper_lims)
        return Normal(means_unconstrained, torch.ones(self.param_size + 2).to(self.state_lims.device))

    def ecg_dynamics(self, params, state, Tn):
        params_ecg = params[:, 0:18]
        P, Q, R, S, T, Tminus = torch.split(params_ecg, 3, dim=1)
        chunked_pars = [P, Q, R, S, T, Tminus]  # [a, theta, b]
        x, y, z = torch.split(state, 1, dim=-1)
        alpha = 1 - torch.sqrt(x ** 2 + y ** 2)
        theta = torch.atan2(y, x)
        omega = (torch.pi * 2) / Tn
        dx = alpha * x - omega.reshape(x.shape[0], 1) * y
        dy = alpha * y + omega.reshape(x.shape[0], 1) * x
        sums = []
        theta_is_close_to_zero = torch.abs(torch.abs(theta)) < 0.0005
        for par in chunked_pars:
            deltatheta = (theta - par[:, 1].reshape(theta.shape))
            deltatheta -= torch.round(deltatheta / (2 * torch.pi)) * 2 * torch.pi
            sums.append(
                par[:, 0].reshape(par.shape[0], 1)
                * deltatheta
                * torch.exp(-((deltatheta ** 2) / (2 * (par[:, 2] ** 2)).reshape(-1, 1)))
            )
        zsum = torch.stack(sums, -1).sum(-1).reshape(x.shape[0], 1)
        dz = -zsum - z
        dstate = torch.cat([dx, dy, dz], dim=-1)
        return dstate, theta_is_close_to_zero

    def gaussian_pdf(self, x, mean, std, device):
        coeff = 1 / (std * torch.sqrt(torch.tensor([2], dtype=torch.float, requires_grad=True).to(device)))
        exponent = -0.5 * ((x - mean) / (std))
        return coeff * torch.exp(exponent)

    def S(self, freq, params_rr, device):
        mean_1 = params_rr[:, 0].reshape(-1, 1)
        std_1 = params_rr[:, 1].reshape(-1, 1)
        mean_2 = params_rr[:, 2].reshape(-1, 1)
        std_2 = params_rr[:, 3].reshape(-1, 1)
        s1 = self.gaussian_pdf(freq, mean_1, std_1, device)
        s2 = self.gaussian_pdf(freq, mean_2, std_2, device)*params_rr[:, 4].reshape(-1, 1)
        return s1 + s2

    def make_Sf(self, params_rr, device):
        f = torch.linspace(0, 0.5, 2**4, requires_grad=True).to(device)
        S_f = self.S(f, params_rr, device)
        return S_f, f

    def make_Tn(self, params_rr, device):
        S_f, f = self.make_Sf(params_rr, device)
        amplitudes = torch.sqrt(S_f)
        phases = torch.linspace(0, 1, 2**4, requires_grad = True).to(device) * 2 * torch.pi
        T = torch.fft.ifft(torch.polar(amplitudes, phases), n = 2**4).real + 0
        return T

    ## the normal sigmoid is too aggressive a constraint and we loose too much resolution if we use it
    def smooth_bound(self, x, min_val, max_val, temp, weights):
        f1 = torch.log(1 + torch.exp(x / temp)) / (1 + torch.abs(x))
        f2 = torch.tanh(x / temp)
        f3 = 2 * torch.sigmoid(x / temp) - 1
        mixture = weights[0] * f1 + weights[1] * f2 + weights[2] * f3
        mixture = mixture / (weights.sum())
        return (max_val - min_val) * (mixture + 1) / 2 + min_val

    def constrain(self, x, min, max):
        return self.smooth_bound(x, min_val= min, max_val=max, temp=self.temperature,
                              weights=torch.softmax(self.mixture_weights, dim=0))

    def decode(self, nonseq_encoding, length, window_length, device):
        param, state = nonseq_encoding[..., :self.param_size], nonseq_encoding[..., self.param_size:]
        param.requires_grad_(True)
        state.requires_grad_(True)
        param = self.constrain(param, min=self.param_lims[:, 0], max=self.param_lims[:, 1])
        state = self.constrain(state, min=self.state_lims[:, 0], max=self.state_lims[:, 1])
        ##put the initial state on the unit circle
        state = torch.cat([torch.cos(state[:, 0]).reshape(-1, 1), torch.sin(state[:, 0]).reshape(-1, 1), state[:, 1].reshape(-1, 1)], dim = 1).to(device)
        ##constrain z0
        params_rr = param[:, 18:23]
        Tn = self.make_Tn(params_rr, device)
        ##we need to have a gradient to backprop on
        dynamics_out = self.ecg_dynamics(param, state, Tn[:, 0].reshape(-1, 1))[0]
        jacobian = functional.jacobian(lambda x,y,z: self.ecg_dynamics(y, x, z)[0], (state, param, Tn[:, 0].reshape(-1, 1)))
        ##these jacobians are all different sizes, so sum the norms instead of norming the sums
        jac_state = torch.norm(jacobian[0], p='fro') + torch.norm(jacobian[1], p='fro') + torch.norm(jacobian[2], p='fro')
        states = [state]
        i_s = torch.ones(Tn.shape[0], dtype=torch.int, requires_grad=False).reshape(nonseq_encoding.shape[0], -1).to(device) ##counts number of revolution cycles passed in each trajectory
        for step in range(0, length - 1, 1):
            operative_Tn = Tn[torch.arange(Tn.size(0)).unsqueeze(1), torch.min(i_s.clone(), Tn.shape[1] * torch.ones_like(i_s.clone()))]
            operative_Tn = operative_Tn.reshape(-1,1)  ##The clone line is extremely important, making sure we don't keep using the modified in-place counter
            dstate_temp, theta_is_close_to_zero = self.ecg_dynamics(param, state,  operative_Tn.reshape(-1, 1))  # With just one heartbeat, it doesn't matter
            norms = torch.linalg.vector_norm(dstate_temp, ord=2, dim=-1, keepdim=True)
            threshold = 1e6
            scaling_factors = torch.where(norms > threshold, threshold / norms, torch.ones_like(norms, requires_grad=True))
            clipped_dstate_temp = dstate_temp * scaling_factors
            dstates = clipped_dstate_temp.view(-1, 3)
            state = state + dstates * torch.exp(self.dt)
            states.append(state)
            i_s = i_s + theta_is_close_to_zero*1
        states = torch.stack(states, dim=-2)
        return states, param, state, jac_state

    def forward(self, x, baseline_df, length_sim, length_window, onlyz = False, fromz = False, use_baselines = False, z_in = None, use_time_series = True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not fromz:
            if use_baselines:
                baseline_z = self.norm(baseline_df)
                baseline_z = self.baseline_proj0(baseline_z)
                baseline_z  = self.cat_act(baseline_z)
                baseline_z = self.baseline_proj1(baseline_z)
                z = baseline_z
            elif use_time_series:
                output, (_, cells) = self.encoder_lstm(x.permute(2, 0, 1))
                z = cells.transpose(0, 1).flatten(-2)
                z =  self.non_seq_proj(z)
            nonseq_encoding_mean, nonseq_encoding_std = torch.chunk(z, 2, dim=-1)
            nonseq_encoding_std = torch.exp(0.5*nonseq_encoding_std)
            if self.training:
                nonseq_encoding_sample = torch.randn_like(nonseq_encoding_mean).to(device) * nonseq_encoding_std + nonseq_encoding_mean
            else:
                nonseq_encoding_sample = nonseq_encoding_mean
            if onlyz:
                return nonseq_encoding_sample, Normal(nonseq_encoding_mean, nonseq_encoding_std)
        else:
            nonseq_encoding_sample = z_in
        states, param, init_state, jac = self.decode(nonseq_encoding_sample, length_sim, length_window, device)
        decoder_input = states.permute(1, 0, 2) #[batch, time, state] --> [time, batch, state]
        output, (_, _) = self.decoder_model(decoder_input)
        decoding = self.decoder_proj(output)
        return decoding, jac