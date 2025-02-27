import numpy as np
from numpy import linalg as LA
from scipy.fft import fft, fftfreq
import pandas as pd
from LabData.DataLoaders.ECGLoader import ECGLoader
from LabData.DataLoaders.CGMLoader import CGMLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

CGM_inst = CGMLoader()
df = CGM_inst.get_data().df

##Pick a random person's time series (one of the two listed below) to start with
one_person = df.loc["100863", :]  # 10K_9999409119
x1 = one_person[["GlucoseValue"]].dropna().T.values[0]
x2 = one_person[["PPGR"]].dropna().T.values[0]


##generate the time-dependent reconstruction bounds and normalization facors
def generateRC_params(t, M, N, Nprime):
    if t >= 1 and t <= M - 1:
        Mt = 1 / t
        Lt = 1
        Ut = t
    elif t >= M and t <= Nprime:
        Mt = 1 / M
        Lt = 1
        Ut = M
    else:
        Mt = 1 / (N - t + 1)
        Lt = t - N + M
        Ut = M
    return Mt, Lt, Ut


def calculate_PhaseQuadrature(PC1, PC2):
    ab_corr = np.correlate(PC1, PC2, "full")  ##cross correlate the series
    ##assume delta t is 1, have to change otherwise
    ##the max of the cross correlation will be 722 (the time point corresponding to the end of the time series) if the two time series are equal, and the closeness to 722 indicates the degree of phase quadrature
    t_shift = abs(PC1.shape[0] - 1 - ab_corr.argmax())  ##subtract one because we have an index
    return t_shift


###finds the right set of components to recnostruct over by one of two methods:
##how = "manual" to find pairs of approximately equal SSA eigenalues and return their indices in addition to index zero
##how = anything else goes by statistical power in comparison to SSA on a random datatset with the same distribution
def find_lset(W, Akay, x, M=None, eps=1e-3, how="manual"):
    if how == "manual":
        ##first find the list of pairs of approximately equal eigenvalues.
        ##only use each eigenvalue once
        counter = 0
        pairs = {0}
        ##we only want one eigenvalue from the eigenvalue eigenvector pairs, but keep both for now to help with the intersection with the set from the Akays
        while counter < W.size:
            ##pick an eigenvalue
            pick_eig = W[counter]
            for j in range(W.size):
                if abs(pick_eig - W[j]) <= eps and counter != j:
                    pairs.add(j)
                    pairs.add(counter)  ##only add one eigenvalue from the pair to represent that mode
            counter += 1
        ##find the list of pairs of t-pcs in approximate phase quadarture
        approx_equal_pcs = {0}
        for PC_counter_1 in range(0, Akay.shape[1], 1):  # don't consider zero
            for PC_counter_2 in range(0, Akay.shape[1], 1):
                if PC_counter_1 != PC_counter_2:
                    PC1 = Akay[:, PC_counter_1]
                    PC2 = Akay[:, PC_counter_2]
                    phasequad = calculate_PhaseQuadrature(PC1, PC2)
                    inPhaseQuadrature = phasequad < 2
                    if inPhaseQuadrature:
                        approx_equal_pcs.add(PC_counter_1)
                        approx_equal_pcs.add(PC_counter_2)
        pairs = pairs.intersection(approx_equal_pcs)
        pairs = list(pairs)
        return pairs
    else:
        ##based on https://github.com/kieferk/pymssa/blob/master/pymssa/mssa.py
        ##see which eigenvalues are beyond the 95% percentile (in real part) of a bootstapped sample
        random_x = np.random.normal(0, np.std(x), size=x.size)
        random_W, random_Akay = SSA(random_x, M, reconstruct=False)
        real_random_W = np.real(random_W)
        pctl = np.percentile(real_random_W, 95, axis=0)
        maxKeep = np.where(pctl > W)[0][
            0]  ##find first place where noise eigenvalue real > actual eigenvalue real to truncate
        return list(range(maxKeep))  ##Keep all eigs up to that point


def SSA(x, M=40, how="VGA", reconstruct=True, lset_method="manual"):
    ##normalize the the time series
    meansig = np.mean(x)
    x = (x - np.mean(x)) / np.std(x)
    N = x.size
    K = N - M + 1
    if how == "BK":
        ##Slower (VG version) Generate lag-covariance matrix
        Cx = np.zeros([M, M])
        for i in range(M):
            for j in range(M):
                for t in range(N - np.abs(i - j)):
                    Cx[i, j] += x[t] * x[t + np.abs(i - j)]
                Cx[i, i] *= (1 / (N - abs(i - j)))
    else:
        ##Faster (BK version): use the covariance trajectory matrix
        X = np.zeros([M, K])
        ##stack M times
        for k in range(M):
            X[k, :] = x[range(k, k + K, 1)]
        Cx = (1 / K) * np.matmul(X, X.T)
    ##perform eigendecomposition of whichever matrix we've chosen
    W, V = LA.eig(Cx)
    ##sort by decreasing power to make sure we grab the trend
    sortindices = np.argsort(W[::-1])  ##sort in descending order
    W = W[sortindices]  ##eigenvalues
    V = V[:, sortindices]  ##eigenvectors
    Akay = np.zeros([K, M])
    ##compute temporal principal components (T-PCS)
    for k in range(0, M, 1):
        for t in range(0, K, 1):
            selected_window = x[range(t, t + M, 1)] * V[:, k].T
            Akay[t, k] = np.sum(selected_window)
    if not reconstruct:
        return W, Akay
    else:
        ##find the set of T-EOFS to use in reconstruction based on finding pairs of nearly equal SSA eigenvalues
        lset = find_lset(W, Akay, x, M, how=lset_method)
        lset = [0, 1, 2, 3, 4, 5, 6]
        ##project the T-PCS onto the T-EOFS to generate the reconstructed components (RC)
        Rkay = Mts = np.zeros([x.size])
        Nprime = N - M + 1
        for t in range(1, N, 1):
            Mt, Lt, Ut = generateRC_params(t, M, N, Nprime)
            Mts[t] = Mt
            for k in lset:
                if Lt != Ut:
                    for j in range(Lt, Ut, 1):
                        Rkay[t] += Akay[t - j, k] * V[j, k]
                else:
                    j = 1
                    Rkay[t] += Akay[t - j, k] * V[j, k]
        Rkay = Rkay * Mts
        ##visualize results
        print("I got here")
        plt.scatter(range(Rkay.size), (Rkay - np.mean(Rkay) / np.std(Rkay)))
        plt.title("Reconstruction, M = " + str(M))
        plt.show()
        plt.scatter(range(x.size), x)
        plt.title("Actual Data")
        plt.show()
        return Rkay, Akay


##this doesn't really work yet.
##calibrate the SSA for a single time series
def calibrate_SSA(x, Mset=list(range(1, 200, 5)), dt=1):
    calib_dict = {}
    origx = x
    x = (x - np.mean(x)) / np.std(x)
    for M in Mset:
        ##we just calibrate based on the principal compoennts
        W, Akay = SSA(origx, M=M,
                      reconstruct=False)  ##ssa already centers and normalizes the data so just feed it the original time series
        lset = find_lset(W, Akay, x, M=M)
        meanerror = 0
        for pcCounter in lset:  ##only consider the T-PCS we're reconstructing over
            individual_PC = Akay[:, pcCounter]
            fft_PC = (2 / x.size) * np.square(np.abs(fft(individual_PC)))[0:x.size // 2]
            fft_x = (2 / x.size) * np.square(np.abs(fft(x)[0:x.size // 2]))
            freqs = fftfreq(x.size, dt)[:x.size // 2]  ##same for both
            ##each pc should pick up one peak from the original FFT of X
            # the rest of its FFT should be zero, but we don't really care about it
            sorted_PC_peak_ind = np.argsort(fft_PC[::-1])[0]
            pc_peak = fft_PC[sorted_PC_peak_ind]
            ##allow for zero pcs
            # grab the FFT at that peak index and check error
            x_peak = fft_x[sorted_PC_peak_ind]
            ##store the mean error over all the reconstructed PCS
            meanerror += (pc_peak - x_peak)
        meanerror /= len(lset)
        calib_dict[M] = meanerror
        print("M =", M, "error = ", meanerror)
    return calib_dict
