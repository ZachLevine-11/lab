import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats

#adapted from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5870860/#sup1
def find_thresh(sample ,CDFstep=0.001, nPerm = 1000, CDFMax = 0.99):
    cdf_values_seq = np.arange(start = 0.9, stop = CDFMax, step = CDFstep)
    thresholds = np.zeros([len(cdf_values_seq), nPerm])
    # Iterate over all CDF values and permutations
    for s2 in list(range(nPerm)):
        #randomly sample the data with replacement and make a CDF
        CDF=ECDF(np.random.choice(sample, len(sample), replace = True))
        unq_d=np.sort(sample.unique())
        CDFvals=CDF(unq_d)
        ##for each CDF value of interest, find the expression value corresponding to it from the x axis of the CDF
        for s1 in list(range(len(cdf_values_seq))):
            for i in range(len(CDFvals)):
                if CDFvals[i] >= cdf_values_seq[s1] and CDFvals[i] <= CDFMax:
                    break
            corresponding_expression = unq_d[i]
            thresholds[s1, s2] = corresponding_expression ##for each percentile, for each iteration, gives the expression value corresponding to that percentile of the cdf
    cdf_target, thresh  = None, None
    for i in range(len(cdf_values_seq)):
        cdf_value = cdf_values_seq[i]
        if len(set(thresholds[i,:])) > 1:
            cdf_target = cdf_value
            thresh = stats.mode(thresholds[i,:])[0]
            break
    return cdf_target, thresh