import numpy as np
from numpy.linalg import svd
from numpy.random import normal
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree, ClusterNode
from scipy.stats import median_abs_deviation
from scipy.stats import norm
from functools import reduce
from LabUtils.addloglevels import sethandlers
from pvalnode import pvalNode
from LabQueue.qp import qp
# from GeneticsPipeline.config import gencove_logs_path
import os
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import squareform
from scores_work import stack_matrices_and_bonferonni_correct, raw_matrices_save_path_prs, raw_matrices_save_path_pqtl

sethandlers()


# based on scipy to tree but returns testnodes instead of normal nodes.
def to_tree_pvalNodes(linkageMat, x):  ##cluster data and original data
    # Number of original objects is equal to the number of rows plus 1.
    n = x.shape[0] + 1
    # Create a list full of None's to store the node objects
    d = [None] * (n * 2 - 1)
    # Create the nodes corresponding to the n original objects.
    for i in range(0, n):
        d[i] = pvalNode(ClusterNode(i))
    nd = None
    for i, row in enumerate(linkageMat):
        fi = int(row[0])
        fj = int(row[1])
        nd = pvalNode(ClusterNode(i + n, d[fi], d[fj], row[2]))
        #                            ^ id   ^ left ^ right ^ dist
        d[n + i] = nd
    return nd


##kept out of the class to make queing easier
##make a single random matrix and see which associations in the original beat it
##return a dataframe with 1's where the original association beats it, and 0's elsewhere
def randomCluster(original_stackmat, i, withSave):
    if withSave:  ##write and read in from files
        make_all_saves(random_shuffle_prsLoader=True, i=i, withSave=True)
        permuted_stackmat = stack_matrices(i=i)  ##use original p values without the empty rows or columns removed
        ##to keep the indices in alignment and not have to drop the missing values, compare the p values directly
    else:
        ##the matrices will be stacked in make_all_saves
        permuted_stackmat = make_all_saves(random_shuffle_prsLoader=True, i=i, withSave=False)
    ##do for each PRS, the vectorized version does not work
    ##for duplicate columns, just keep one in the corrected matrix
    original_stackmat = original_stackmat.loc[:, ~original_stackmat.columns.duplicated()]
    permuted_stackmat = permuted_stackmat.loc[:, ~permuted_stackmat.columns.duplicated()]
    pvals = pd.DataFrame(0, columns=original_stackmat.columns, index=original_stackmat.index)
    for col in pvals.columns:
        pvals[col] = (original_stackmat[col] < permuted_stackmat[col]).apply(lambda x: 1 if x else 0)
    print("finished scoring simulation: " + str(i))
    return pvals


class clusterTree:
    def __init__(self, by="PRS", redo_original=False, withSave=False):
        if redo_original and not withSave:
            print("Creating real matrix")
            self.original_stackmat = make_all_saves(random_shuffle_prsLoader=False, i=-1, withSave=withSave)
            print("Real matrix finished!")
        elif redo_original and withSave:
            print("Creating real matrix")
            make_all_saves(random_shuffle_prsLoader=False, i=-1, withSave=withSave)
            self.original_stackmat = stack_matrices(i=-1)
            print("Real matrix finished!")
        else:
            self.original_stackmat = stack_matrices(i=-1)
            print("Original matrix loaded in from files")
        self.stackmat = log10res(self.original_stackmat.dropna())
        ##keep only PRSES with at least one significant interaction
        self.stackmat = self.stackmat[
            pd.Series(self.stackmat.columns[(self.stackmat < 0.005).any()].values.tolist()).drop_duplicates()]
        ##keep only phenotypes with at least one significnat interaction
        self.stackmat = self.stackmat.loc[
            pd.Series(self.stackmat.index[(self.stackmat < 0.005).any(1)].values.tolist()).drop_duplicates(),]
        if by == "PRS":
            self.rawcluster = linkage(self.stackmat)  # TODO: consider changing the distance metric
        else:
            self.rawcluster = linkage(self.stackmat.T)
        self.clusterDists = pd.DataFrame(self.rawcluster)[2]
        self.tree = pvalNode(to_tree_pvalNodes(self.rawcluster, self.stackmat))
        self.sigclusters = {}
        self.ps = [None] * (self.tree.count)
        self.leaves = {}
        self.ClusterIndices = {}
        self.all_nodes = {}
        self.by = by
        self.pvals_df = None

    def getLeaves(self):  ##get a list of leaves of the tree
        leftvisited = set()
        rightvisited = set()
        leavecache = [None] * (self.tree.count)
        leavecache[0] = self.tree
        self.all_nodes[0] = self.tree
        i = 0
        while i >= 0:
            currentnode = leavecache[i]
            nodeid = currentnode.id
            if currentnode.is_leaf():
                self.leaves[nodeid] = currentnode
                i -= 1
            else:
                if nodeid not in leftvisited:
                    leavecache[i + 1] = currentnode.left
                    leftvisited.add(nodeid)
                    i += 1
                    ##only need to do when we check the left or right node or we end up with double the keys
                    self.all_nodes[nodeid] = currentnode
                elif nodeid not in rightvisited:
                    leavecache[i + 1] = currentnode.right
                    rightvisited.add(nodeid)
                    i += 1
                else:
                    i -= 1

    def generateSSTIndices(x1, x2):  ##not using for now
        def sst(x):
            return ((x - x.mean(0)) ** 2).sum().sum()

        val = (sst(x1) + sst(x2)) / sst(pd.concat([x1, x2]))
        return val

    ##difference between internal distance and average distance in tree excluding this node
    def generateDistIndices(self, node, override=False, rawcluster=None):
        if not override:
            return (node.dist / self.clusterDists[self.clusterDists != node.dist].mean())
        else:
            temp_clusterDists = pd.DataFrame(rawcluster)[2]
            return (node.dist / temp_clusterDists[temp_clusterDists != node.dist].mean())

    def getClusterIndices(self, how="dist"):  ##need to have traversed the tree at least once for this to work
        if how != "dist":
            assert False
        for node_key in self.all_nodes.keys():
            self.ClusterIndices[node_key] = self.generateDistIndices(self.all_nodes[node_key])

    # based on sigclust2 in R
    def testClusters(self, nsim=1000, how="sigclust2_tree"):
        if how == "sigclust2_tree":
            ##compute eigenvalues of null gaussian dist
            mad = median_abs_deviation(x=self.stackmat, scale=1.4826)  ##match the R implementation
            backvar = mad ** 2
            avgx = (self.stackmat - list(self.stackmat.mean(0))).T
            u, s, v = svd(avgx)
            eigval_dat = (s ** 2) / (self.stackmat.shape[0] - 1)
            eigval_sim = eigval_dat
            eigval_sim[eigval_sim < 0] = 0
            ##repeat to match the size of the array
            eigval_sim_stacked = pd.concat([pd.Series(eigval_sim)] * self.stackmat.shape[0], axis=0)
            clusterindices_finalmerge_sim = [0] * nsim
            for i in range(nsim):
                generatedsim = normal(loc=0, size=self.stackmat.shape[0] * self.stackmat.shape[1],
                                      scale=np.sqrt(eigval_sim_stacked))
                generated_sim_range = range(0, len(generatedsim), self.stackmat.shape[1])
                j = 0
                thestack = []
                while j < len(generated_sim_range) - 1:
                    thestack.append(pd.Series(generatedsim[generated_sim_range[j]:generated_sim_range[j + 1]]))
                    j += 1
                generated_sim_final = pd.concat(thestack, axis=1).T
                generated_sim_final = abs(generated_sim_final)
                ##just use index for root node
                random_cluster = linkage(generated_sim_final)
                random_tree = to_tree(random_cluster, rd=False)
                clusterindices_finalmerge_sim[i] = self.generateDistIndices(random_tree, override=True,
                                                                            rawcluster=random_cluster)
                print("finished simulation: " + str(i))
            for node_key in self.all_nodes:
                self.all_nodes[node_key].pval = norm.cdf(self.ClusterIndices[node_key],
                                                         np.mean(clusterindices_finalmerge_sim),
                                                         np.std(clusterindices_finalmerge_sim))
        elif how == "random_matrix_loop":
            i = 0
            stackmats = [] * nsim
            while i < nsim:
                print("starting simulation: " + str(i))
                stackmats[i] = randomCluster(original_stackmat=self.original_stackmat, i=i)
                i += 1
                ##the scores / nsims are the p values
            res = reduce(lambda x, y: x.add(y, fill_value=0), stackmats)
            self.pvals_df = res / nsim
        elif how == "random_matrix_q_save":  ##save all intermediate results for each simulation, very memory intensive
            # os.chdir(gencove_logs_path)
            os.chdir("/net/mraid20/export/mb/logs/")
            with qp(jobname="z_genetics", max_r=10, max_u=10, _suppress_handlers_warning=True) as q:
                q.startpermanentrun()
                i = 0
                fundict = {}
                while i < nsim:
                    fundict[i] = q.method(randomCluster, (
                    self.original_stackmat, i, True))  ##test can be "t" for t test or "r" for regression))
                    i += 1
                fundict = {k: q.waitforresult(v) for k, v in fundict.items()}
            res = reduce(lambda x, y: x.add(y, fill_value=0), fundict.values())
            self.pvals_df = 1 - res / (nsim + 1)
        elif how == "random_matrix_q_nosave":  ##do all intermediate steps without saving
            os.chdir("/net/mraid20/export/mb/logs/")
            #            os.chdir(gencove_logs_path)
            with qp(jobname="perms", max_r=10, max_u=10, _suppress_handlers_warning=True) as q:
                q.startpermanentrun()
                i = 0
                fundict = {}
                while i < nsim:
                    fundict[i] = q.method(randomCluster, (
                    self.original_stackmat, i, False))  ##test can be "t" for t test or "r" for regression))
                    i += 1
                fundict = {k: q.waitforresult(v) for k, v in fundict.items()}
            res = reduce(lambda x, y: x.add(y, fill_value=0), fundict.values())
            self.pvals_df = 1 - res / (
                        nsim + 1)  ##equivalent of adding a small epsilon to the results so we never get a p value of zero

    def getSigNodes(self, alpha0=0.05):
        depth = 10


##if using the sigclust2 method, need to call these first
# mainTree.getLeaves() ##populate the list of all_leaves through tree travesal
# mainTree.getClusterIndices() ##set the clusterindices

##pipeline for permutation testing
def do_pipeline():
    mainTree = clusterTree("PRS", redo_original=True, withSave=False)  # create the tree
    mainTree.testClusters(how="random_matrix_q_nosave", nsim=250)


##after correction
def count_sigs(filename="~/Desktop/corrected_first_25_prs_100_sims.csv", alpha=0.05):
    stackmat = pd.read_csv(filename).set_index("Unnamed: 0")
    record = stackmat.to_dict()
    numsigs = {}
    for prsName in record.keys():
        numsigs[prsName] = 0
        prevnumsigs = numsigs
        print("starting PRS: " + str(prsName))
        ##each entry is itself a dictionary
        phenos = pd.Series(record[prsName])
        phenos[phenos.isna()] = 1  ##saves us trouble later on
        phenos = phenos.sort_values()
        i = 1
        while i <= len(phenos):
            if phenos[i - 1] < alpha:
                numsigs[prsName] += 1
            i += 1
    numsigs = pd.Series(numsigs)
    numsigs.to_csv("~/Desktop/numsigassocs_25.csv")
    return numsigs
