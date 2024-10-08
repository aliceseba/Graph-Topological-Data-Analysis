from .GTDA_utils import find_components
import numpy as np
import scipy.sparse as sp
from collections import defaultdict, Counter
import itertools
import seaborn as sns
from tqdm import tqdm
import copy

def is_overlap(x,y):
    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        if (xi[0] >= yi[0] and xi[0] <= yi[1]) or (yi[0] >= xi[0] and yi[0] <= xi[1]):
            continue
        else:
            return False
    return True


"""
nn_model: an instance of NN_model class
labels_to_eval: list of Int, choose which labels to split
nbins: Int, number of bins along each len
overlap: Float, overlap ratio 
    Unlike GTDA, we follow the exact definition of mapper by making each bin have the same length 
    and the overlap ratio between two adjacent bins b_i, b_{i+1} is |b_i intersect b_{i+1}}|/|b_i|
extra_lens: Numpy array, any extra lens to use for splitting
"""

class TDA(object):
    def __init__(self,nn_model,labels_to_eval,verbose=False):
        self.A = nn_model.A
        self.preds = np.copy(nn_model.preds)
        self.labels_to_eval = copy.copy(labels_to_eval)
        self.verbose = verbose
    
    def _compute_bin_ubs(self,bin_id,overlap,col_id,nbins):
        r = self.bin_sizes[col_id]*overlap
        gap = self.bin_sizes[col_id]-r
        if bin_id >= nbins or bin_id < 0:
            return -1*float("inf")
        curr_ub = self.pre_lbs[col_id]+gap*(1+bin_id)
        if bin_id != nbins-1:
            curr_ub += r
        return curr_ub

    def build_mixing_matrix(
        self,selected_nodes=None,normalize=True,extra_lens=None,standardize=False):
        if selected_nodes is None:
            selected_nodes = list(range(self.preds.shape[0]))
        Ar = (self.A>0).astype(np.float64)
        M = np.copy(self.preds)
        if extra_lens is not None:
            M = np.hstack([M,extra_lens])
        selected_col = self.labels_to_eval
        if extra_lens is not None:
            selected_col += list(range(self.preds.shape[1],M.shape[1]))
        M = M[selected_nodes,:][:,selected_col].copy()
        if standardize:
            for i in range(M.shape[1]):
                M[:,i] = (M[:,i]-np.mean(M[:,i]))/np.std(M[:,i])
        if normalize:
            for i in range(M.shape[1]):
                if np.max(M[:,i]) != np.min(M[:,i]):
                    M[:,i] = (M[:,i]-np.min(M[:,i]))/(np.max(M[:,i])-np.min(M[:,i]))
        return M,Ar
    
    def compute_bin_id(self,point,overlap,nbins):
        assignments = []
        for j,val in enumerate(point):
            bin_size = self.bin_sizes[j]
            r = bin_size*overlap
            if val >= self.pre_ubs[j]-r:
                bin_id = nbins-1
            elif val == self.pre_lbs[j]:
                bin_id = 0
            else:
                bin_id = int(
                    np.floor((val-self.pre_lbs[j])/(bin_size-r)))
            bin_ids = [bin_id]
            new_bin_id = bin_id-1
            if val <= self._compute_bin_ubs(new_bin_id,overlap,j,nbins):
                bin_ids.append(new_bin_id)
            assignments.append(bin_ids)
        return itertools.product(*assignments)

    def _find_bins(self,M,overlap,nbins):
        bin_nums = 0
        self.bin_key_map = {}
        self.bin_sizes = np.zeros(M.shape[1])
        self.pre_lbs = np.zeros(M.shape[1])
        self.pre_ubs = np.zeros(M.shape[1])
        for col in range(M.shape[1]):
            self.pre_lbs[col] = np.min(M[:,col])
            self.pre_ubs[col] = np.max(M[:,col])
            self.bin_sizes[col] = (self.pre_ubs[col]-self.pre_lbs[col])/(nbins-(nbins-1)*overlap)
        bin_map = {}
        bins = defaultdict(list)
        print("Generate bins...")
        for i in tqdm(range(M.shape[0]),disable=1-self.verbose):
            point = M[i,:]
            for bin_key in self.compute_bin_id(point,overlap,nbins):
                if bin_key not in self.bin_key_map:
                    self.bin_key_map[bin_key] = bin_nums
                    bin_map[bin_nums] = bin_key
                    bins[bin_nums].append(i)
                    bin_nums += 1
                else:
                    assigned_id = self.bin_key_map[bin_key]
                    bins[assigned_id].append(i)
        return bins
    

    def find_reeb_nodes(self,M,Ar,nbins=2,overlap=0.1,cluster_fn=None):
        self.bins = self._find_bins(M,overlap,nbins)
        self.final_components = {}
        self.component_bin_id = {}
        self.bin_component_id = defaultdict(list)
        self.num_total_components = 0
        print("Find reeb nodes...")
        for bin_id,curr_bin in tqdm(self.bins.items(),disable=1-self.verbose):
            curr_bin = np.array(curr_bin)
            if cluster_fn is not None:
                cluster_labels = cluster_fn.fit_predict(Ar[curr_bin,:])
                components = defaultdict(list)
                for i,l in enumerate(cluster_labels):
                    if l != -1:
                        components[l].append(i)
                components = list(components.values())
            else:
                _,components = find_components(Ar[curr_bin,:][:,curr_bin],size_thd=0)
            for component in components:
                self.final_components[self.num_total_components] = curr_bin[component].tolist()
                self.component_bin_id[self.num_total_components] = bin_id
                self.bin_component_id[bin_id].append(self.num_total_components)
                self.num_total_components += 1
        self._remove_duplicate_components()
    
    def _remove_duplicate_components(self):
        all_c = sorted([
            sorted(self.final_components[key]) for key in self.final_components.keys()])
        filtered_c = list(k for k,_ in itertools.groupby(all_c))
        self.final_components_unique = {i:c for i,c in enumerate(filtered_c)}

    def build_reeb_graph(self,M):
        all_edge_index = [[], []]
        print("Build reeb graph...")
        reeb_dim = np.max(list(self.final_components_unique.keys()))+1
        ei,ej = [],[]
        for key,c in self.final_components_unique.items():
            ei += [key]*len(c)
            ej += c
        bipartite_g = sp.csr_matrix((np.ones(len(ei)),(ei,ej)),shape=(reeb_dim,M.shape[0]))
        bipartite_g_t = bipartite_g.T.tocsr()
        ei,ej = [],[]
        for i in tqdm(self.final_components_unique.keys(),disable=1-self.verbose):
            neighs = set(bipartite_g_t[bipartite_g[i,:].indices].indices)
            neighs.remove(i)
            neighs = list(neighs)
            all_edge_index[0] += [i]*len(neighs)
            all_edge_index[1] += neighs
        A_tmp = sp.csr_matrix(
            (np.ones(len(all_edge_index[1])),(all_edge_index[0],all_edge_index[1])),shape=(
                reeb_dim,reeb_dim))
        A_tmp = ((A_tmp+A_tmp.T)>0).astype(np.float64)
        return A_tmp