# train Airports USA
from torch_geometric.datasets import Airports
import numpy as np
import scipy.sparse as sp
from argparse import Namespace
import torch
from torch import cat
import torch.nn.functional as F
from torch_geometric.utils import degree
from sklearn.metrics import confusion_matrix 
import copy
import os.path as osp
from GTDA.GTDA_utils import *
from GTDA.models import GIN, GCN, SAGE, MLP

torch.manual_seed(12345)

dict = {1:'USA', 2:'Europe', 3:'Brazil'}
path = '\Graph-Topological-Data-Analysis\dataset'
name=dict[2]
dataset = Airports(root=path, name=name)

n_nodes = dataset._data.num_nodes
n_edges = dataset._data.num_edges 

args = {
    "train_rate": 0.5,
    "val_rate": 0.1,
    "lr": 1e-4,
    "weight_decay":5e-2,
    "hidden": [64, 128, 256, 512],
    "epochs": 1500,
    "early_stopping": 800,
    "dropout": 0.01,
    "num_layers": [2]
}
args = Namespace(**args)

# %%
def run(args, dataset, data, model):

    def train(model, optimizer, data):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)[data.train_mask]
        out = F.log_softmax(out, dim=1)
        loss = F.nll_loss(out, data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        del out

    def test(model, data):
        model.eval()
        logits, accs, losses, preds = model(data.x, data.edge_index), [], [], []
        logits = F.log_softmax(logits, dim=1)
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(F.log_softmax(model(data.x, data.edge_index)[mask], dim=1), data.y[mask])

            accs.append(acc)
            losses.append(loss.detach().cpu())
        preds = logits.max(1)[1]
        return accs, preds, losses

    device = torch.device('cpu')
    
    data = random_splits(
        data, dataset.num_classes, train_percent=args.train_rate,
        val_percent=args.val_rate,seed=123)
    model, data = model.to(device), data.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    best_model = None
    for epoch in range(args.epochs):
        train(model, optimizer, data)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)
        
        if epoch%30 == 0:
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {tmp_test_acc:.4f}')
            print(f'Train Loss: {train_loss:.4f}, Val loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            best_model = copy.deepcopy(model.cpu())
            model.to(device)

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.min().item():
                    break
    if best_model is None:
        best_model = copy.deepcopy(model.cpu())
    best_model.to(device)
    return test_acc, best_val_acc, best_val_loss, preds, best_model, data

A = sp.csr_matrix((np.ones(n_edges), (dataset.edge_index[0], dataset.edge_index[1])), (n_nodes, n_nodes))
A = ((A+A.T)>0).astype(int)
X = dataset.x
x_degree = (A+A.T).sum(0).tolist()[0]
x_degree=np.array(x_degree).astype(int)
for k in range(n_nodes):
    X[k,k]=x_degree[k]


dataset_g = data_generator(G=A,X=X,labels=dataset.y.numpy(),name=name)
data = dataset_g[0]

torch.manual_seed(123)
models = {'GIN': GIN, 'GCN':GCN}
name_model = 'GIN'
train_rates = [0.3, 0.5, 0.7]
best_acc = 0
for train_rate in train_rates:
    args.train_rate = train_rate
    for hidden in args.hidden:
        for n_layers in args.num_layers:
            model = models[name_model](dataset.num_node_features, hidden, dataset.num_classes, n_layers, args.dropout)
            # model.reset_parameters()
            print(model, f'hidden: {hidden}, n_layers: {n_layers}')
            test_acc, best_val_acc, val_loss, preds, model, data = run(args, dataset, data, model)
            print(f'test: {test_acc:.4f}, val:{best_val_acc:.4f}')

            if test_acc>best_acc:
                best_acc = test_acc
                best_model = copy.deepcopy(model.cpu())
            if best_model is None:
                best_model = copy.deepcopy(model.cpu())
            best_model.to('cpu')

print(best_acc)

# ## save results ##
best_model.eval()
with torch.set_grad_enabled(False):
    tmp = best_model(data.x, data.edge_index)
    tmp = F.softmax(tmp, dim=1)
    preds = tmp.cpu().detach().numpy()

savepath = f"dataset/precomputed/{name}"
if not osp.isdir(savepath):
    os.makedirs(savepath)
np.save(f"{savepath}/prediction_lens.npy",preds)
train_nodes = np.nonzero(data.train_mask.cpu().detach().numpy())[0]
val_nodes = np.nonzero(data.val_mask.cpu().detach().numpy())[0]
test_nodes = np.nonzero(data.test_mask.cpu().detach().numpy())[0]
with open(f"{savepath}/train_nodes.txt","w") as f:
    for node in train_nodes:
        f.write(f"{node}\n")
with open(f"{savepath}/val_nodes.txt","w") as f:
    for node in val_nodes:
        f.write(f"{node}\n")
with open(f"{savepath}/test_nodes.txt","w") as f:
    for node in test_nodes:
        f.write(f"{node}\n")
