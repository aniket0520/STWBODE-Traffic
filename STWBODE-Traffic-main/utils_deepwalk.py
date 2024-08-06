import os
import csv
import numpy as np
from fastdtw import fastdtw
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import networkx as nx
from gensim.models import Word2Vec

files = {
    'pems03': ['PEMS03/pems03.npz', 'PEMS03/distance.csv'],
    'pems04': ['PEMS04/pems04.npz', 'PEMS04/distance.csv'],
    'pems07': ['PEMS07/pems07.npz', 'PEMS07/distance.csv'],
    'pems08': ['PEMS08/pems08.npz', 'PEMS08/distance.csv'],
    'pemsbay': ['PEMSBAY/pems_bay.npz', 'PEMSBAY/distance.csv'],
    'pemsD7M': ['PeMSD7M/PeMSD7M.npz', 'PeMSD7M/distance.csv'],
    'pemsD7L': ['PeMSD7L/PeMSD7L.npz', 'PeMSD7L/distance.csv']
}

def generate_graph(num_nodes, distance_matrix, threshold=0.5):
    G = nx.Graph()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if distance_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=distance_matrix[i, j])
    return G

def random_walk(G, start_node, walk_length):
    walk = [start_node]
    for _ in range(walk_length - 1):
        neighbors = list(G.neighbors(walk[-1]))
        if len(neighbors) > 0:
            next_node = np.random.choice(neighbors)
            walk.append(next_node)
        else:
            break
    return walk

def generate_random_walks(G, num_walks, walk_length):
    walks = []
    for _ in range(num_walks):
        for node in G.nodes():
            walks.append(random_walk(G, node, walk_length))
    return walks

def train_word2vec(walks, embedding_dim):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, vector_size=embedding_dim, window=5, min_count=0, sg=1, workers=4, epochs=10)
    return model

def read_data(args):
    """read data, generate spatial adjacency matrix and semantic adjacency matrix by dtw"""

    filename = args.filename
    file = files[filename]
    filepath = "./data/"
    if args.remote:
        filepath = '/home/lantu.lqq/ftemp/data/'
    data = np.load(filepath + file[0])['data']
    mean_value = np.mean(data, axis=(0, 1)).reshape(1, 1, -1)
    std_value = np.std(data, axis=(0, 1)).reshape(1, 1, -1)
    data = (data - mean_value) / std_value
    mean_value = mean_value.reshape(-1)[0]
    std_value = std_value.reshape(-1)[0]

    if not os.path.exists(f'data/{filename}_dtw_distance.npy'):
        data_mean = np.mean([data[:, :, 0][24*12*i: 24*12*(i+1)] for i in range(data.shape[0]//(24*12))], axis=0)
        data_mean = data_mean.squeeze().T 
        dtw_distance = np.zeros((data_mean.shape[0], data_mean.shape[0]))
        for i in tqdm(range(data_mean.shape[0])):
            for j in range(i, data_mean.shape[0]):
                dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
        for i in range(data_mean.shape[0]):
            for j in range(i):
                dtw_distance[i][j] = dtw_distance[j][i]
        np.save(f'data/{filename}_dtw_distance.npy', dtw_distance)

    dist_matrix = np.load(f'data/{filename}_dtw_distance.npy')

    if not os.path.exists(f'data/{filename}_spatial_distance.npy'):
        with open(filepath + file[1], 'r') as fp:
            dist_matrix = np.zeros((data.shape[1], data.shape[1])) + np.float('inf')
            file = csv.reader(fp)
            for line in file:
                break
            for line in file:
                start = int(line[0])
                end = int(line[1])
                dist_matrix[start][end] = float(line[2])
                dist_matrix[end][start] = float(line[2])
            np.save(f'data/{filename}_spatial_distance.npy', dist_matrix)

    dist_matrix = np.load(f'data/{filename}_spatial_distance.npy')
    num_nodes = data.shape[1]

    G = generate_graph(num_nodes, dist_matrix)
    walks = generate_random_walks(G, num_walks=10, walk_length=80)
    model = train_word2vec(walks, embedding_dim=64)

    node_embeddings = {int(node): model.wv[node] for node in model.wv.index_to_key}

    # Assuming node embeddings should be used to create adjacency matrices
    dtw_matrix = np.zeros((num_nodes, num_nodes))
    sp_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dtw_matrix[i, j] = np.linalg.norm(node_embeddings[i] - node_embeddings[j])
                sp_matrix[i, j] = np.linalg.norm(node_embeddings[i] - node_embeddings[j])

    print("dtw_matrix shape:", dtw_matrix.shape)
    print("sp_matrix shape:", sp_matrix.shape)

    return torch.from_numpy(data.astype(np.float32)), mean_value, std_value, dtw_matrix, sp_matrix



def get_normalized_adj(A):
    """
    Returns a tensor, the degree normalized adjacency matrix.
    """
    if A.ndim != 2:
        raise ValueError("The adjacency matrix A must be 2-dimensional")

    print("Adjacency matrix A shape:", A.shape)

    alpha = 0.8
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    print("Degree matrix D shape:", D.shape)
    
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A), diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)

    print("Normalized adjacency matrix A_reg shape:", A_reg.shape)
    return torch.from_numpy(A_reg.astype(np.float32))



class MyDataset(Dataset):
    def __init__(self, data, split_start, split_end, his_length, pred_length):
        split_start = int(split_start)
        split_end = int(split_end)
        self.data = data[split_start: split_end]
        self.his_length = his_length
        self.pred_length = pred_length
    
    def __getitem__(self, index):
        x = self.data[index: index + self.his_length].permute(1, 0, 2)
        y = self.data[index + self.his_length: index + self.his_length + self.pred_length][:, :, 0].permute(1, 0)
        return torch.Tensor(x), torch.Tensor(y)
    
    def __len__(self):
        return self.data.shape[0] - self.his_length - self.pred_length + 1


def generate_dataset(data, args):
    """
    Args:
        data: input dataset, shape like T * N
        batch_size: int 
        train_ratio: float, the ratio of the dataset for training
        his_length: the input length of time series for prediction
        pred_length: the target length of time series of prediction

    Returns:
        train_dataloader: torch tensor, shape like batch * N * his_length * features
        test_dataloader: torch tensor, shape like batch * N * pred_length * features
    """
    batch_size = args.batch_size
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    his_length = args.his_length
    pred_length = args.pred_length

    train_dataset = MyDataset(data, 0, data.shape[0] * train_ratio, his_length, pred_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = MyDataset(data, data.shape[0] * train_ratio, data.shape[0] * (train_ratio + valid_ratio), his_length, pred_length)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MyDataset(data, data.shape[0] * (train_ratio + valid_ratio), data.shape[0], his_length, pred_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader
