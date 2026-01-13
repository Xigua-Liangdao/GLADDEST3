import os
import pickle
from itertools import repeat

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    Batch,
)
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import random_split
from sklearn.model_selection import StratifiedKFold

from datasets.abstract_dataset import AbstractDatasetInfos


class SpectreDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.dataset_name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        if self.name in ['Tox21_p53', 'Tox21_HSE', 'Tox21_MMP', 'Tox21_PPAR_gamma']:
            split_counts_path = os.path.join(self.processed_dir, 'split_counts.pt')
            if os.path.exists(split_counts_path):
                self.split_counts = torch.load(split_counts_path)

    @property
    def raw_file_names(self):
        if self.name == "sbm":
            return ["sbm.pt"]
        elif self.name == "comm20":
            return ["community.pt"]
        elif self.name == "planar":
            return ["planar.pt"]
        elif self.name == "tree":
            return ["tree.pt"]
        elif self.name in ["ego", "protein"]:
            return [f"{self.name}_split.pkl"]
        elif self.name == "bzr":
            prefix = self.name.upper()
            return [f"{prefix}_A.txt", f"{prefix}_graph_labels.txt", f"{prefix}_graph_indicator.txt"]
        elif self.name == "dhfr":
            prefix = self.name.upper()
            return [f"{prefix}_A.txt", f"{prefix}_graph_labels.txt", f"{prefix}_graph_indicator.txt"]
        elif self.name == "cox2":
            prefix = self.name.upper()
            return [f"{prefix}_A.txt", f"{prefix}_graph_labels.txt", f"{prefix}_graph_indicator.txt", f"{prefix}_node_labels.txt", f"{prefix}_node_attributes.txt"]
        elif self.name == "BZ_CO":
            # BZ_CO mixed dataset: BZR as training (ID), COX2 as testing (OOD)
            return ["BZR_A.txt", "BZR_graph_labels.txt", "BZR_graph_indicator.txt", "BZR_node_labels.txt", "BZR_node_attributes.txt",
                    "COX2_A.txt", "COX2_graph_labels.txt", "COX2_graph_indicator.txt", "COX2_node_labels.txt", "COX2_node_attributes.txt"]
        elif self.name == "EN_PR":
            # EN_PR mixed dataset: ENZYMES as training (ID), PROTEINS as testing (OOD)
            return ["ENZYMES_A.txt", "ENZYMES_graph_labels.txt", "ENZYMES_graph_indicator.txt", "ENZYMES_node_labels.txt", "ENZYMES_node_attributes.txt",
                    "PROTEINS_A.txt", "PROTEINS_graph_labels.txt", "PROTEINS_graph_indicator.txt", "PROTEINS_node_labels.txt", "PROTEINS_node_attributes.txt"]
        elif self.name == 'Tox21_p53':
            return ['Tox21_p53_training', 'Tox21_p53_evaluation', 'Tox21_p53_testing']
        elif self.name == 'Tox21_HSE':
            return ['Tox21_HSE_training', 'Tox21_HSE_evaluation', 'Tox21_HSE_testing']
        elif self.name == 'Tox21_MMP':
            return ['Tox21_MMP_training', 'Tox21_MMP_evaluation', 'Tox21_MMP_testing']
        elif self.name == 'Tox21_PPAR_gamma':
            return ['Tox21_PPAR_gamma_training', 'Tox21_PPAR_gamma_evaluation', 'Tox21_PPAR_gamma_testing']
        elif self.name == 'AIDS':
            return ['AIDS_training', 'AIDS_evaluation', 'AIDS_testing']
        elif self.name == 'COLLAB':
            return ['COLLAB_training', 'COLLAB_evaluation', 'COLLAB_testing']
        elif self.name == 'DD':
            return ['DD_training', 'DD_evaluation', 'DD_testing']
        elif self.name == 'ENZYMES':
            return ['ENZYMES_training', 'ENZYMES_evaluation', 'ENZYMES_testing']
        elif self.name == 'NCI1' :
            return ['NCI1_training', 'NCI1_evaluation', 'NCI1_testing']
        elif self.name == 'IMDB-BINARY':
            return ['IMDB-BINARY_training', 'IMDB-BINARY_evaluation', 'IMDB-BINARY_testing']
        else:
            raise NotImplementedError(f"Dataset {self.name} not implemented.")

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass # Data is assumed to be in the root directory

    def process(self):
        pyg_list = []
        if self.name == "bzr":
            prefix = self.name.upper()
            # Corrected and simplified logic for BZR
            graph_indicator_path = os.path.join(self.raw_dir, f'{prefix}_graph_indicator.txt')
            graph_indicator = np.loadtxt(graph_indicator_path, dtype=np.int64) - 1
            
            edge_list_path = os.path.join(self.raw_dir, f'{prefix}_A.txt')
            edges = np.loadtxt(edge_list_path, delimiter=',', dtype=np.int64) - 1
            edge_index_all = torch.from_numpy(edges.T).to(torch.int64)

            graph_labels_path = os.path.join(self.raw_dir, f'{prefix}_graph_labels.txt')
            graph_labels = torch.from_numpy(np.loadtxt(graph_labels_path, dtype=np.int64))
            graph_labels[graph_labels == -1] = 0  # Remap labels from {-1, 1} to {0, 1}

            num_graphs = graph_labels.size(0)
            slices = torch.from_numpy(np.bincount(graph_indicator)).cumsum(0)
            slices = torch.cat([torch.tensor([0]), slices])

            for i in range(num_graphs):
                start, end = slices[i], slices[i+1]
                
                edge_mask = (edge_index_all[0] >= start) & (edge_index_all[0] < end)
                sub_edge_index = edge_index_all[:, edge_mask] - start
                
                num_sub_nodes = end - start
                
                edge_attr = torch.zeros(sub_edge_index.size(1), 2, dtype=torch.float)
                edge_attr[:, 1] = 1

                data = Data(
                    x=torch.ones(num_sub_nodes, 1),
                    edge_index=sub_edge_index,
                    edge_attr=edge_attr,
                    y=torch.tensor([[graph_labels[i].item()]], dtype=torch.long),
                    n_nodes=torch.tensor([num_sub_nodes])
                )
                pyg_list.append(data)
        elif self.name == "dhfr":
            prefix = self.name.upper()
            # DHFR processing logic, similar to BZR (both have node_attributes and node_labels)
            graph_indicator_path = os.path.join(self.raw_dir, f'{prefix}_graph_indicator.txt')
            graph_indicator = np.loadtxt(graph_indicator_path, dtype=np.int64) - 1
            
            edge_list_path = os.path.join(self.raw_dir, f'{prefix}_A.txt')
            edges = np.loadtxt(edge_list_path, delimiter=',', dtype=np.int64) - 1
            edge_index_all = torch.from_numpy(edges.T).to(torch.int64)

            graph_labels_path = os.path.join(self.raw_dir, f'{prefix}_graph_labels.txt')
            graph_labels = torch.from_numpy(np.loadtxt(graph_labels_path, dtype=np.int64))
            graph_labels[graph_labels == -1] = 0  # Remap labels from {-1, 1} to {0, 1}
            
            # Load node labels and attributes
            node_labels_path = os.path.join(self.raw_dir, f'{prefix}_node_labels.txt')
            node_labels = torch.from_numpy(np.loadtxt(node_labels_path, delimiter=',', dtype=np.int64))
            
            node_attrs_path = os.path.join(self.raw_dir, f'{prefix}_node_attributes.txt')
            node_attrs = torch.from_numpy(np.loadtxt(node_attrs_path, delimiter=',', dtype=np.float32))
            
            # One-hot encode node labels
            num_node_classes = node_labels.max() + 1
            node_features_onehot = F.one_hot(node_labels, num_classes=num_node_classes).float()
            
            # Merge node features: attributes + one-hot labels
            node_features = torch.cat([node_attrs, node_features_onehot], dim=1)

            num_graphs = graph_labels.size(0)
            slices = torch.from_numpy(np.bincount(graph_indicator)).cumsum(0)
            slices = torch.cat([torch.tensor([0]), slices])

            for i in range(num_graphs):
                start, end = slices[i], slices[i+1]
                
                edge_mask = (edge_index_all[0] >= start) & (edge_index_all[0] < end)
                sub_edge_index = edge_index_all[:, edge_mask] - start
                
                num_sub_nodes = end - start
                
                # Simple edge features
                edge_features = torch.ones(sub_edge_index.size(1), 1).float()

                data = Data(
                    x=node_features[start:end],
                    edge_index=sub_edge_index,
                    edge_attr=edge_features,
                    y=torch.tensor([[graph_labels[i].item()]], dtype=torch.long),
                    n_nodes=torch.tensor([num_sub_nodes])
                )
                pyg_list.append(data)
        elif self.name == "cox2":
            prefix = self.name.upper()
            # COX2 processing logic, similar to DHFR (both have node_attributes and node_labels)
            graph_indicator_path = os.path.join(self.raw_dir, f'{prefix}_graph_indicator.txt')
            graph_indicator = np.loadtxt(graph_indicator_path, dtype=np.int64) - 1
            
            edge_list_path = os.path.join(self.raw_dir, f'{prefix}_A.txt')
            edges = np.loadtxt(edge_list_path, delimiter=',', dtype=np.int64) - 1
            edge_index_all = torch.from_numpy(edges.T).to(torch.int64)

            graph_labels_path = os.path.join(self.raw_dir, f'{prefix}_graph_labels.txt')
            graph_labels = torch.from_numpy(np.loadtxt(graph_labels_path, dtype=np.int64))
            graph_labels[graph_labels == -1] = 0  # Remap labels from {-1, 1} to {0, 1}
            
            # Load node labels and attributes
            node_labels_path = os.path.join(self.raw_dir, f'{prefix}_node_labels.txt')
            node_labels = torch.from_numpy(np.loadtxt(node_labels_path, delimiter=',', dtype=np.int64))
            
            node_attrs_path = os.path.join(self.raw_dir, f'{prefix}_node_attributes.txt')
            node_attrs = torch.from_numpy(np.loadtxt(node_attrs_path, delimiter=',', dtype=np.float32))
            
            # One-hot encode node labels
            num_node_classes = node_labels.max() + 1
            node_features_onehot = F.one_hot(node_labels, num_classes=num_node_classes).float()
            
            # Merge node features: attributes + one-hot labels
            node_features = torch.cat([node_attrs, node_features_onehot], dim=1)

            num_graphs = graph_labels.size(0)
            slices = torch.from_numpy(np.bincount(graph_indicator)).cumsum(0)
            slices = torch.cat([torch.tensor([0]), slices])

            for i in range(num_graphs):
                start, end = slices[i], slices[i+1]
                
                edge_mask = (edge_index_all[0] >= start) & (edge_index_all[0] < end)
                sub_edge_index = edge_index_all[:, edge_mask] - start
                
                num_sub_nodes = end - start
                
                # Simple edge features
                edge_features = torch.ones(sub_edge_index.size(1), 1).float()

                data = Data(
                    x=node_features[start:end],
                    edge_index=sub_edge_index,
                    edge_attr=edge_features,
                    y=torch.tensor([[graph_labels[i].item()]], dtype=torch.long),
                    n_nodes=torch.tensor([num_sub_nodes])
                )
                pyg_list.append(data)
        elif self.name in ['AIDS']:
            prefix = self.name.upper()
            graph_indicator_path = os.path.join(self.raw_dir, f'{prefix}_graph_indicator.txt')
            graph_indicator = np.loadtxt(graph_indicator_path, dtype=np.int64) - 1
            
            edge_list_path = os.path.join(self.raw_dir, f'{prefix}_A.txt')
            edges = np.loadtxt(edge_list_path, delimiter=',', dtype=np.int64) - 1
            edge_index_all = torch.from_numpy(edges.T).to(torch.int64)

            graph_labels_path = os.path.join(self.raw_dir, f'{prefix}_graph_labels.txt')
            graph_labels = torch.from_numpy(np.loadtxt(graph_labels_path, dtype=np.int64))
            graph_labels[graph_labels == -1] = 0

            node_labels_path = os.path.join(self.raw_dir, f'{prefix}_node_labels.txt')
            node_labels = torch.from_numpy(np.loadtxt(node_labels_path, delimiter=',', dtype=np.int64))
            node_labels = F.one_hot(node_labels, num_classes=node_labels.max() + 1).float()

            edge_labels_path = os.path.join(self.raw_dir, f'{prefix}_edge_labels.txt')
            edge_labels = torch.from_numpy(np.loadtxt(edge_labels_path, delimiter=',', dtype=np.int64))
            edge_labels = F.one_hot(edge_labels, num_classes=edge_labels.max() + 1).float()

            num_graphs = graph_labels.size(0)
            slices = torch.from_numpy(np.bincount(graph_indicator)).cumsum(0)
            slices = torch.cat([torch.tensor([0]), slices])

            for i in range(num_graphs):
                start, end = slices[i], slices[i+1]
                
                edge_mask = (edge_index_all[0] >= start) & (edge_index_all[0] < end)
                sub_edge_index = edge_index_all[:, edge_mask] - start
                
                num_sub_nodes = end - start

                data = Data(
                    x=node_labels[start:end],
                    edge_index=sub_edge_index,
                    edge_attr=edge_labels[edge_mask],
                    y=torch.tensor([[graph_labels[i].item()]], dtype=torch.long),
                    n_nodes=torch.tensor([num_sub_nodes])
                )
                pyg_list.append(data)
        elif self.name in ['COLLAB', 'IMDB-BINARY']:
            prefix = self.name.upper()
            graph_indicator_path = os.path.join(self.raw_dir, f'{prefix}_graph_indicator.txt')
            graph_indicator = np.loadtxt(graph_indicator_path, dtype=np.int64) - 1
            
            edge_list_path = os.path.join(self.raw_dir, f'{prefix}_A.txt')
            edges = np.loadtxt(edge_list_path, delimiter=',', dtype=np.int64) - 1
            edge_index_all = torch.from_numpy(edges.T).to(torch.int64)

            graph_labels_path = os.path.join(self.raw_dir, f'{prefix}_graph_labels.txt')
            graph_labels = torch.from_numpy(np.loadtxt(graph_labels_path, dtype=np.int64))
            graph_labels[graph_labels == -1] = 0

            num_graphs = graph_labels.size(0)
            slices = torch.from_numpy(np.bincount(graph_indicator)).cumsum(0)
            slices = torch.cat([torch.tensor([0]), slices])

            # Compute label distribution to determine anomaly/normal classes
            unique, counts = np.unique(graph_labels.numpy(), return_counts=True)
            if self.name == 'IMDB-BINARY':
                anomaly_label = unique[np.argmin(counts)]
                normal_label = unique[np.argmax(counts)]
            elif self.name == 'COLLAB':
                anomaly_label = unique[np.argmin(counts)]
                normal_labels = [l for l in unique if l != anomaly_label]

            # Shuffle indices globally
            indices = np.random.permutation(num_graphs)
            new_pyg_list = []
            for i in indices:
                start, end = slices[i], slices[i+1]
                num_sub_nodes = end - start
                edge_mask = (edge_index_all[0] >= start) & (edge_index_all[0] < end)
                sub_edge_index = edge_index_all[:, edge_mask] - start
                node_features = torch.ones(num_sub_nodes, 1).float()
                edge_features = torch.ones(sub_edge_index.size(1), 1).float()
                label = graph_labels[i].item()
                y = 1 if label == anomaly_label else 0  # 1 means anomaly, 0 means normal
                data = Data(
                    x=node_features,
                    edge_index=sub_edge_index,
                    edge_attr=edge_features,
                    y=torch.tensor([[y]], dtype=torch.long),
                    n_nodes=torch.tensor([num_sub_nodes])
                )
                new_pyg_list.append(data)
            pyg_list = new_pyg_list
        elif self.name in ['DD', 'NCI1']:
            prefix = self.name.upper()
            graph_indicator_path = os.path.join(self.raw_dir, f'{prefix}_graph_indicator.txt')
            graph_indicator = np.loadtxt(graph_indicator_path, dtype=np.int64) - 1
            
            edge_list_path = os.path.join(self.raw_dir, f'{prefix}_A.txt')
            edges = np.loadtxt(edge_list_path, delimiter=',', dtype=np.int64) - 1
            edge_index_all = torch.from_numpy(edges.T).to(torch.int64)

            graph_labels_path = os.path.join(self.raw_dir, f'{prefix}_graph_labels.txt')
            graph_labels = torch.from_numpy(np.loadtxt(graph_labels_path, dtype=np.int64))
            graph_labels[graph_labels == -1] = 0
            
            node_labels_path = os.path.join(self.raw_dir, f'{prefix}_node_labels.txt')
            node_labels = torch.from_numpy(np.loadtxt(node_labels_path, delimiter=',', dtype=np.int64))
            node_labels = F.one_hot(node_labels, num_classes=node_labels.max() + 1).float()

            num_graphs = graph_labels.size(0)
            slices = torch.from_numpy(np.bincount(graph_indicator)).cumsum(0)
            slices = torch.cat([torch.tensor([0]), slices])
            if self.name == 'DD':
                # Apply node count filtering for DD dataset
                min_num_nodes = 0
                max_num_nodes = 50000

            for i in range(num_graphs):
                start, end = slices[i], slices[i+1]
                num_sub_nodes = end - start
                
                # Filter by node count only for DD
                if self.name == 'DD':
                    if num_sub_nodes < min_num_nodes or num_sub_nodes > max_num_nodes:
                        continue
                
                edge_mask = (edge_index_all[0] >= start) & (edge_index_all[0] < end)
                sub_edge_index = edge_index_all[:, edge_mask] - start
                
                # Use the actual node features from one-hot encoding
                node_features = node_labels[start:end]
                # Create dummy edge features
                edge_features = torch.ones(sub_edge_index.size(1), 1).float()

                data = Data(
                    x=node_features,
                    edge_index=sub_edge_index,
                    edge_attr=edge_features,
                    y=torch.tensor([[graph_labels[i].item()]], dtype=torch.long),
                    n_nodes=torch.tensor([num_sub_nodes])
                )
                pyg_list.append(data)
        elif self.name == 'ENZYMES':
            # ENZYMES special handling: load raw data, shuffle globally, keep true labels only for test
            prefix = self.name.upper()
            graph_indicator_path = os.path.join(self.raw_dir, f'{prefix}_graph_indicator.txt')
            graph_indicator = np.loadtxt(graph_indicator_path, dtype=np.int64) - 1
            
            edge_list_path = os.path.join(self.raw_dir, f'{prefix}_A.txt')
            edges = np.loadtxt(edge_list_path, delimiter=',', dtype=np.int64) - 1
            edge_index_all = torch.from_numpy(edges.T).to(torch.int64)

            graph_labels_path = os.path.join(self.raw_dir, f'{prefix}_graph_labels.txt')
            graph_labels = torch.from_numpy(np.loadtxt(graph_labels_path, dtype=np.int64))
            # Keep original labels (1..6); no remapping
            
            node_labels_path = os.path.join(self.raw_dir, f'{prefix}_node_labels.txt')
            node_labels = torch.from_numpy(np.loadtxt(node_labels_path, delimiter=',', dtype=np.int64))
            node_labels = F.one_hot(node_labels, num_classes=node_labels.max() + 1).float()

            num_graphs = graph_labels.size(0)
            slices = torch.from_numpy(np.bincount(graph_indicator)).cumsum(0)
            slices = torch.cat([torch.tensor([0]), slices])

            # Build all graph data and keep original labels
            all_graph_data = []
            for i in range(num_graphs):
                start, end = slices[i], slices[i+1]
                num_sub_nodes = end - start
                
                edge_mask = (edge_index_all[0] >= start) & (edge_index_all[0] < end)
                sub_edge_index = edge_index_all[:, edge_mask] - start
                
                node_features = node_labels[start:end]
                edge_features = torch.ones(sub_edge_index.size(1), 1).float()

                # Keep raw graph data and label; splits handled later in datamodule
                graph_data = {
                    'x': node_features,
                    'edge_index': sub_edge_index,
                    'edge_attr': edge_features,
                    'n_nodes': torch.tensor([num_sub_nodes]),
                    'original_label': graph_labels[i].item()  # original label
                }
                all_graph_data.append(graph_data)
            
            # Temporarily save all data; actual split and label handling in the datamodule
            for graph_data in all_graph_data:
                data = Data(
                    x=graph_data['x'],
                    edge_index=graph_data['edge_index'],
                    edge_attr=graph_data['edge_attr'],
                    y=torch.tensor([[graph_data['original_label']]], dtype=torch.long),  # keep original label
                    n_nodes=graph_data['n_nodes']
                )
                pyg_list.append(data)
        elif self.name == 'BZ_CO':
            # BZ_CO mixed dataset: BZR for training (ID), COX2 for testing (OOD)
            # First, align node label dimensions
            
            # 1) Process BZR
            bzr_graph_indicator_path = os.path.join(self.raw_dir, 'BZR_graph_indicator.txt')
            bzr_graph_indicator = np.loadtxt(bzr_graph_indicator_path, dtype=np.int64) - 1
            
            bzr_edge_list_path = os.path.join(self.raw_dir, 'BZR_A.txt')
            bzr_edges = np.loadtxt(bzr_edge_list_path, delimiter=',', dtype=np.int64) - 1
            bzr_edge_index_all = torch.from_numpy(bzr_edges.T).to(torch.int64)

            bzr_graph_labels_path = os.path.join(self.raw_dir, 'BZR_graph_labels.txt')
            bzr_graph_labels = torch.from_numpy(np.loadtxt(bzr_graph_labels_path, dtype=np.int64))
            bzr_graph_labels[bzr_graph_labels == -1] = 0  # 重映射到{0, 1}
            
            bzr_node_labels_path = os.path.join(self.raw_dir, 'BZR_node_labels.txt')
            bzr_node_labels = torch.from_numpy(np.loadtxt(bzr_node_labels_path, delimiter=',', dtype=np.int64))
            
            bzr_node_attrs_path = os.path.join(self.raw_dir, 'BZR_node_attributes.txt')
            bzr_node_attrs = torch.from_numpy(np.loadtxt(bzr_node_attrs_path, delimiter=',', dtype=np.float32))
            
            # 2) Process COX2
            cox2_graph_indicator_path = os.path.join(self.raw_dir, 'COX2_graph_indicator.txt')
            cox2_graph_indicator = np.loadtxt(cox2_graph_indicator_path, dtype=np.int64) - 1
            
            cox2_edge_list_path = os.path.join(self.raw_dir, 'COX2_A.txt')
            cox2_edges = np.loadtxt(cox2_edge_list_path, delimiter=',', dtype=np.int64) - 1
            cox2_edge_index_all = torch.from_numpy(cox2_edges.T).to(torch.int64)

            cox2_graph_labels_path = os.path.join(self.raw_dir, 'COX2_graph_labels.txt')
            cox2_graph_labels = torch.from_numpy(np.loadtxt(cox2_graph_labels_path, dtype=np.int64))
            cox2_graph_labels[cox2_graph_labels == -1] = 0  # remap to {0,1}
            
            cox2_node_labels_path = os.path.join(self.raw_dir, 'COX2_node_labels.txt')
            cox2_node_labels = torch.from_numpy(np.loadtxt(cox2_node_labels_path, delimiter=',', dtype=np.int64))
            
            cox2_node_attrs_path = os.path.join(self.raw_dir, 'COX2_node_attributes.txt')
            cox2_node_attrs = torch.from_numpy(np.loadtxt(cox2_node_attrs_path, delimiter=',', dtype=np.float32))

            # 3) Unify node label dimension: create a global one-hot space
            all_node_labels = torch.cat([bzr_node_labels, cox2_node_labels])
            unique_labels = torch.unique(all_node_labels)
            max_label = unique_labels.max().item()
            num_node_classes = max_label + 1
            
            # One-hot encode with unified size
            bzr_node_features_onehot = F.one_hot(bzr_node_labels, num_classes=num_node_classes).float()
            cox2_node_features_onehot = F.one_hot(cox2_node_labels, num_classes=num_node_classes).float()
            
            # 4) Merge node features: attributes + one-hot labels
            bzr_node_features = torch.cat([bzr_node_attrs, bzr_node_features_onehot], dim=1)
            cox2_node_features = torch.cat([cox2_node_attrs, cox2_node_features_onehot], dim=1)
            
            # 5) Build BZR graphs
            bzr_num_graphs = bzr_graph_labels.size(0)
            bzr_slices = torch.from_numpy(np.bincount(bzr_graph_indicator)).cumsum(0)
            bzr_slices = torch.cat([torch.tensor([0]), bzr_slices])
            
            bzr_graphs = []
            for i in range(bzr_num_graphs):
                start, end = bzr_slices[i], bzr_slices[i+1]
                edge_mask = (bzr_edge_index_all[0] >= start) & (bzr_edge_index_all[0] < end)
                sub_edge_index = bzr_edge_index_all[:, edge_mask] - start
                num_sub_nodes = end - start
                
                # Simple edge features
                edge_features = torch.ones(sub_edge_index.size(1), 1).float()
                
                data = Data(
                    x=bzr_node_features[start:end],
                    edge_index=sub_edge_index,
                    edge_attr=edge_features,
                    y=torch.tensor([[bzr_graph_labels[i].item()]], dtype=torch.long),
                    n_nodes=torch.tensor([num_sub_nodes]),
                    dataset_source=torch.tensor([0])  # 0 for BZR
                )
                bzr_graphs.append(data)
            
            # 6) Build COX2 graphs
            cox2_num_graphs = cox2_graph_labels.size(0)
            cox2_slices = torch.from_numpy(np.bincount(cox2_graph_indicator)).cumsum(0)
            cox2_slices = torch.cat([torch.tensor([0]), cox2_slices])
            
            cox2_graphs = []
            for i in range(cox2_num_graphs):
                start, end = cox2_slices[i], cox2_slices[i+1]
                edge_mask = (cox2_edge_index_all[0] >= start) & (cox2_edge_index_all[0] < end)
                sub_edge_index = cox2_edge_index_all[:, edge_mask] - start
                num_sub_nodes = end - start
                
                # Simple edge features
                edge_features = torch.ones(sub_edge_index.size(1), 1).float()
                
                data = Data(
                    x=cox2_node_features[start:end],
                    edge_index=sub_edge_index,
                    edge_attr=edge_features,
                    y=torch.tensor([[cox2_graph_labels[i].item()]], dtype=torch.long),
                    n_nodes=torch.tensor([num_sub_nodes]),
                    dataset_source=torch.tensor([1])  # 1 for COX2
                )
                cox2_graphs.append(data)
            
            # 7) Concatenate graphs (BZR first, then COX2)
            pyg_list.extend(bzr_graphs)
            pyg_list.extend(cox2_graphs)
            
            # Save counts for later split
            self.bzr_graph_count = len(bzr_graphs)
            self.cox2_graph_count = len(cox2_graphs)
        elif self.name == 'EN_PR':
            # EN_PR mixed dataset: similar to BZ_CO, using ENZYMES and PROTEINS
            
            # 1) Load ENZYMES
            enzymes_graph_indicator_path = os.path.join(self.raw_dir, 'ENZYMES_graph_indicator.txt')
            enzymes_graph_indicator = np.loadtxt(enzymes_graph_indicator_path, dtype=np.int64) - 1
            
            enzymes_edge_list_path = os.path.join(self.raw_dir, 'ENZYMES_A.txt')
            enzymes_edges = np.loadtxt(enzymes_edge_list_path, delimiter=',', dtype=np.int64) - 1
            enzymes_edge_index_all = torch.from_numpy(enzymes_edges.T).to(torch.int64)
            
            enzymes_graph_labels_path = os.path.join(self.raw_dir, 'ENZYMES_graph_labels.txt')
            enzymes_graph_labels = torch.from_numpy(np.loadtxt(enzymes_graph_labels_path, dtype=np.int64))
            # Map ENZYMES labels to 0 (normal)
            enzymes_graph_labels = torch.zeros_like(enzymes_graph_labels)
            
            enzymes_node_labels_path = os.path.join(self.raw_dir, 'ENZYMES_node_labels.txt')
            enzymes_node_labels = torch.from_numpy(np.loadtxt(enzymes_node_labels_path, delimiter=',', dtype=np.int64))
            
            enzymes_node_attrs_path = os.path.join(self.raw_dir, 'ENZYMES_node_attributes.txt')
            enzymes_node_attrs = torch.from_numpy(np.loadtxt(enzymes_node_attrs_path, delimiter=',', dtype=np.float32))

            # 2) Load PROTEINS
            proteins_graph_indicator_path = os.path.join(self.raw_dir, 'PROTEINS_graph_indicator.txt')
            proteins_graph_indicator = np.loadtxt(proteins_graph_indicator_path, dtype=np.int64) - 1
            
            proteins_edge_list_path = os.path.join(self.raw_dir, 'PROTEINS_A.txt')
            proteins_edges = np.loadtxt(proteins_edge_list_path, delimiter=',', dtype=np.int64) - 1
            proteins_edge_index_all = torch.from_numpy(proteins_edges.T).to(torch.int64)
            
            proteins_graph_labels_path = os.path.join(self.raw_dir, 'PROTEINS_graph_labels.txt')
            proteins_graph_labels = torch.from_numpy(np.loadtxt(proteins_graph_labels_path, dtype=np.int64))
            # Map PROTEINS labels to 1 (anomaly)
            proteins_graph_labels = torch.ones_like(proteins_graph_labels)
            
            proteins_node_labels_path = os.path.join(self.raw_dir, 'PROTEINS_node_labels.txt')
            proteins_node_labels = torch.from_numpy(np.loadtxt(proteins_node_labels_path, delimiter=',', dtype=np.int64))
            
            proteins_node_attrs_path = os.path.join(self.raw_dir, 'PROTEINS_node_attributes.txt')
            proteins_node_attrs_raw = np.loadtxt(proteins_node_attrs_path, dtype=np.float32)
            # PROTEINS的节点属性是一维的，需要转换为二维
            if proteins_node_attrs_raw.ndim == 1:
                proteins_node_attrs = torch.from_numpy(proteins_node_attrs_raw.reshape(-1, 1))
            else:
                proteins_node_attrs = torch.from_numpy(proteins_node_attrs_raw)

            # 3) Unify node label dimension: create a global one-hot space
            all_node_labels = torch.cat([enzymes_node_labels, proteins_node_labels])
            unique_labels = torch.unique(all_node_labels)
            max_label = unique_labels.max().item()
            num_node_classes = max_label + 1
            
            # One-hot encode with unified size
            enzymes_node_features_onehot = F.one_hot(enzymes_node_labels, num_classes=num_node_classes).float()
            proteins_node_features_onehot = F.one_hot(proteins_node_labels, num_classes=num_node_classes).float()
            
            # 4) Unify attribute dimensions across datasets
            enzymes_attr_dim = enzymes_node_attrs.shape[1]  # 18维
            proteins_attr_dim = proteins_node_attrs.shape[1]  # 1维
            
            # Expand PROTEINS attr dim to match ENZYMES
            if proteins_attr_dim < enzymes_attr_dim:
                # Zero-pad PROTEINS attrs to match ENZYMES
                padding_size = enzymes_attr_dim - proteins_attr_dim
                padding = torch.zeros(proteins_node_attrs.shape[0], padding_size, dtype=proteins_node_attrs.dtype)
                proteins_node_attrs = torch.cat([proteins_node_attrs, padding], dim=1)
                # silent: avoid verbose dimension logs
            elif enzymes_attr_dim < proteins_attr_dim:
                # Zero-pad ENZYMES attrs to match PROTEINS
                padding_size = proteins_attr_dim - enzymes_attr_dim
                padding = torch.zeros(enzymes_node_attrs.shape[0], padding_size, dtype=enzymes_node_attrs.dtype)
                enzymes_node_attrs = torch.cat([enzymes_node_attrs, padding], dim=1)
                # silent: avoid verbose dimension logs
            
            # 5) Merge node features: unified attrs + one-hot labels
            enzymes_node_features = torch.cat([enzymes_node_attrs, enzymes_node_features_onehot], dim=1)
            proteins_node_features = torch.cat([proteins_node_attrs, proteins_node_features_onehot], dim=1)
            
            # silent: avoid verbose dimension logs
            
            # 6) Build ENZYMES graphs
            enzymes_num_graphs = enzymes_graph_labels.size(0)
            enzymes_slices = torch.from_numpy(np.bincount(enzymes_graph_indicator)).cumsum(0)
            enzymes_slices = torch.cat([torch.tensor([0]), enzymes_slices])
            
            enzymes_graphs = []
            for i in range(enzymes_num_graphs):
                start, end = enzymes_slices[i], enzymes_slices[i+1]
                edge_mask = (enzymes_edge_index_all[0] >= start) & (enzymes_edge_index_all[0] < end)
                sub_edge_index = enzymes_edge_index_all[:, edge_mask] - start
                num_sub_nodes = end - start
                
                # Simple edge features
                edge_features = torch.ones(sub_edge_index.size(1), 1).float()
                
                data = Data(
                    x=enzymes_node_features[start:end],
                    edge_index=sub_edge_index,
                    edge_attr=edge_features,
                    y=torch.tensor([[enzymes_graph_labels[i].item()]], dtype=torch.long),
                    n_nodes=torch.tensor([num_sub_nodes]),
                    dataset_source=torch.tensor([0])  # 0 for ENZYMES
                )
                enzymes_graphs.append(data)
            
            # 7) Build PROTEINS graphs
            proteins_num_graphs = proteins_graph_labels.size(0)
            proteins_slices = torch.from_numpy(np.bincount(proteins_graph_indicator)).cumsum(0)
            proteins_slices = torch.cat([torch.tensor([0]), proteins_slices])
            
            proteins_graphs = []
            for i in range(proteins_num_graphs):
                start, end = proteins_slices[i], proteins_slices[i+1]
                edge_mask = (proteins_edge_index_all[0] >= start) & (proteins_edge_index_all[0] < end)
                sub_edge_index = proteins_edge_index_all[:, edge_mask] - start
                num_sub_nodes = end - start
                
                # Simple edge features
                edge_features = torch.ones(sub_edge_index.size(1), 1).float()
                
                data = Data(
                    x=proteins_node_features[start:end],
                    edge_index=sub_edge_index,
                    edge_attr=edge_features,
                    y=torch.tensor([[proteins_graph_labels[i].item()]], dtype=torch.long),
                    n_nodes=torch.tensor([num_sub_nodes]),
                    dataset_source=torch.tensor([1])  # 1 for PROTEINS
                )
                proteins_graphs.append(data)
            
            # 8) Concatenate graphs (ENZYMES first, then PROTEINS)
            pyg_list.extend(enzymes_graphs)
            pyg_list.extend(proteins_graphs)
            
            # Save counts for later split
            self.enzymes_graph_count = len(enzymes_graphs)
            self.proteins_graph_count = len(proteins_graphs)
        elif self.name in ['Tox21_p53', 'Tox21_HSE', 'Tox21_MMP', 'Tox21_PPAR_gamma']:
            # First, find the global max for node and edge labels across all splits
            max_node_label = -1
            max_edge_label = -1
            for split in ['training', 'evaluation', 'testing']:
                dir_name = f'{self.name}_{split}'
                path = os.path.join(self.raw_dir, dir_name)
                
                node_labels_path = os.path.join(path, f'{self.name}_{split}_node_labels.txt')
                if os.path.exists(node_labels_path):
                    node_labels = np.loadtxt(node_labels_path, dtype=np.int64)
                    if node_labels.size > 0:
                        max_node_label = max(max_node_label, np.max(node_labels))

                edge_labels_path = os.path.join(path, f'{self.name}_{split}_edge_labels.txt')
                if os.path.exists(edge_labels_path):
                    edge_labels = np.loadtxt(edge_labels_path, dtype=np.int64)
                    if edge_labels.size > 0:
                        max_edge_label = max(max_edge_label, np.max(edge_labels))

            num_node_classes = max_node_label + 1
            num_edge_classes = max_edge_label + 1

            self.split_counts = {}
            for split in ['training', 'evaluation', 'testing']:
                dir_name = f'{self.name}_{split}'
                file_prefix = f'{self.name}_{split}'
                path = os.path.join(self.raw_dir, dir_name)

                edge_index = np.loadtxt(os.path.join(path, f'{file_prefix}_A.txt'), delimiter=',', dtype=np.int64) - 1
                edge_index = torch.from_numpy(edge_index.T).to(torch.int64)

                edge_labels_path = os.path.join(path, f'{file_prefix}_edge_labels.txt')
                if os.path.exists(edge_labels_path):
                    edge_labels = np.loadtxt(edge_labels_path, dtype=np.int64)
                    edge_labels = F.one_hot(torch.from_numpy(edge_labels), num_classes=num_edge_classes).float()
                else:
                    edge_labels = torch.ones(edge_index.shape[1], 1).float() # Dummy edge features

                graph_indicator = np.loadtxt(os.path.join(path, f'{file_prefix}_graph_indicator.txt'), dtype=np.int64) - 1

                graph_labels = np.loadtxt(os.path.join(path, f'{file_prefix}_graph_labels.txt'), dtype=np.int64)
                graph_labels[graph_labels == -1] = 0
                graph_labels = torch.from_numpy(graph_labels)
                
                node_labels = np.loadtxt(os.path.join(path, f'{file_prefix}_node_labels.txt'), dtype=np.int64)
                node_labels = F.one_hot(torch.from_numpy(node_labels), num_classes=num_node_classes).float()

                num_graphs = graph_labels.size(0)
                self.split_counts[split] = num_graphs
                slices = torch.from_numpy(np.bincount(graph_indicator)).cumsum(0)
                slices = torch.cat([torch.tensor([0]), slices])

                for i in range(num_graphs):
                    start_node, end_node = slices[i], slices[i+1]
                    edge_mask = (edge_index[0] >= start_node) & (edge_index[0] < end_node)

                    sub_edge_index = edge_index[:, edge_mask] - start_node

                    data = Data(
                        x=node_labels[start_node:end_node],
                        edge_index=sub_edge_index,
                        edge_attr=edge_labels[edge_mask],
                        y=torch.tensor([[graph_labels[i].item()]], dtype=torch.long),
                        n_nodes=torch.tensor([end_node - start_node])
                    )
                    pyg_list.append(data)
        else:  # ego, protein
            with open(os.path.join(self.raw_dir, f"{self.name}_split.pkl"), "rb") as f:
                adjs = pickle.load(f)
            for split in ['train', 'val', 'test']:
                for adj in adjs[split]:
                    x = torch.ones(adj.shape[0], 1, dtype=torch.float)
                    edge_index, _ = dense_to_sparse(torch.from_numpy(adj).float())
                    y = torch.zeros(1, 0).long()
                    data = Data(x=x, edge_index=edge_index, y=y, n_nodes=torch.tensor([adj.shape[0]]))
            pyg_list.append(data)

        if self.name in ['Tox21_p53', 'Tox21_HSE', 'Tox21_MMP', 'Tox21_PPAR_gamma']:
            torch.save(self.split_counts, os.path.join(self.processed_dir, 'split_counts.pt'))
            
        data, slices = self.collate(pyg_list)
        torch.save((data, slices), self.processed_paths[0])


class SpectreGraphDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset_name = cfg.dataset.name
        self.datadir = cfg.dataset.datadir
        self.batch_size = cfg.train.batch_size
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None


    def prepare_data(self) -> None:
        # For mixed datasets like BZ_CO and EN_PR, datadir already includes dataset name
        root_path = self.datadir
        SpectreDataset(root=root_path, name=self.dataset_name)

    def setup(self, stage: str = None) -> None:
        # For mixed datasets like BZ_CO and EN_PR, datadir already includes dataset name
        root_path = self.datadir
        dataset = SpectreDataset(root=root_path, name=self.dataset_name)

        if self.dataset_name in ['Tox21_p53', 'Tox21_HSE', 'Tox21_MMP', 'Tox21_PPAR_gamma']:
            # Tox21 series: special handling consistent with main_Tox.py/data_loader.py
            # Use split_counts to split correctly, skipping the evaluation split
            training_count = dataset.split_counts['training']
            evaluation_count = dataset.split_counts['evaluation']
            testing_count = dataset.split_counts['testing']
            
            # Training data: use only label-0 samples from training split
            train_normal_graphs = []
            test_normal_graphs = []  # label-0 samples from testing split
            test_anomaly_graphs = []  # label-1 samples from testing split
            
            # Skip evaluation split; use only training and testing
            for i, data in enumerate(dataset):
                if i < training_count:
                    # training split
                    if data.y.item() == 0:
                        train_normal_graphs.append(data)
                elif i >= training_count + evaluation_count:
                    # testing split (evaluation skipped)
                    if data.y.item() == 0:
                        test_normal_graphs.append(data)
                    else:
                        test_anomaly_graphs.append(data)
            
            # Silent: dataset counts for Tox21 (train/test)
            
            # Training data: use training normals; hide labels
            train_data = []
            for data in train_normal_graphs:
                data_clone = data.clone()
                data_clone.y = torch.zeros(1, 0, dtype=torch.long)  # hide labels
                train_data.append(data_clone)
            self._train_dataset = train_data
            
            # Validation data: a small subset of training
            num_val = len(train_data) // 10
            self._val_dataset = train_data[:num_val]
            
            # Test data: all testing samples
            test_data = []
            
            # Testing normals: testing label 0 -> final label 0 (normal)
            for data in test_normal_graphs:
                data_clone = data.clone()
                data_clone.y = torch.tensor([[0]], dtype=torch.long)  # normal
                test_data.append(data_clone)
                
            # Testing anomalies: testing label 1 -> final label 1 (anomaly)
            for data in test_anomaly_graphs:
                data_clone = data.clone()
                data_clone.y = torch.tensor([[1]], dtype=torch.long)  # anomaly
                test_data.append(data_clone)
            
            self._test_dataset = test_data
            
            # Silent: distribution summary for Tox21 (ID/OOD)
        elif self.dataset_name == 'EN_PR':
            # Custom split: train 540 normals (ENZYMES); test 60 ID (ENZYMES) + 60 OOD (PROTEINS)
            # Assumes process() maps ENZYMES y=0 and PROTEINS y=1.
            rng = np.random.RandomState(42)

            enzymes = [d for d in dataset if d.y.item() == 0]
            proteins = [d for d in dataset if d.y.item() == 1]

            need_train_normals = 540
            need_test_id = 60
            need_test_ood = 60

            if len(enzymes) < (need_train_normals + need_test_id):
                raise ValueError(f"Insufficient ENZYMES samples: required {need_train_normals + need_test_id}, got {len(enzymes)}")
            if len(proteins) < need_test_ood:
                raise ValueError(f"Insufficient PROTEINS samples: required {need_test_ood}, got {len(proteins)}")

            enz_idx = np.arange(len(enzymes))
            prot_idx = np.arange(len(proteins))
            rng.shuffle(enz_idx)
            rng.shuffle(prot_idx)

            train_idx = enz_idx[:need_train_normals]
            test_id_idx = enz_idx[need_train_normals:need_train_normals + need_test_id]
            test_ood_idx = prot_idx[:need_test_ood]

            # Build training set (hide labels)
            train_data = []
            for i in train_idx:
                d = enzymes[i].clone()
                d.y = torch.zeros(1, 0, dtype=torch.long)
                train_data.append(d)
            self._train_dataset = train_data

            # Validation set: keep empty to ensure exactly 540 training samples
            self._val_dataset = []

            # Build test set: ID=0, OOD=1
            test_data = []
            for i in test_id_idx:
                d = enzymes[i].clone()
                d.y = torch.tensor([[0]], dtype=torch.long)
                test_data.append(d)
            for i in test_ood_idx:
                d = proteins[i].clone()
                d.y = torch.tensor([[1]], dtype=torch.long)
                test_data.append(d)
            self._test_dataset = test_data

            # Silent: EN_PR split summary

        elif self.dataset_name == 'BZ_CO':
            # Custom split: BZR as normal (ID), COX2 as anomaly (OOD)
            # Target: train 364 normals (BZR), test 41 ID (BZR) + 41 OOD (COX2)
            rng = np.random.RandomState(42)

            # Prefer dataset_source (0=BZR, 1=COX2); fallback to order/counts if missing
            try:
                bzr = [d for d in dataset if hasattr(d, 'dataset_source') and d.dataset_source.item() == 0]
                cox2 = [d for d in dataset if hasattr(d, 'dataset_source') and d.dataset_source.item() == 1]
            except Exception:
                bzr_count = getattr(dataset, 'bzr_graph_count', None)
                if bzr_count is None:
                    raise ValueError("BZ_CO data lacks dataset_source and bzr_graph_count is unknown; cannot perform custom split")
                data_list = list(dataset)
                bzr = data_list[:bzr_count]
                cox2 = data_list[bzr_count:]

            need_train_normals = 364
            need_test_id = 41
            need_test_ood = 41

            if len(bzr) < (need_train_normals + need_test_id):
                raise ValueError(f"Insufficient BZR samples: required {need_train_normals + need_test_id}, got {len(bzr)}")
            if len(cox2) < need_test_ood:
                raise ValueError(f"Insufficient COX2 samples: required {need_test_ood}, got {len(cox2)}")

            bzr_idx = np.arange(len(bzr))
            cox_idx = np.arange(len(cox2))
            rng.shuffle(bzr_idx)
            rng.shuffle(cox_idx)

            # 使用全部BZR：364训练 + 41测试ID = 405（典型BZR大小）
            train_idx = bzr_idx[:need_train_normals]
            test_id_idx = bzr_idx[need_train_normals:need_train_normals + need_test_id]
            test_ood_idx = cox_idx[:need_test_ood]

            # Build training set (hide labels)
            train_data = []
            for i in train_idx:
                d = bzr[i].clone()
                d.y = torch.zeros(1, 0, dtype=torch.long)
                train_data.append(d)
            self._train_dataset = train_data

            # Validation set: keep empty to ensure exactly 364 training samples
            self._val_dataset = []

            # Test set: ID=0, OOD=1
            test_data = []
            for i in test_id_idx:
                d = bzr[i].clone()
                d.y = torch.tensor([[0]], dtype=torch.long)
                test_data.append(d)
            for i in test_ood_idx:
                d = cox2[i].clone()
                d.y = torch.tensor([[1]], dtype=torch.long)
                test_data.append(d)
            self._test_dataset = test_data

            # Silent: BZ_CO split summary
            
        else:
            # Other datasets: follow data_loader.py logic
            # Use StratifiedKFold with 5 folds and random_state=0
            from sklearn.model_selection import StratifiedKFold
            
            data_list = []
            label_list = []
            
            for data in dataset:
                data_list.append(data)
                label_list.append(data.y.item())
            
            # Use the same seed and fold as data_loader.py
            kfd = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
            splits = list(kfd.split(data_list, label_list))
            train_index, test_index = splits[0]  # 使用第一折
            
            data_train_ = [data_list[i] for i in train_index]
            data_test = [data_list[i] for i in test_index]
            
            # Training data: keep normals according to dataset rules
            normal_graphs = []
            if self.dataset_name in ['ENZYMES',' PROTEINS', 'DD']:
                # Treat label 1 as anomaly; others as normal
                for data in data_train_:
                    if data.y.item() != 1:
                        normal_graphs.append(data)
            else:
                # Others: treat label 0 as anomaly; others as normal
                for data in data_train_:
                    if data.y.item() != 0:
                        normal_graphs.append(data)
            
            # Count ID/OOD in test set (per dataset rules)
            test_id_count = 0
            test_ood_count = 0
            for data in data_test:
                if self.dataset_name in ['ENZYMES',' PROTEINS', 'DD']:
                    # ENZYMES: original label 1 -> OOD, others -> ID
                    if data.y.item() == 1:
                        test_ood_count += 1
                    else:
                        test_id_count += 1
                else:
                    # Others: original label 0 -> OOD, others -> ID
                    if data.y.item() == 0:
                        test_ood_count += 1
                    else:
                        test_id_count += 1
            
            # Silent: other datasets split summary
            
            # Training data: hide labels
            train_data = []
            for idx, data in enumerate(normal_graphs):
                data_clone = data.clone()
                data_clone.y = torch.zeros(1, 0, dtype=torch.long)  # hide labels
                train_data.append(data_clone)
            self._train_dataset = train_data
            
            # Validation data: a small subset of training
            num_val = len(train_data) // 10
            self._val_dataset = train_data[:num_val]
            
            # Test data: map labels according to dataset rules
            test_data = []
            for data in data_test:
                data_clone = data.clone()
                if self.dataset_name == 'ENZYMES':
                    # ENZYMES: original 1 -> 1 (anomaly), others -> 0 (normal)
                    if data.y.item() == 1:
                        data_clone.y = torch.tensor([[1]], dtype=torch.long)
                    else:
                        data_clone.y = torch.tensor([[0]], dtype=torch.long)
                elif self.dataset_name == 'PROTEINS' or self.dataset_name == 'DD':
                    # PROTEINS/DD: original 1 -> 1 (anomaly), others -> 0 (normal)
                    if data.y.item() == 1:
                        data_clone.y = torch.tensor([[1]], dtype=torch.long)
                    else:
                        data_clone.y = torch.tensor([[0]], dtype=torch.long)
                else:
                    # Others: original 0 -> 1 (anomaly), others -> 0 (normal)
                    if data.y.item() == 0:
                        data_clone.y = torch.tensor([[1]], dtype=torch.long)
                    else:
                        data_clone.y = torch.tensor([[0]], dtype=torch.long)
                test_data.append(data_clone)
            
            self._test_dataset = test_data
        
    # silent: dataset split summary is hidden in normal runs

    def train_dataloader(self):
        return torch_geometric.loader.DataLoader(
            self._train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch_geometric.loader.DataLoader(
            self._val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return torch_geometric.loader.DataLoader(
            self._test_dataset, batch_size=self.batch_size, shuffle=False
        )
    
    def node_counts(self):
        return self._train_dataset.get_all_n_nodes()
    
    def node_types(self):
        return self._train_dataset.get_node_types()
        
    def edge_counts(self):
        return self._train_dataset.get_edge_counts()

class SpectreDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        
        # Access datasets through the datamodule
        self.train_dataset = datamodule._train_dataset
        self.val_dataset = datamodule._val_dataset
        self.test_dataset = datamodule._test_dataset

        self.dataset_name = datamodule.dataset_name
        
        self.n_nodes = self.get_n_nodes()
        self.node_types = self.get_node_types()
        self.edge_types = self.get_edge_types()
        
        super().complete_infos(self.n_nodes, self.node_types)

    def get_n_nodes(self):
        """ The distribution of the number of nodes per graph. """
        all_n_nodes = [data.n_nodes.item() for data in self.train_dataset]
        return torch.tensor(all_n_nodes)

    def get_node_types(self):
        """ The distribution of node types. """
        num_node_types = self.train_dataset[0].x.size(1)
        node_types = torch.zeros(num_node_types)
        for data in self.train_dataset:
            node_types += data.x.sum(dim=0)
        return node_types

    def get_edge_types(self):
        """ The distribution of edge types. """
        if hasattr(self.train_dataset[0], 'edge_attr') and self.train_dataset[0].edge_attr is not None:
            num_edge_types = self.train_dataset[0].edge_attr.size(1)
            edge_types = torch.zeros(num_edge_types)
            for data in self.train_dataset:
                if data.edge_attr is not None:
                    edge_types += data.edge_attr.sum(dim=0)
        else: # If no edge_attr, assume one edge type
            edge_types = torch.tensor([len(data.edge_index[0]) for data in self.train_dataset]).sum()
        return edge_types

