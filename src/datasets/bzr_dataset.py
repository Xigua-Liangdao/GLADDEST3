from torch_geometric.datasets import TUDataset
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT

import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data, InMemoryDataset, download_url
import pandas as pd

from src import utils
from src.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges, compute_molecular_metrics
from src.datasets.abstract_dataset import AbstractDatasetInfos, MolecularDataModule
class BZRDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir   # e.g., 'data/BZR'
        dataset = TUDataset(root=self.datadir, name='BZR')
        n = len(dataset)
        # Simple split; can be parameterized if needed
        datasets = {
            "train": dataset[:int(n*0.8)],
            "val": dataset[int(n*0.8):int(n*0.9)],
            "test": dataset[int(n*0.9):],
        }
        # Assume datasets is a dict with train/val/test lists of PyG Data objects
        for split in ["train", "val", "test"]:
            for data in datasets[split]:
                if not hasattr(data, "edge_attr") or data.edge_attr is None:
                    num_edges = data.edge_index.size(1)
                    data.edge_attr = torch.zeros(num_edges, 1)
                elif data.edge_attr.dim() == 1:
                    data.edge_attr = data.edge_attr.unsqueeze(1)

        super().__init__(cfg, datasets)


import torch
import numpy as np
import os

class BZRInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False, meta=None):
        self.name = "bzr"
        self.input_dims = None
        self.output_dims = None
        self.remove_h = False   # not used here
        self.compute_fcd = False   # not needed for BZR

        # Atom types for BZR (may be incomplete; adjust as needed)
        self.atom_decoder = ["C", "N", "O", "F", "Cl", "Br", "H"]
        self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
        # Common atomic weights (customizable)
        self.atom_weights = {0: 12, 1: 14, 2: 16, 3: 19, 4: 35.4, 5: 79.9, 6: 1}
        self.valencies = [4, 3, 2, 1, 1, 1, 1]  # may not be used
        self.num_atom_types = len(self.atom_decoder)
        self.max_weight = 350   # not used

        # meta file names
        meta_files = dict(
            n_nodes=f"{self.name}_n_counts.txt",
            node_types=f"{self.name}_atom_types.txt",
            edge_types=f"{self.name}_edge_types.txt",
            valency_distribution=f"{self.name}_valencies.txt",
        )

        # default None
        if meta is None:
            meta = dict(
                n_nodes=None,
                node_types=None,
                edge_types=None,
                valency_distribution=None,
            )
        # Load from files if present
        assert set(meta.keys()) == set(meta_files.keys())
        for k, v in meta_files.items():
            if (k not in meta or meta[k] is None) and os.path.exists(v):
                meta[k] = np.loadtxt(v)
                setattr(self, k, meta[k])

        # Compute (or recompute) distributions
        if recompute_statistics or not hasattr(self, "n_nodes") or self.n_nodes is None:
            self.n_nodes = datamodule.node_counts()
            # silent: avoid printing distributions in normal runs
            np.savetxt(meta_files["n_nodes"], self.n_nodes.numpy())
            self.max_n_nodes = len(self.n_nodes) - 1

        if recompute_statistics or not hasattr(self, "node_types") or self.node_types is None:
            self.node_types = datamodule.node_types()
            # silent: avoid printing distributions in normal runs
            np.savetxt(meta_files["node_types"], self.node_types.numpy())

        if recompute_statistics or not hasattr(self, "edge_types") or self.edge_types is None:
            self.edge_types = datamodule.edge_counts()
            # silent: avoid printing distributions in normal runs
            np.savetxt(meta_files["edge_types"], self.edge_types.numpy())

        # If datamodule has valency_count, compute valency distribution
        if hasattr(datamodule, "valency_count"):
            if recompute_statistics or not hasattr(self, "valency_distribution") or self.valency_distribution is None:
                valencies = datamodule.valency_count(self.max_n_nodes)
                # silent: avoid printing distributions in normal runs
                np.savetxt(meta_files["valency_distribution"], valencies.numpy())
                self.valency_distribution = valencies
        
        # Complete attributes for compatibility with the main project interface
        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

def get_smiles(cfg, datamodule, dataset_infos):
    """
    For TU datasets like BZR, there are no SMILES annotations.
    For compatibility with main pipeline, just return empty lists.
    """
    return {
        "train": [],
        "val": [],
        "test": [],
    }

