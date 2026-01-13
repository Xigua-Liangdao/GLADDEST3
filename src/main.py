import os
import logging
import pathlib
import warnings

# Early suppression for import-time warnings from C/C++ extensions before heavy imports
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Early suppression for import-time runtime warnings from extensions
warnings.filterwarnings("ignore", category=RuntimeWarning)

import graph_tool
import torch

torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from graph_discrete_flow_model import GraphDiscreteFlowModel
from models.extra_features import DummyExtraFeatures, ExtraFeatures
from analysis.visualization import NonMolecularVisualization

# Suppress Lightning's verbose user warnings globally
warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=r"You are using a CUDA device.*Tensor Cores")
warnings.filterwarnings("ignore", message=r"You are using `torch.load` with `weights_only=False`.*")

# Reduce Lightning/Fabric info logging
for name in [
    "pytorch_lightning",
    "lightning",
    "lightning.pytorch",
    "lightning_fabric",
]:
    logging.getLogger(name).setLevel(logging.ERROR)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Convert datadir to absolute path
    if not os.path.isabs(cfg.dataset.datadir):
        cfg.dataset.datadir = os.path.join(get_original_cwd(), cfg.dataset.datadir)

    pl.seed_everything(cfg.train.seed)
    dataset_config = cfg["dataset"]

    if dataset_config["name"] in [
        "bzr",
        "Tox21_p53",
        "Tox21_HSE",
        "Tox21_MMP",
        "Tox21_PPAR_gamma",
        "AIDS",
        "ENZYMES",
        "EN_PR",
        "cox2", 
        "NCI1",
    ]:
        from datasets.spectre_dataset import (
            SpectreGraphDataModule,
            SpectreDatasetInfos,
        )
        from analysis.spectre_utils import (
            SamplingMetrics,
        )

        datamodule = SpectreGraphDataModule(cfg)  # get data module, then use it to build metrics
        datamodule.prepare_data()
        datamodule.setup()
        if dataset_config["name"] in ["bzr", "Tox21_p53", "Tox21_HSE","Tox21_MMP","Tox21_PPAR_gamma","AIDS","ENZYMES","EN_PR","cox2","NCI1"]:
            sampling_metrics = SamplingMetrics(datamodule)
        else:
            raise NotImplementedError(
                f"Dataset {dataset_config['name']} not implemented"
            )
    
        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
    
        train_metrics = TrainAbstractMetricsDiscrete()
        visualization_tools = NonMolecularVisualization(dataset_name=cfg.dataset.name)
       
        extra_features = ExtraFeatures(
            cfg.model.extra_features,
            cfg.model.rrwp_steps,
            dataset_info=dataset_infos,
        )
        domain_features = DummyExtraFeatures()
    
        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )
    model_kwargs = {
        "dataset_infos": dataset_infos,
        "train_metrics": train_metrics,
        "sampling_metrics": sampling_metrics,
        "visualization_tools": visualization_tools,
        "extra_features": extra_features,
        "domain_features": domain_features,
        "test_labels": (
            datamodule.test_labels
            if ("qm9" in cfg.dataset.name and cfg.general.conditional)
            else None
        ),
    }

    utils.create_folders(cfg)
    model = GraphDiscreteFlowModel(cfg=cfg, **model_kwargs)

    callbacks = []
    if cfg.train.save_model:
        # Compute a robust saving schedule and print it for visibility
        try:
            every_n_epochs = int(cfg.general.sample_every_val) * int(cfg.general.check_val_every_n_epochs)
        except Exception:
            every_n_epochs = 1
        if every_n_epochs <= 0:
            every_n_epochs = 1

        # Optional step-based checkpointing to ensure artifacts even if killed early
        save_every_n_train_steps = getattr(cfg.general, "save_every_n_train_steps", 0) or 0
        if save_every_n_train_steps < 0:
            save_every_n_train_steps = 0

        # silent: avoid checkpoint schedule logs in normal runs

        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename="{epoch}",
            save_top_k=-1,
            save_last=True,
            every_n_epochs=every_n_epochs,
            every_n_train_steps=save_every_n_train_steps if save_every_n_train_steps else None,
        )
        callbacks.append(checkpoint_callback)

        # Failsafe: also save on every train epoch end and on train end,
        # so we get checkpoints even if there is no validation loop.
        class SaveAlwaysCallback(Callback):
            def __init__(self, base_dir: str):
                self.base_dir = base_dir
                os.makedirs(self.base_dir, exist_ok=True)

            def on_train_epoch_end(self, trainer, pl_module):
                epoch = int(trainer.current_epoch)
                path = os.path.join(self.base_dir, f"epoch={epoch}.ckpt")
                try:
                    trainer.save_checkpoint(path)
                    # silent: avoid repetitive checkpoint logs
                except Exception as e:
                    pass

            def on_train_end(self, trainer, pl_module):
                path = os.path.join(self.base_dir, "last.ckpt")
                try:
                    trainer.save_checkpoint(path)
                    # silent
                except Exception as e:
                    pass

        callbacks.append(SaveAlwaysCallback(base_dir=f"checkpoints/{cfg.general.name}"))

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == "debug":
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy="ddp_find_unused_parameters_true",  # Needed to  old checkpoints
        accelerator="gpu" if use_gpu else "cpu",
        # Use the number of GPUs requested instead of a hardcoded index; PL will pick the first N available.
        devices=(int(cfg.general.gpus) if use_gpu else None),
        max_epochs=cfg.train.n_epochs,
        max_steps=cfg.train.max_steps if getattr(cfg.train, "max_steps", None) else None,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=name == "debug",
        enable_progress_bar=False,
        callbacks=callbacks,
        log_every_n_steps=50 if name != "debug" else 1,
        accumulate_grad_batches=getattr(cfg.train, "accumulate_grad_batches", 1),
        limit_train_batches=getattr(cfg.train, "limit_train_batches", None),
        limit_val_batches=getattr(cfg.train, "limit_val_batches", None),
        limit_test_batches=getattr(cfg.train, "limit_test_batches", None),
        logger=[],
    )
    # silent: remove one-off batch shape preview

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
    else:
        # Start by evaluating test_only_path
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            # silent
            files_list = os.listdir(directory)
            for file in files_list:
                if ".ckpt" in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    # silent
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
