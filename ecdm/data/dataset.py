import os
from glob import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from functools import partial
import lightning.pytorch as pl
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from torch.utils.data import DataLoader

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from ecdm.util import instantiate_from_config

IMG_FORMATS = (
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    # "tif",
    "tiff",
    "webp",
    "pfm",
)  # include image suffixes


def exists(x):
    return x is not None


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train=None,
        validation=None,
        test=None,
        predict=None,
        wrap=False,
        num_workers=None,
        shuffle_test_loader=False,
        use_worker_init_fn=False,
        shuffle_val_dataloader=False,
        collate_fn=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = partial(
                self._train_dataloader, collate_fn=collate_fn
            )
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(
                self._val_dataloader, shuffle=shuffle_val_dataloader
            )
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(
                self._test_dataloader, shuffle=shuffle_test_loader
            )
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs
        )
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self, collate_fn=None):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            worker_init_fn=init_fn,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def _val_dataloader(self, shuffle=False):
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=None,
            shuffle=shuffle,
            pin_memory=True,
        )

    def _test_dataloader(self, shuffle=False):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=None,
            shuffle=shuffle,
            pin_memory=True,
        )

    def _predict_dataloader(self, shuffle=False):
        return DataLoader(
            self.datasets["predict"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=None,
            pin_memory=True,
        )


class MixedDataset(Dataset):
    def __init__(
        self,
        taeget_dataset_root_path,
        taeget_dataset_HF_root_path,
        extra_dataset_root_path,
        extra_dataset_HF_root_path,
        mode="train",
        image_size=(512, 640),
    ):
        super().__init__()
        assert mode in ["train", "test"]

        self.taeget_dataset_root_path = Path(taeget_dataset_root_path)
        self.taeget_dataset_HF_root_path = Path(taeget_dataset_HF_root_path)

        self.extra_dataset_root_path = Path(extra_dataset_root_path)
        self.extra_dataset_HF_root_path = Path(extra_dataset_HF_root_path)

        self.taeget_dataset_root_path = glob(
            str(Path(self.taeget_dataset_root_path) / "**" / "*.*"), recursive=True
        )
        self.taeget_dataset_HF_root_path = glob(
            str(Path(self.taeget_dataset_HF_root_path) / "**" / "*.*"), recursive=True
        )
        self.extra_dataset_root_path = glob(
            str(Path(self.extra_dataset_root_path) / "**" / "*.*"), recursive=True
        )
        self.extra_dataset_HF_root_path = glob(
            str(Path(self.extra_dataset_HF_root_path) / "**" / "*.*"), recursive=True
        )

        self.target_tir_images = sorted(
            x.replace("/", os.sep)
            for x in self.taeget_dataset_root_path
            if x.split(".")[-1].lower() in IMG_FORMATS
        )
        self.target_tir_HF_images = sorted(
            x.replace("/", os.sep)
            for x in self.taeget_dataset_HF_root_path
            if x.split(".")[-1].lower() in IMG_FORMATS
        )
        self.extra_vis_images = sorted(
            x.replace("/", os.sep)
            for x in self.extra_dataset_root_path
            if x.split(".")[-1].lower() in IMG_FORMATS
        )
        self.extra_vis_HF_images = sorted(
            x.replace("/", os.sep)
            for x in self.extra_dataset_HF_root_path
            if x.split(".")[-1].lower() in IMG_FORMATS
        )
        self.extra_vis_images = self.extra_vis_images[:30000]
        self.extra_vis_HF_images = self.extra_vis_HF_images[:30000]

        self.numbers = len(self.extra_vis_images)
        self.target_len = len(self.target_tir_images)

        self.transform = T.Compose(
            [
                T.Resize(tuple(image_size)),
            ]
        )

    def __len__(self):
        return self.numbers

    def __getitem__(self, index):
        extra_vis_img_path = self.extra_vis_images[index]
        extra_vis_HF_img_path = self.extra_vis_HF_images[index]

        target_index = int(index % self.target_len)
        target_tir_img_path = self.target_tir_images[target_index]
        target_tir_HF_img_path = self.target_tir_HF_images[target_index]

        extra_vis_img = Image.open(extra_vis_img_path)
        extra_vis_HF_img = Image.open(extra_vis_HF_img_path)
        target_tir_img = Image.open(target_tir_img_path)
        target_tir_HF_img = Image.open(target_tir_HF_img_path)

        extra_vis_img = extra_vis_img.convert("RGB")
        extra_vis_HF_img = extra_vis_HF_img.convert("RGB")
        target_tir_img = target_tir_img.convert("RGB")
        target_tir_HF_img = target_tir_HF_img.convert("RGB")

        extra_vis_img = self.transform(extra_vis_img)
        extra_vis_HF_img = self.transform(extra_vis_HF_img)
        target_tir_img = self.transform(target_tir_img)
        target_tir_HF_img = self.transform(target_tir_HF_img)

        extra_vis_img = np.array(extra_vis_img).astype(np.uint8)
        extra_vis_HF_img = np.array(extra_vis_HF_img).astype(np.uint8)
        target_tir_img = np.array(target_tir_img).astype(np.uint8)
        target_tir_HF_img = np.array(target_tir_HF_img).astype(np.uint8)

        extra_vis_img = (extra_vis_img / 127.5 - 1.0).astype(np.float32)
        extra_vis_HF_img = (extra_vis_HF_img / 127.5 - 1.0).astype(np.float32)
        target_tir_img = (target_tir_img / 127.5 - 1.0).astype(np.float32)
        target_tir_HF_img = (target_tir_HF_img / 127.5 - 1.0).astype(np.float32)

        example = {}
        example["vis_img"] = extra_vis_img
        example["vis_edge"] = extra_vis_HF_img
        example["tir_img"] = target_tir_img
        example["tir_edge"] = target_tir_HF_img
        example["img_id"] = extra_vis_img_path

        return example


class LLVIPDataset(Dataset):
    def __init__(
        self,
        root,
        mode="train",
        size=(512, 640),
        edge_type="high_filter",
    ):
        super().__init__()
        self.mode = mode
        assert self.mode in ["train", "full", "test"]
        self.path = Path(root)
        assert self.path.exists(), "The pathfile of KAIST dataset is not exist."
        self.edge_type = edge_type
        assert self.edge_type in ["high_filter", "pidinet", "teed"]

        self.dataset_root_path = Path(root)
        self.visible_all_path = self.dataset_root_path / "visible"
        self.thermal_all_path = self.dataset_root_path / "infrared"

        if self.mode == "train":
            self.visible_path = self.visible_all_path / "train"
            self.visible_HF_path = self.visible_all_path / ("train_" + self.edge_type)
            self.thermal_path = self.thermal_all_path / "train"
            self.thermal_HF_path = self.thermal_all_path / ("train_" + self.edge_type)
        else:
            self.visible_path = self.visible_all_path / "test"
            self.visible_HF_path = self.visible_all_path / ("test_" + self.edge_type)
            self.thermal_path = self.thermal_all_path / "test"
            self.thermal_HF_path = self.thermal_all_path / ("test_" + self.edge_type)

        self.visible_root_path = glob(
            str(Path(self.visible_path) / "**" / "*.*"), recursive=True
        )
        self.visible_images = sorted(
            x.replace("/", os.sep)
            for x in self.visible_root_path
            if x.split(".")[-1].lower() in IMG_FORMATS
        )

        self.visible_HF_root_path = glob(
            str(Path(self.visible_HF_path) / "**" / "*.*"), recursive=True
        )
        self.visible_HF_images = sorted(
            x.replace("/", os.sep)
            for x in self.visible_HF_root_path
            if x.split(".")[-1].lower() in IMG_FORMATS
        )

        self.thermal_root_path = glob(
            str(Path(self.thermal_path) / "**" / "*.*"), recursive=True
        )
        self.thermal_images = sorted(
            x.replace("/", os.sep)
            for x in self.thermal_root_path
            if x.split(".")[-1].lower() in IMG_FORMATS
        )

        self.thermal_HF_root_path = glob(
            str(Path(self.thermal_HF_path) / "**" / "*.*"), recursive=True
        )
        self.thermal_HF_images = sorted(
            x.replace("/", os.sep)
            for x in self.thermal_HF_root_path
            if x.split(".")[-1].lower() in IMG_FORMATS
        )

        self.numbers = len(self.visible_images)

        self.transform = T.Compose(
            [
                T.Resize(tuple(size)),
            ]
        )

    def __len__(self):
        return self.numbers

    def __getitem__(self, index):
        visible_path = self.visible_images[index]
        visible_HF_path = self.visible_HF_images[index]
        thermal_path = self.thermal_images[index]
        thermal_HF_path = self.thermal_HF_images[index]

        vis_img = Image.open(visible_path)
        vis_HF_img = Image.open(visible_HF_path)
        tir_img = Image.open(thermal_path)
        tir_HF_img = Image.open(thermal_HF_path)

        vis_img = vis_img.convert("L")
        vis_HF_img = vis_HF_img.convert("L")
        tir_img = tir_img.convert("L")
        tir_HF_img = tir_HF_img.convert("L")

        vis_img = self.transform(vis_img)
        vis_HF_img = self.transform(vis_HF_img)
        tir_img = self.transform(tir_img)
        tir_HF_img = self.transform(tir_HF_img)

        vis_img = np.array(vis_img).astype(np.uint8)
        vis_HF_img = np.array(vis_HF_img).astype(np.uint8)
        tir_img = np.array(tir_img).astype(np.uint8)
        tir_HF_img = np.array(tir_HF_img).astype(np.uint8)

        vis_img = (vis_img / 127.5 - 1.0).astype(np.float32)
        vis_HF_img = (vis_HF_img / 127.5 - 1.0).astype(np.float32)
        tir_img = (tir_img / 127.5 - 1.0).astype(np.float32)
        tir_HF_img = (tir_HF_img / 127.5 - 1.0).astype(np.float32)

        example = {}
        example["vis_img"] = vis_img
        example["vis_edge"] = vis_HF_img
        example["tir_img"] = tir_img
        example["tir_edge"] = tir_HF_img
        example["img_id"] = visible_path

        return example
