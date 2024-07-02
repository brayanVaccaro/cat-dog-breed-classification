import os
from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


class DataLoaderHelper:
    def __init__(self, root_dir="./data"):
        self.root_dir = root_dir
        self.annotations_file = os.path.join(
            f"{root_dir}\\oxford-iiit-pet", "annotations", "trainval.txt"
        )
        self.data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        }

        self.dataset = self.create_oxford_train()
        self.classes_to_idx = self.dataset.class_to_idx
        self.index_to_class = {v: k for k, v in self.classes_to_idx.items()}

    def create_oxford_train(self):
        dataset = OxfordIIITPet(
            root=self.root_dir, transform=self.data_transforms["train"], download=True
        )
        return dataset

    def load_data(self, selected_classes=None):
        dataset = OxfordIIITPet(
            root=self.root_dir, transform=self.data_transforms["train"], download=True
        )

        if selected_classes:
            return self.filter_and_split_dataset(selected_classes)
        else:
            return self.split_dataset()

    def filter_and_split_dataset(self, selected_classes):
        class_to_idx = self.dataset.class_to_idx
        selected_indices = [
            i
            for i, (_, label) in enumerate(self.dataset)
            if self.dataset.classes[label] in selected_classes
        ]

        selected_targets = [self.dataset[i][1] for i in selected_indices]

        # Split the dataset into training and validation sets
        train_indices, val_indices = train_test_split(
            selected_indices, test_size=0.2, stratify=selected_targets, random_state=42
        )

        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)

        # Apply the respective transformations
        train_dataset.dataset.transform = self.data_transforms["train"]
        val_dataset.dataset.transform = self.data_transforms["val"]

        dataloaders = {
            "train": DataLoader(
                train_dataset, batch_size=40, shuffle=True, num_workers=4
            ),
            "val": DataLoader(val_dataset, batch_size=5, shuffle=False, num_workers=4),
        }
        dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

        filtered_class_to_idx = {
            key: self.dataset.class_to_idx[key]
            for key in selected_classes
            if key in self.dataset.class_to_idx
        }
        filtered_classes = [
            cls for cls in self.dataset.class_to_idx.keys() if cls in selected_classes
        ]

        for x in ["train", "val"]:
            dataloaders[x].dataset.classes = filtered_classes
            dataloaders[x].dataset.class_to_idx = filtered_class_to_idx

        return dataloaders, dataset_sizes

    def split_dataset(self):
        targets = [label for _, label in self.dataset]

        # Split the dataset into training and validation sets
        train_indices, val_indices = train_test_split(
            range(len(self.dataset)), test_size=0.25, stratify=targets, random_state=42
        )

        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)

        # Apply the respective transformations
        train_dataset.dataset.transform = self.data_transforms["train"]
        val_dataset.dataset.transform = self.data_transforms["val"]

        dataloaders = {
            "train": DataLoader(
                train_dataset, batch_size=46, shuffle=True, num_workers=4
            ),
            "val": DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=4),
        }
        dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

        return dataloaders, dataset_sizes

    def load_test_data(self, selected_classes):
        dataset = OxfordIIITPet(
            root=self.root_dir,
            split="test",
            transform=self.data_transforms["val"],
            download=True,
        )

        if selected_classes:
            class_to_idx = dataset.class_to_idx
            selected_indices = [
                i
                for i, (_, label) in enumerate(dataset)
                if dataset.classes[label] in selected_classes
            ]

            test_dataset = Subset(dataset, selected_indices)
        else:
            test_dataset = dataset

        dataloader = DataLoader(
            test_dataset, batch_size=40, shuffle=False, num_workers=4
        )
        return dataloader, len(test_dataset)

    def matplotlib_imshow(self, img: Tensor, one_channel):
        if one_channel:
            img = img.mean(
                dim=0
            )  # Converte l'immagine a scala di grigi facendo la media sui canali
        else:
            # Denormalizza l'immagine
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (
                img.cpu() * std[:, None, None] + mean[:, None, None]
            )  # Applica la denormalizzazione su ciascun canale

        npimg = img.numpy()
        if one_channel:
            plt.imshow(npimg, cmap="Greys")
        else:
            npimg = np.clip(
                npimg, 0, 1
            )  # Clipping dei valori per essere nell'intervallo corretto
            plt.imshow(
                np.transpose(npimg, (1, 2, 0))
            )  # Trasposizione per convertire l'immagine da (C, H, W) a (H, W, C)

    def breeds(self):
        cat_data = []
        dog_data = []
        with open(self.annotations_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[1]) - 1  # CLASS-ID - 1
                species = int(parts[2])
                if species == 1:  # Gatto
                    cat_data.append(class_id)
                elif species == 2:  # Cane
                    dog_data.append(class_id)

        # Creazione delle liste di razze di gatti e cani con indici corrispondenti
        cat_breeds = {
            breed: idx for breed, idx in self.classes_to_idx.items() if idx in cat_data
        }
        dog_breeds = {
            breed: idx for breed, idx in self.classes_to_idx.items() if idx in dog_data
        }

        return dog_breeds, cat_breeds
