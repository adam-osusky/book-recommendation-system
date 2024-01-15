from dataclasses import dataclass
import os

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import BinaryAccuracy

from data.data_loaders import get_books_data
from util.job import ConfigurableJob
from util.logger import get_logger


class MF(LightningModule):
    def __init__(
        self,
        latent_dim: int,
        num_users: int,
        num_items: int,
        lr: float = 1e-3,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.U = nn.Embedding(num_embeddings=num_users, embedding_dim=latent_dim)
        self.I = nn.Embedding(num_embeddings=num_items, embedding_dim=latent_dim)
        self.user_bias = nn.Embedding(num_embeddings=num_users, embedding_dim=1)
        self.item_bias = nn.Embedding(num_embeddings=num_items, embedding_dim=1)
        self.lr = lr
        self.accuracy = BinaryAccuracy()

    def forward(self, sample: dict[str, torch.Tensor]) -> torch.Tensor:
        U = self.U(sample["user_id"])
        b_u = self.user_bias(sample["user_id"])

        I_pos = self.I(sample["pos_book_id"])
        b_i_pos = self.item_bias(sample["pos_book_id"])

        I_neg = self.I(sample["neg_book_id"])
        b_i_neg = self.item_bias(sample["neg_book_id"])

        pos = torch.sum(U * I_pos, dim=1) + torch.squeeze(b_u) + torch.squeeze(b_i_pos)
        pos = torch.sigmoid(pos)

        neg = torch.sum(U * I_neg, dim=1) + torch.squeeze(b_u) + torch.squeeze(b_i_neg)
        neg = torch.sigmoid(neg)

        pred = torch.cat([pos, neg])

        return pred

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        batch_size = pred.shape[0] // 2
        target = torch.cat(
            [torch.ones_like(pred[:batch_size]), torch.zeros_like(pred[batch_size:])]
        )
        loss = nn.BCELoss()(pred, target)

        self.log("train_loss", loss, prog_bar=True)

        acc = self.accuracy(pred, target)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        batch_size = pred.shape[0] // 2
        target = torch.cat(
            [torch.ones_like(pred[:batch_size]), torch.zeros_like(pred[batch_size:])]
        )
        val_loss = nn.BCELoss()(pred, target)
        self.log("val_loss", val_loss, prog_bar=True)

        acc = self.accuracy(pred, target)
        self.log("val_acc", acc, prog_bar=True)

        return val_loss

    def configure_optimizers(self) -> Adam:
        return Adam(params=self.parameters(), lr=self.lr)


@dataclass
class MFJob(ConfigurableJob):
    lr: float = 1e-3
    batch_size: int = 32
    latent_dim: int = 16
    num_workers: int = 0
    max_epochs: int = 1
    use_feats: bool = False
    seed: int = 69

    def run(self) -> None:
        logger = get_logger()
        logger.info(
            f"Started Matrix Factorization job with this args:\n{self.get_config()}"
        )
        log_dir = os.path.join("./logs/mf", self.job_name)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "config.yaml"), "w") as f:
            print(self.get_config(), file=f)

        seed_everything(seed=self.seed)

        logger.info("Loading the dataset and dataloaders.")
        data = get_books_data(
            batch_size=self.batch_size,
            use_feats=self.use_feats,
            num_workers=self.num_workers,
            seed=self.seed,
        )
        train_dl, val_dl, _ = data.dls

        logger.info("Initializing the model and trainer.")
        model = MF(
            latent_dim=self.latent_dim,
            num_users=data.num_users,
            num_items=data.num_books,
            lr=self.lr,
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(log_dir, "checkpoints"),
            filename="best_model",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        )
        trainer = Trainer(
            max_epochs=self.max_epochs,
            default_root_dir=log_dir,
            accelerator="gpu",
            devices=1,
            val_check_interval=0.1,
            callbacks=[checkpoint_callback],
            gradient_clip_val=0.1,
            # precision="16-mixed",
            # log_every_n_steps=100,
        )

        logger.info("Training the model.")
        trainer.fit(
            model=model,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl,
        )
