import os
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import AdamW
from torchmetrics.classification import BinaryAccuracy

from data.data_loaders import get_books_data
from data.eval import BookEvaluator
from util.job import ConfigurableJob
from util.logger import get_logger


class BPRLoss(nn.Module):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(BPRLoss, self).__init__()

    def forward(self, positive, negative):
        distances = positive - negative
        loss = -torch.sum(torch.log(torch.sigmoid(distances)), dim=0, keepdim=True)
        return loss


class HingeLossRec(nn.Module):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(HingeLossRec, self).__init__()

    def forward(self, positive, negative, margin=0.5):
        distances = positive - negative
        loss = torch.sum(torch.maximum(-distances + margin, torch.tensor(0.0)))
        return loss


class MF(LightningModule):
    def __init__(
        self,
        latent_dim: int,
        num_users: int,
        num_items: int,
        loss: Literal["BCE", "BPR"] = "BCE",
        lr: float = 1e-3,
        wd: float = 1e-5,
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
        self.loss = loss
        self.wd = wd
        self.dropout = nn.Dropout()
        self.save_hyperparameters()

    def forward(self, sample: dict[str, torch.Tensor]) -> torch.Tensor:
        U = self.U(sample["user_id"])
        b_u = self.user_bias(sample["user_id"])

        I_pos = self.I(sample["pos_book_id"])
        b_i_pos = self.item_bias(sample["pos_book_id"])

        I_neg = self.I(sample["neg_book_id"])
        b_i_neg = self.item_bias(sample["neg_book_id"])

        U = self.dropout(U)
        I_pos = self.dropout(I_pos)
        I_neg = self.dropout(I_neg)

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
        if self.loss == "BCE":
            loss = nn.BCELoss()(pred, target)
        else:
            loss = BPRLoss()(pred[:batch_size], pred[batch_size:])

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

    def configure_optimizers(self) -> AdamW:
        return AdamW(params=self.parameters(), lr=self.lr, weight_decay=self.wd)

    def inference(self, sample):
        U = self.U(sample["user_id"])
        b_u = self.user_bias(sample["user_id"])

        It = self.I(sample["book_id"])
        b_i = self.item_bias(sample["book_id"])

        U = self.dropout(U)
        It = self.dropout(It)

        pred = torch.sum(U * It, dim=1) + torch.squeeze(b_u) + torch.squeeze(b_i)
        pred = torch.sigmoid(pred)

        return pred

    def recommend(self, user_id: int, books, k: int = 10):
        sample = {
            "user_id": torch.tensor([user_id], dtype=torch.long),
            "book_id": torch.tensor(books, dtype=torch.long),
        }

        preds = self.inference(sample=sample)
        _, top_indices = torch.topk(preds, k=k, dim=0)

        return books[top_indices]


class DeepMF(LightningModule):
    def __init__(
        self,
        latent_dim: int,
        num_users: int,
        num_items: int,
        loss: Literal["BCE", "BPR", "Hinge"] = "BCE",
        lr: float = 1e-3,
        wd: float = 1e-5,
        hidden_layers: list[int] = field(default_factory=list),
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.U = nn.Embedding(num_embeddings=num_users, embedding_dim=latent_dim)
        self.I = nn.Embedding(num_embeddings=num_items, embedding_dim=latent_dim)
        self.mlp_U = nn.Embedding(num_embeddings=num_users, embedding_dim=latent_dim)
        self.mlp_I = nn.Embedding(num_embeddings=num_items, embedding_dim=latent_dim)
        mlp = []
        hidden_layers = [2 * latent_dim] + hidden_layers
        for i in range(1, len(hidden_layers)):
            mlp.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout())
        self.mlp = nn.Sequential(*mlp)
        self.prediction_layer = nn.Linear(hidden_layers[-1] + latent_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.lr = lr
        self.accuracy = BinaryAccuracy()
        self.loss = loss
        self.wd = wd
        self.dropout = nn.Dropout()

        self.save_hyperparameters()

    def forward(self, sample: dict[str, torch.Tensor]) -> torch.Tensor:
        U = self.U(sample["user_id"])
        mlp_U = self.mlp_U(sample["user_id"])
        U = self.dropout(U)
        mlp_U = self.dropout(mlp_U)

        preds = []
        for prefix in ["pos", "neg"]:
            I = self.I(sample[f"{prefix}_book_id"])
            I = self.dropout(I)
            gmf = U * I
            mlp_I = self.mlp_I(sample[f"{prefix}_book_id"])
            mlp_I = self.dropout(mlp_I)
            mlp = self.mlp(torch.cat([mlp_U, mlp_I], dim=1))
            con_res = torch.cat([gmf, mlp], dim=1)
            pred = self.prediction_layer(con_res)
            preds.append(self.sigmoid(pred))

        pred = torch.cat([preds[0], preds[1]], dim=0)

        return pred

    def inference(self, sample):
        U = self.U(sample["user_id"])
        U = self.dropout(U)
        mlp_U = self.mlp_U(sample["user_id"])
        mlp_U = self.dropout(mlp_U)

        I = self.I(sample["book_id"])
        I = self.dropout(I)
        gmf = U * I
        mlp_I = self.mlp_I(sample["book_id"])
        mlp_I = self.dropout(mlp_I)
        mlp = self.mlp(torch.cat([mlp_U.repeat(mlp_I.shape[0], 1), mlp_I], dim=1))
        con_res = torch.cat([gmf, mlp], dim=1)
        pred = self.prediction_layer(con_res)
        pred = self.sigmoid(pred)

        return pred

    def recommend(self, user_id: int, books, k: int = 10):
        sample = {
            "user_id": torch.tensor([user_id], dtype=torch.long),
            "book_id": torch.tensor(books, dtype=torch.long),
        }

        preds = self.inference(sample=sample)
        _, top_indices = torch.topk(preds, k=k, dim=0)

        return books[top_indices].squeeze()

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        batch_size = pred.shape[0] // 2
        target = torch.cat(
            [torch.ones_like(pred[:batch_size]), torch.zeros_like(pred[batch_size:])]
        )
        if self.loss == "BCE":
            loss = nn.BCELoss()(pred, target)
        elif self.loss == "BPR":
            loss = BPRLoss()(pred[:batch_size], pred[batch_size:])
        else:
            loss = HingeLossRec()(pred[:batch_size], pred[batch_size:])

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

    def configure_optimizers(self) -> AdamW:
        return AdamW(params=self.parameters(), lr=self.lr, weight_decay=self.wd)


@dataclass
class MFJob(ConfigurableJob):
    model: Literal["MF", "DeepMF"] = "MF"
    hidden_layers: list[int] = field(default_factory=list)
    lr: float = 1e-3
    batch_size: int = 32
    latent_dim: int = 16
    num_workers: int = 0
    max_epochs: int = 1
    use_feats: bool = False
    seed: int = 69
    test_size: float = 0.05
    val_size: float = 0.05
    loss: Literal["BCE", "BPR", "Hinge"] = "BCE"
    wd: float = 1e-5

    k: int = 10
    pool_size: int = 100

    def run(self) -> None:
        logger = get_logger()
        logger.info(
            f"Started Matrix Factorization job with this args:\n{self.get_config()}"
        )
        log_dir = os.path.join("./logs/mf", self.job_name)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "config.txt"), "w") as f:
            print(self.get_config(), file=f)

        seed_everything(seed=self.seed)

        logger.info("Loading the dataset and dataloaders.")
        data = get_books_data(
            batch_size=self.batch_size,
            use_feats=self.use_feats,
            num_workers=self.num_workers,
            seed=self.seed,
            test_size=self.test_size,
            val_size=self.val_size,
        )
        train_dl, val_dl, _ = data.dls

        logger.info("Initializing the model and trainer.")
        models = {
            "MF": (MF, {}),
            "DeepMF": (DeepMF, {"hidden_layers": self.hidden_layers}),
        }
        model = models[self.model][0](
            latent_dim=self.latent_dim,
            num_users=data.num_users,
            num_items=data.num_books,
            lr=self.lr,
            loss=self.loss,
            wd=self.wd,
            **models[self.model][1],
        )
        print(model)
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

        logger.info(
            f"Started evaluation at top {self.k} with pool size = {self.pool_size}"
        )
        model.eval()
        evaluator = BookEvaluator(df=data.df_val, seed=self.seed)
        evaluation = evaluator.evaluate(model=model, k=self.k, pool_size=self.pool_size)
        logger.info(f"The evaluation ended: {evaluation}")

        with open(os.path.join(log_dir, "eval"), "w") as f:
            print(evaluation, file=f)

        logger.info("Job finished succesfully.")
