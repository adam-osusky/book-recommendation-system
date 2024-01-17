import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from data.data_loaders import train_val_test_split
from data.eval import BookEvaluator, Recommender
from util.job import ConfigurableJob
from util.logger import get_logger


class SVDRec(Recommender):
    def __init__(
        self, df: pd.DataFrame, latent_dim: int, implicit_value: int = 10
    ) -> None:
        df = df[["book_id", "user_id", "Book-Rating"]].copy()
        df.loc[df["Book-Rating"] == 0, "Book-Rating"] = implicit_value
        self.df = df
        self.interactions = df.pivot_table(
            index="user_id", columns="book_id", values="Book-Rating", fill_value=0
        )

        interaction_matrix = csr_matrix(self.interactions)
        U, sigma, V_T = svds(interaction_matrix, k=latent_dim)
        predictions = U @ np.diag(sigma) @ V_T
        sorted_indices = np.argsort(predictions, axis=1)
        self.sorted_preds = sorted_indices[:, ::-1]

    def recommend(self, user_id: int, books, k=10) -> np.ndarray:
        recommends = self.sorted_preds[user_id]
        pool_recommends = recommends[np.isin(recommends, books)]

        return pool_recommends[:k]


@dataclass
class SVDJob(ConfigurableJob):
    impl_fill_value: int = 10
    book_ds: str = "./data/book-feedback.csv"
    seed: int = 69
    test_size: float = 0.05
    val_size: float = 0.05
    latent_dim: int = 10

    k: int = 10
    pool_size: int = 100

    def run(self) -> None:
        logger = get_logger()
        logger.info(f"Started SVD job with this args:\n{self.get_config()}")
        log_dir = os.path.join("./logs/svd", self.job_name)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "config.txt"), "w") as f:
            print(self.get_config(), file=f)

        logger.info("Loading the dataset.")
        df = pd.read_csv(self.book_ds)
        train, val, test = train_val_test_split(
            df, test_size=self.test_size, val_size=self.val_size
        )

        if self.pool_size < 1:
            self.pool_size = len(val)

        logger.info("Fitting the model.")
        recommender = SVDRec(
            df=train, latent_dim=self.latent_dim, implicit_value=self.impl_fill_value
        )

        logger.info(
            f"Started evaluation at top {self.k} with pool size = {self.pool_size}"
        )
        evaluator = BookEvaluator(df=val, seed=self.seed)
        evaluation = evaluator.evaluate(
            model=recommender, k=self.k, pool_size=self.pool_size
        )

        logger.info(f"The evaluation ended: {evaluation}")

        with open(os.path.join(log_dir, "eval"), "w") as f:
            print(evaluation, file=f)

        logger.info("Job finished succesfully.")
