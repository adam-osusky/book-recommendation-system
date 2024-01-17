import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.data_loaders import train_val_test_split
from data.eval import BookEvaluator, Recommender
from util.job import ConfigurableJob
from util.logger import get_logger


class CorrBaseline(Recommender):
    def __init__(self, df: pd.DataFrame, implicit_value: int = 10) -> None:
        df = df[["book_id", "user_id", "Book-Rating"]].copy()
        df.loc[df["Book-Rating"] == 0, "Book-Rating"] = implicit_value
        self.df = df
        self.interactions = df.pivot_table(
            index="user_id", columns="book_id", values="Book-Rating", fill_value=0
        )

    def recommend(self, user_id: int, books, k=10) -> np.ndarray:
        user_top = self.df[self.df["user_id"] == user_id]["book_id"]

        pool = self.interactions
        if books is not None:
            pool = self.interactions.loc[:, books]

        column_mean = self.interactions[user_top].mean(axis=1)

        recommendations = (
            pool.corrwith(column_mean).sort_values(ascending=False).index[:k]
        )

        recommendations = np.array(recommendations)
        return recommendations

    # def get_user_top(self, user_id: int, k: int = 10):
    #     return list(
    #         self.df[self.df["user_id"] == user_id]
    #         .sort_values(by="Book-Rating", ascending=False)["book_id"]
    #         # .head(k)["book_id"]
    #     )


@dataclass
class CorrBaselineJob(ConfigurableJob):
    impl_fill_value: int = 10
    book_ds: str = "./data/book-feedback.csv"
    seed: int = 69
    test_size: float = 0.05
    val_size: float = 0.05

    k: int = 10
    pool_size: int = 100

    def run(self) -> None:
        logger = get_logger()
        logger.info(
            f"Started Correlation baseline job with this args:\n{self.get_config()}"
        )
        log_dir = os.path.join("./logs/cb", self.job_name)
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
        recommender = CorrBaseline(df=train, implicit_value=self.impl_fill_value)

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
