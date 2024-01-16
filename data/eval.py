import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class Recommender(ABC):
    @abstractmethod
    def recommend(self, user_id: int, books, k: int = 10) -> np.ndarray:
        pass


class BookEvaluator:
    def __init__(self, df: pd.DataFrame, seed: int = 69) -> None:
        df = df.head(1000)
        self.user_interactions = df.set_index("user_id")
        self.df = df
        self.all = set(df["book_id"])
        self.seed = seed

    def get_users_interacted(self, user_id: int) -> set[int]:
        books = self.user_interactions.loc[user_id]["book_id"]

        if isinstance(books, pd.Series):
            return set(books.values)
        else:
            return set([books])

    def get_users_not_interacted(self, user_id: int, pool_size: int = 0):
        interacted = self.get_users_interacted(user_id)
        non_interacted = self.all - set(interacted)

        if pool_size > 0:
            non_interacted = random.sample(sorted(non_interacted), k=pool_size)

        return non_interacted

    def precision_k(self, relevant: set[int], preds: set[int], k: int) -> float:
        num_relevant_preds = len(relevant & set(preds))
        precision_k = num_relevant_preds / k
        return precision_k

    def recall_k(self, relevant: set[int], preds: set[int], k: int) -> float:
        num_relevant_preds = len(relevant & set(preds))
        precision_k = num_relevant_preds / len(relevant)
        return precision_k

    def evaluate_user(
        self, model: Recommender, user_id: int, k: int = 10, pool_size: int = 0
    ) -> dict[str, float]:
        interacted = self.get_users_interacted(user_id=user_id)
        not_interacted = self.get_users_not_interacted(
            user_id=user_id, pool_size=pool_size - len(interacted)
        )

        books = np.concatenate(
            [
                np.fromiter(iter=interacted, dtype=int, count=len(interacted)),
                np.fromiter(iter=not_interacted, dtype=int, count=len(not_interacted)),
            ],
            axis=0,
        )

        recommendations = set(model.recommend(user_id, books, k))

        user_metric = {
            "precision": self.precision_k(
                relevant=interacted, preds=recommendations, k=k
            ),
            "recall": self.recall_k(relevant=interacted, preds=recommendations, k=k),
        }

        return user_metric

    def evaluate(self, model: Recommender, k: int = 10, pool_size: int = 0):
        random.seed(self.seed)
        metrics = []

        for user in tqdm(self.user_interactions.index):
            metrics.append(
                self.evaluate_user(model=model, user_id=user, k=k, pool_size=pool_size)
            )

        metrics = pd.DataFrame(metrics)
        result = {
            "precision": metrics["precision"].mean(),
            "recall": metrics["recall"].mean(),
        }

        return result
