from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset


class BooksDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        use_feats: bool = False,
        trained_ohs: list[OneHotEncoder] | None = None,
        rating_target: bool = False,
        num_books: int = 0,
    ) -> None:
        name_to_idx = df.columns.to_series().reset_index(drop=True).to_dict()
        name_to_idx = {v: k for k, v in name_to_idx.items()}
        self.name_to_idx = name_to_idx

        self.user_ids = df.iloc[:, name_to_idx["user_id"]].values
        user_cols = [name_to_idx["Location"], name_to_idx["Age"]]
        self.user_feats = df.iloc[:, user_cols].values

        self.book_ids = df.iloc[:, name_to_idx["book_id"]].values
        book_cols = [
            name_to_idx["Book-Title"],
            name_to_idx["Book-Author"],
            name_to_idx["Year-Of-Publication"],
            name_to_idx["Publisher"],
        ]
        self.book_feats = df.iloc[:, book_cols].values

        self.ratings = df.iloc[:, name_to_idx["Book-Rating"]]
        self.rating_target = rating_target

        self.use_feats = use_feats
        if use_feats:
            if trained_ohs:
                self.set_oh(trained_ohs)
            else:
                self.get_oh()

        self.book_features_dict = (
            df.drop_duplicates(subset="book_id", keep="first")
            .fillna(value="")
            .set_index("book_id")[
                ["Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"]
            ]
            .to_dict("index")
        )
        self.all_books = set(self.book_ids)
        self.user_rated_books = (
            df.groupby("user_id")["book_id"].unique().apply(set).to_dict()
        )
        self.num_books = num_books

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx) -> dict[str, int | str]:
        user_id = self.user_ids[idx]
        neg_book_id, neg_book_feat = self.get_negative_book(user_id=user_id)
        sample = {
            "user_id": user_id,
            "pos_book_id": self.book_ids[idx],
            "neg_book_id": neg_book_id,
        }

        if self.use_feats:
            sample["user_features"] = self.user_oh.transform(
                self.user_feats[idx, None]
            )[0]  # type: ignore
            sample["pos_book_features"] = self.book_oh.transform(
                self.book_feats[idx, None]
            )[0]  # type: ignore
            sample["neg_book_features"] = self.book_oh.transform(neg_book_feat)

        if self.rating_target:
            sample["rating"] = self.ratings[idx]

        return sample

    def get_oh(self) -> None:
        self.user_oh = OneHotEncoder(
            sparse_output=False,
            min_frequency=100,
            handle_unknown="infrequent_if_exist",
        ).fit(self.user_feats)

        self.book_oh = OneHotEncoder(
            sparse_output=False,
            min_frequency=100,
            handle_unknown="infrequent_if_exist",
        ).fit(self.book_feats)

    def set_oh(self, encs: list[OneHotEncoder]) -> None:
        self.user_oh = encs[0]
        self.book_oh = encs[1]

    def get_negative_book(self, user_id):
        # book_id = random.choice(list(self.all_books - self.user_rated_books[user_id]))  NOTE: this line was a huge bottleneck
        done = False
        while not done:
            book_id = random.randint(0, self.num_books - 1)
            if book_id not in self.user_rated_books[user_id]:
                done = True

        book_feat = None
        if self.use_feats:
            book_feat = self.book_features_dict.get(
                book_id, {0: "", 1: "", 2: "", 3: ""}
            )
            book_feat = np.array([list(book_feat.values())])
        return book_id, book_feat


def train_val_test_split(
    df: pd.DataFrame, test_size: float = 0.05, val_size: float = 0.05, seed=69
) -> tuple[Any, Any, Any]:
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["user_id"], random_state=seed
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, stratify=train_df["user_id"], random_state=seed
    )
    return train_df, val_df, test_df


@dataclass
class BookData:
    dls: tuple[DataLoader, ...]
    num_users: int
    num_books: int
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_test: pd.DataFrame


def get_books_data(
    book_ds: str = "./data/book-feedback.csv",
    use_feats=False,
    batch_size=32,
    test_size: float = 0.05,
    val_size: float = 0.05,
    num_workers=0,
    seed: int = 69,
) -> Any:
    df = pd.read_csv(book_ds)
    train_df, val_df, test_df = train_val_test_split(
        df=df, test_size=test_size, val_size=val_size, seed=seed
    )
    num_books = df["book_id"].nunique()

    data_loaders = ()
    prev_ohs = None
    for data, is_train in zip([train_df, val_df, test_df], [True, False, False]):
        ds = BooksDataset(
            df=data,
            use_feats=use_feats,
            trained_ohs=None if is_train else prev_ohs,
            num_books=num_books,
        )
        dl = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
        )
        data_loaders += (dl,)
        prev_ohs = [ds.user_oh, ds.book_oh] if hasattr(ds, "user_oh") else None

    data = BookData(
        dls=data_loaders,
        num_users=df["user_id"].nunique(),
        num_books=num_books,
        df_train=train_df,
        df_val=val_df,
        df_test=test_df,
    )
    return data
