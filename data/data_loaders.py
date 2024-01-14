from typing import Any

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

        self.use_feats = use_feats
        if use_feats:
            if trained_ohs:
                self.set_oh(trained_ohs)
            else:
                self.get_oh()

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx) -> dict[str, int | str]:
        sample = {
            "user_id": self.user_ids[idx],
            "book_id": self.book_ids[idx],
        }

        if self.use_feats:
            sample["user_features"] = self.user_oh.transform(
                self.user_feats[idx, None]
            )[0]
            sample["book_features"] = self.book_oh.transform(
                self.book_feats[idx, None]
            )[0]

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


def get_books_dataloaders(
    book_ds: str = "../data/book-feedback.csv",
    use_feats=False,
    batch_size=32,
    test_size: float = 0.1,
    val_size: float = 0.05,
    num_workers=0,
    seed: int = 69,
) -> Any:
    df = pd.read_csv(book_ds)
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["user_id"], random_state=seed
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, stratify=train_df["user_id"], random_state=seed
    )

    data_loaders = ()
    prev_ohs = None
    for data, is_train in zip([train_df, val_df, test_df], [True, False, False]):
        ds = BooksDataset(
            df=data, use_feats=use_feats, trained_ohs=None if is_train else prev_ohs
        )
        dl = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
        )
        data_loaders += (dl,)
        prev_ohs = [ds.user_oh, ds.book_oh] if hasattr(ds, "user_oh") else None
    return data_loaders
