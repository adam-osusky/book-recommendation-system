import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Self


@dataclass
class ConfigurableJob(ABC):
    """
    Lightweight base class for configurable jobs.
    """

    job_name: str

    @abstractmethod
    def run(self) -> None:
        pass

    def get_config(self) -> str:
        return json.dumps(self.__dict__, indent=4)

    @classmethod
    def from_json(cls, data: dict) -> Self:
        """
        Create an instance of the class from a JSON str.
        """

        # Generate a job name based on timestamp and random adjective-noun pair
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        adjective = random.choice(
            seq=[
                "red",
                "green",
                "blue",
                "fast",
                "slow",
                "happy",
                "sad",
                "strong",
                "accurate",
                "biased",
                "brave",
                "nice",
                "open",
            ]
        )
        noun = random.choice(
            seq=[
                "hinton",
                "lecun",
                "bengio",
                "goodfellow",
                "karpathy",
                "straka",
                "mikolov",
                "kaiming",
                "sutskever",
                "kingma",
                "goedel",
                "medved",
                "socrates",
                "aristotle",
            ]
        )

        job_name = f"{timestamp}_{adjective}_{noun}"
        data["job_name"] = job_name

        return cls(**data)
