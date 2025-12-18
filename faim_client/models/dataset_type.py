from enum import Enum


class DatasetType(str, Enum):
    FINE_TUNE = "fine_tune"
    TABULAR_PRETRAIN = "tabular_pretrain"

    def __str__(self) -> str:
        return str(self.value)
