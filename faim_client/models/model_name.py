from enum import Enum


class ModelName(str, Enum):
    FLOWSTATE = "flowstate"
    TOTO = "toto"

    def __str__(self) -> str:
        return str(self.value)
