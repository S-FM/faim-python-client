from enum import Enum


class ModelName(str, Enum):
    CHRONOS2 = "chronos2"
    FLOWSTATE = "flowstate"
    LIMIX = "limix"
    TIREX = "tirex"

    def __str__(self) -> str:
        return str(self.value)
