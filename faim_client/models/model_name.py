from enum import Enum


class ModelName(str, Enum):
    CHRONOS2 = "chronos2"
    FLOWSTATE = "flowstate"
    TIREX = "tirex"

    def __str__(self) -> str:
        return str(self.value)
