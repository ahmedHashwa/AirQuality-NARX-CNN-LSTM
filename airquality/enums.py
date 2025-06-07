from enum import Enum, auto
class ScaleMethod(Enum):
    NoScaler = auto()
    MinMaxScaler = auto()
    StandardScaler = auto()


class ReshapeMethod(Enum):
    NoReshape = auto()
    TwoDShape = auto()
    ThreeDShape = auto()
    FourDShape = auto()


class SplitMode(Enum):
    KFold = auto()
    KFoldTimeSeries = auto()
    BlockingTimeSeriesSplit = auto()
    MonthsIntervals = auto()