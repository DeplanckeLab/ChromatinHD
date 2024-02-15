from chromatinhd.data.motifscan import Motifscan
from chromatinhd.flow import StoredDataFrame

from . import plot


class Associations(Motifscan):
    association = StoredDataFrame()
