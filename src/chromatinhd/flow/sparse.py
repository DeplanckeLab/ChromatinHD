from .objects import Obj, format_size, get_size
from chromatinhd.sparse import SparseDataset as SparseDataset_


class SparseDataset(Obj):
    def __init__(self, name=None):
        super().__init__(name=name)

    def get_path(self, folder):
        return folder / (self.name)

    def __get__(self, obj, type=None):
        if obj is not None:
            if self.name is None:
                raise ValueError(obj)
            name = "_" + str(self.name)
            if not hasattr(obj, name):
                path = self.get_path(obj.path)
                if not path.exists():
                    raise FileNotFoundError(f"File {path} does not exist")
                setattr(obj, name, SparseDataset_.open(self.get_path(obj.path)))
            return getattr(obj, name)

    def __set__(self, obj, value):
        name = "_" + str(self.name)
        setattr(obj, name, value)

    def exists(self, obj):
        return self.get_path(obj.path).exists()

    def _repr_html_(self, obj):
        self.__get__(obj)
        if not str(self.get_path(obj.path)).startswith("memory"):
            size = format_size(get_size(self.get_path(obj.path)))
        else:
            size = ""
        return f"<span class='iconify' data-icon='mdi-axis-arrow-info'></span> <b>{self.name}</b> {size}"
