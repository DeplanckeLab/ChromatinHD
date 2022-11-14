import pathlib

class Flow():
    path:pathlib.Path

    def __init__(self, path = None, folder = None, name = None):
        if path is None:
            assert folder is not None
            if name is None:
                assert self.default_name is not None
                name = self.default_name

            path = folder / name

        self.path = path
        if not path.exists():
            path.mkdir(parents = True)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.path})"