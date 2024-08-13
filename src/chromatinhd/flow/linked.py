from .objects import Obj, Instance, Linked


class LinkedDict(Obj):
    def __init__(self, cls=Linked, name=None, kwargs=None):
        super().__init__(name=name)
        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs
        self.cls = cls

    def get_path(self, folder):
        return folder / self.name

    def __get__(self, obj, type=None):
        if obj is not None:
            name = "_" + str(self.name)
            if not hasattr(obj, name):
                x = LinkedDictInstance(self.name, self.get_path(obj.path), self.cls, obj, self.kwargs)
                setattr(obj, name, x)
            return getattr(obj, name)

    def __set__(self, obj, value, folder=None):
        instance = self.__get__(obj)
        instance.__set__(obj, value)

    def _repr_html_(self, obj=None):
        instance = self.__get__(obj)
        return instance._repr_html_()

    def exists(self, obj):
        return True


class LinkedDictInstance(Instance):
    def __init__(self, name, path, cls, obj, kwargs):
        super().__init__(name=name, path=path, obj=obj)
        self.dict = {}
        self.cls = cls
        self.obj = obj
        self.name = name
        self.path = path
        self.kwargs = kwargs
        if not self.path.exists():
            self.path.mkdir(parents=True)
        for file in self.path.iterdir():
            key = file.name
            self.dict[key] = self.cls(name=key, **self.kwargs)

    def __getitem__(self, key):
        return self.dict[key].__get__(self)

    def __setitem__(self, key, value):
        if key not in self.dict:
            self.dict[key] = self.cls(name=key, **self.kwargs)
        self.dict[key].__set__(self, value)

    def __set__(self, obj, value, folder=None):
        for k, v in value.items():
            self[k] = v

    def __contains__(self, key):
        return key in self.dict

    def __len__(self):
        return len(self.dict)

    def items(self):
        for k in self.dict:
            yield k, self[k]

    def keys(self):
        return self.dict.keys()

    def exists(self):
        return True

    def _repr_html_(self):
        # return f"<span class='iconify' data-icon='mdi-layers'></span> <b>{self.name}</b> ({', '.join([getattr(self.obj, k)._repr_html_() for k in self.keys()])})"
        items = []
        for i, k in zip(range(3), self.keys()):
            items.append(self.dict[k]._repr_html_(self))
        if len(self.keys()) > 3:
            items.append("...")
        return f"<span class='iconify' data-icon='mdi-layers'></span> <b>{self.name}</b> ({', '.join(items)})"
