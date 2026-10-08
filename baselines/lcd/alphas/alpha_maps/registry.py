__ALPHA_MAPS__ = {}

def register_alpha_map(name: str):
    def wrapper(cls):
        if __ALPHA_MAPS__.get(name, None):
            raise NameError(f"Name { name} is already registered!")
        __ALPHA_MAPS__[name] = cls
        return cls
    return wrapper


def get_alpha_map(name: str, **kwargs):
    if __ALPHA_MAPS__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __ALPHA_MAPS__[name](map_type=name, **kwargs)


def get_alpha_map_types():
    return list(__ALPHA_MAPS__.keys())