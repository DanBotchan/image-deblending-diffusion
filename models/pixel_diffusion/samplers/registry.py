__SAMPLERS__ = {}


def register_sampler_method(name: str):
    def wrapper(cls):
        if __SAMPLERS__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __SAMPLERS__[name] = cls
        return cls
    return wrapper


def get_sampler(name: str):
    if __SAMPLERS__.get(name, None) is None:
        raise NameError(f"Name: {name} is not defined!")
    return __SAMPLERS__[name]
