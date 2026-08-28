from core.registry import Registry

CATEGORY = 'alpha_map'


def register_alpha_map(name: str):
    return Registry.register(CATEGORY, name)


def get_alpha_map(name: str, **kwargs):
    return Registry.get(CATEGORY, name)(map_type=name, **kwargs)


def get_alpha_map_types():
    return Registry.list(CATEGORY)
