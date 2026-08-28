from core.registry import Registry

CATEGORY = 'alpha_controller'


def register_alpha_controller(name: str):
    return Registry.register(CATEGORY, name)


def get_alpha_controller(name: str, **kwargs):
    return Registry.get(CATEGORY, name)(**kwargs)
