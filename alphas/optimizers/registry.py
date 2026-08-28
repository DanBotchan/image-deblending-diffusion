from core.registry import Registry

CATEGORY = 'alpha_optimizer'


def register_alpha_optimizer(name: str):
    return Registry.register(CATEGORY, name)


def get_optimizer(name: str, **kwargs):
    return Registry.get(CATEGORY, name)(**kwargs)
