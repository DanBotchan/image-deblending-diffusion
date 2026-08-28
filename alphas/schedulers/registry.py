from core.registry import Registry

CATEGORY = 'alpha_scheduler'


def register_alpha_scheduler(name: str):
    return Registry.register(CATEGORY, name)


def get_scheduler(name: str, **kwargs):
    return Registry.get(CATEGORY, name)(**kwargs)
