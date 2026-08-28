from core.registry import Registry

CATEGORY = 'latent_sampler'


def register_sampler_method(name: str):
    return Registry.register(CATEGORY, name)


def get_sampler(name: str, **kwargs):
    return Registry.get(CATEGORY, name)(**kwargs)
