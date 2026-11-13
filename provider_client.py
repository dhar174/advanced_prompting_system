"""Lazy SDK access: importing the application never requires credentials."""
from functools import lru_cache


@lru_cache(maxsize=1)
def get_client():
    import openai
    return openai.OpenAI()


class LazyClient:
    """Resolve the SDK resource only when a method is called.

    The stable resource objects also allow tests to patch individual SDK calls
    without constructing a live client or supplying dummy credentials.
    """
    def __init__(self, path=()):
        self._path = path

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        resource = LazyClient(self._path + (name,))
        setattr(self, name, resource)
        return resource

    def __call__(self, *args, **kwargs):
        resource = get_client()
        for name in self._path:
            resource = getattr(resource, name)
        return resource(*args, **kwargs)
