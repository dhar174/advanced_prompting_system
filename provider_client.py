"""Lazy SDK access: importing the application never requires credentials."""
from functools import lru_cache
import os


@lru_cache(maxsize=1)
def get_client():
    import openai
    return openai.OpenAI()


def has_configured_provider() -> bool:
    """Return True if credentials exist or a custom test provider fixture is active."""
    if os.getenv("OPENAI_API_KEY"):
        return True
    # Detect patched get_client (e.g., FakeProviderClient in tests or offline E2E),
    # while explicitly excluding the autouse conftest fail-closed guard.
    wrapped = getattr(get_client, "__wrapped__", None)
    if wrapped is None and not hasattr(get_client, "cache_info"):
        return getattr(get_client, "__name__", "") != "unexpected_call"
    return False


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
