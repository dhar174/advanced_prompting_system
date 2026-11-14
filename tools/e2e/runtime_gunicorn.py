"""Test-only provider installation for runtime-image qualification."""
import provider_client
from pathlib import Path
if Path("/app/wsgi.py").is_file():
    assert Path(provider_client.__file__).parent == Path("/app"), provider_client.__file__
from tools.e2e.provider_fixture import FakeProviderClient
_fixture = FakeProviderClient()
provider_client.get_client = lambda: _fixture
