"""Assert the isolation harness and optional dependency contracts themselves."""
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import complexity_measures as complexity
import provider_client


def test_fresh_process_imports_and_network_denial():
    code = r'''
import os, sys, socket
assert sys._aps_network_guard
assert not os.getenv('OPENAI_API_KEY')
import openai, nltk, spacy
def forbidden(*a, **kw):
    raise AssertionError('Import tried to initialize a provider or download/load a model')
openai.OpenAI = forbidden
nltk.download = forbidden
spacy.load = forbidden
# Even a configured environment must not eagerly instantiate a client.
os.environ['OPENAI_API_KEY'] = 'offline-sentinel-not-a-credential'
import conversation_manager, complexity_measures, advanced_prompting
for attempt in (
    lambda: socket.getaddrinfo('example.com', 443),
    lambda: socket.socket().connect(('203.0.113.1', 443)),
    lambda: socket.socket(type=socket.SOCK_DGRAM).sendto(b'x', ('203.0.113.1', 53)),
):
    try:
        attempt()
    except BaseException as error:
        assert type(error).__name__ == 'NetworkForbidden', repr(error)
    else:
        raise AssertionError('External network was not denied')
with socket.socket() as server:
    server.bind(('127.0.0.1', 0))
    server.listen(1)
    with socket.create_connection(server.getsockname(), timeout=2) as client:
        connection, _ = server.accept()
        connection.close()
print('isolation verified')
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'isolation verified' in result.stdout


def test_lazy_client_resolves_only_when_called(monkeypatch):
    create = Mock(return_value='response')
    factory = Mock(return_value=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(provider_client, 'get_client', factory)
    proxy = provider_client.LazyClient()
    method = proxy.chat.completions.create
    assert proxy.chat.completions.create is method
    factory.assert_not_called()
    assert method(model='fixture') == 'response'
    factory.assert_called_once()
    create.assert_called_once_with(model='fixture')


def test_spacy_success_is_cached(monkeypatch):
    model = Mock()
    loader = Mock(return_value=model)
    monkeypatch.setattr(complexity, 'nlp_spacy', None)
    monkeypatch.setattr(complexity.spacy, 'load', loader)
    assert complexity.get_nlp_spacy() is model
    assert complexity.get_nlp_spacy() is model
    loader.assert_called_once_with('en_core_web_sm')


def test_spacy_failed_load_is_cached(monkeypatch):
    loader = Mock(side_effect=OSError('not installed'))
    monkeypatch.setattr(complexity, 'nlp_spacy', None)
    monkeypatch.setattr(complexity.spacy, 'load', loader)
    assert complexity.get_nlp_spacy() is None
    assert complexity.get_nlp_spacy() is None
    assert loader.call_count == 2


@pytest.mark.parametrize('missing', ['tokenizer', 'wordnet', None])
def test_lexical_filter_with_optional_nltk(monkeypatch, missing):
    token = SimpleNamespace(is_alpha=False, has_vector=False)
    monkeypatch.setattr(complexity, 'nlp_spacy', lambda text: [token])
    tokenizer = Mock(side_effect=LookupError('missing') if missing == 'tokenizer' else None, return_value=['Hello', 'world'])
    monkeypatch.setattr(complexity, 'word_tokenize', tokenizer)
    synsets = Mock(side_effect=LookupError('missing') if missing == 'wordnet' else None, return_value=['word'])
    monkeypatch.setattr(complexity, 'wn', SimpleNamespace(synsets=synsets))
    assert complexity.is_real_words('Hello world') == 'Hello world'


def test_embedding_missing_config_is_explicit(monkeypatch):
    monkeypatch.setattr(complexity, 'client', None)
    monkeypatch.delenv('PYTEST_CURRENT_TEST', raising=False)
    monkeypatch.setenv('OFFLINE_MODE', '0')
    with pytest.raises(RuntimeError, match='not configured'):
        complexity.get_embedding('configuration boundary')


def test_embedding_provider_error_is_not_masked(monkeypatch):
    monkeypatch.setattr(complexity, 'client', provider_client.LazyClient())
    monkeypatch.setattr(provider_client, 'get_client', Mock(side_effect=RuntimeError('fixture initialization failed')))
    with pytest.raises(ConnectionError, match='fixture initialization failed'):
        complexity.get_embedding('provider boundary')


def test_flask_readonly_startup():
    from app import app
    with app.test_client() as client:
        result = client.get('/graphql', query_string={'query': '{ hello }'})
        assert result.status_code == 200
        assert result.get_json() == {'data': {'hello': 'Hello, World!'}}
