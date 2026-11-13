"""Offline suite policy. Use tools/run-offline.ps1 for subprocess protection."""
import os
import sys
from pathlib import Path

import pytest


def pytest_configure(config):
    if not getattr(sys, '_aps_network_guard', False):
        raise pytest.UsageError('Run via tools/run-offline.ps1: network guard must be active before collection.')
    config.option.basetemp = str(Path(__file__).parent / '.qualification' / 'pytest-tmp')


@pytest.fixture(autouse=True)
def forbid_unconfigured_provider(monkeypatch):
    import provider_client
    def unexpected_call():
        pytest.fail('Unconfigured SDK call: inject a provider response at this test boundary')
    monkeypatch.setattr(provider_client, 'get_client', unexpected_call)


@pytest.fixture(autouse=True)
def isolated_runtime_directory(monkeypatch, tmp_path):
    """Keep generated histories and artifacts away from tracked user files."""
    monkeypatch.chdir(tmp_path)


from types import SimpleNamespace
from unittest.mock import Mock
import conversation_manager as cm

def response(parsed):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])


@pytest.fixture
def conversation_case(monkeypatch):
    """Keep the production orchestration and vote logic; replace external helpers."""
    monkeypatch.setattr(cm, 'define_problem', lambda *a: 'Fixture task')
    monkeypatch.setattr(cm, 'output_type_determination', lambda *a: cm.OutputType(output_type='Text File', file_extension='.txt'))
    monkeypatch.setattr(cm, 'get_assistant_response', lambda *a, **kw: {'role': 'assistant', 'content': 'Fixture output'})
    monkeypatch.setattr(cm, 'extract_information', lambda text, memory, *a: memory)
    monkeypatch.setattr(cm, 'analyze_to_do_list', lambda memory, *a: memory)
    monkeypatch.setattr(cm, 'summarize_conversation', lambda *a: 'Fixture summary')
    monkeypatch.setattr(cm, 'analyze_final_output', lambda *a: 'Checked fixture output')
    finalize = Mock(return_value='fixture-output.txt')
    monkeypatch.setattr(cm, 'finalize_output', finalize)
    calls = []
    def parse(**kw):
        calls.append(kw['response_format'])
        if kw['response_format'] is cm.OutputType:
            return response(cm.OutputType(output_type='Text File', file_extension='.txt'))
        if kw['response_format'] is cm.MediatorReviseOrDecide:
            return response(cm.MediatorReviseOrDecide(final_decision='Fixture mediator', keep_output=True))
        pytest.fail(f"Unscripted parse {kw['response_format']}")
    monkeypatch.setattr(cm.client.beta.chat.completions, 'parse', parse)
    return finalize, calls
