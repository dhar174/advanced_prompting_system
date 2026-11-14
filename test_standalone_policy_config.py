"""Issue51 canonical configuration and standalone limit precedence."""
import math

import pytest

import advanced_prompting as ap
import execution_modes as em
import provider_client as pc


@pytest.fixture(autouse=True)
def canonical_environment(monkeypatch):
    for key in ('APS_DEFAULT_MODEL','APS_ALLOWED_MODELS','APS_DEFAULT_TEMPERATURE',
                'APS_PROVIDER_TIMEOUT_SECONDS','APS_PROVIDER_MAX_CALLS'):
        monkeypatch.delenv(key,raising=False)


def test_provider_configuration_is_owned_by_provider_module():
    assert em.ProviderConfig is pc.ProviderConfig
    assert em.get_provider_config is pc.get_provider_config
    assert em.validate_provider_model is pc.validate_provider_model
    assert pc.get_provider_config().max_calls==40
    assert pc.get_provider_config().timeout_seconds==120


@pytest.mark.parametrize('key,value', [
    ('APS_PROVIDER_MAX_CALLS','bad'),('APS_PROVIDER_MAX_CALLS','1.5'),
    ('APS_PROVIDER_MAX_CALLS','0'),('APS_PROVIDER_MAX_CALLS','201'),
    ('APS_PROVIDER_TIMEOUT_SECONDS','nan'),('APS_PROVIDER_TIMEOUT_SECONDS','inf'),
    ('APS_PROVIDER_TIMEOUT_SECONDS','0'),('APS_PROVIDER_TIMEOUT_SECONDS','601'),
    ('APS_DEFAULT_TEMPERATURE','nan'),('APS_DEFAULT_TEMPERATURE','2.1'),
])
def test_invalid_canonical_policy_is_rejected_instead_of_clamped(monkeypatch,key,value):
    monkeypatch.setenv(key,value)
    with pytest.raises(ValueError):
        em.get_provider_config()


def test_standalone_limits_preserve_zero_and_optional_omission():
    omitted=ap.PromptEngineeringConfig()
    limited=ap.PromptEngineeringConfig(max_calls=0,timeout_seconds=.05)
    assert omitted.max_calls is None and omitted.timeout_seconds is None
    assert not omitted._model_explicit and omitted.model=='gpt-4o-mini'
    assert limited.max_calls==0 and limited.timeout_seconds==.05
    assert ap.PromptEngineeringConfig(model='gpt-4o-mini')._model_explicit


@pytest.mark.parametrize('kwargs', [
    {'max_calls':-1},{'max_calls':True},{'max_calls':1.5},{'max_calls':201},
    {'timeout_seconds':0},{'timeout_seconds':False},{'timeout_seconds':math.nan},
    {'timeout_seconds':math.inf},{'timeout_seconds':601},
])
def test_invalid_constructor_limits_fail_clearly(kwargs):
    with pytest.raises(ValueError):
        ap.PromptEngineeringConfig(**kwargs)


def test_standalone_resolution_inherits_environment_and_only_tightens(monkeypatch):
    monkeypatch.setenv('APS_PROVIDER_MAX_CALLS','12')
    monkeypatch.setenv('APS_PROVIDER_TIMEOUT_SECONDS','4')
    monkeypatch.setenv('APS_DEFAULT_MODEL','gpt-4o')
    inherited=pc.resolve_standalone_policy()
    assert inherited['default_model']=='gpt-4o'
    assert inherited['max_calls']==12 and inherited['timeout_seconds']==4
    assert pc.resolve_standalone_policy(max_calls=0,timeout_seconds=1)['max_calls']==0
    for kwargs in ({'max_calls':13},{'timeout_seconds':5}):
        with pytest.raises(ValueError):
            pc.resolve_standalone_policy(**kwargs)


def test_default_model_must_be_in_explicit_allowlist(monkeypatch):
    monkeypatch.setenv('APS_ALLOWED_MODELS','gpt-4o')
    with pytest.raises(ValueError):
        em.get_provider_config()
