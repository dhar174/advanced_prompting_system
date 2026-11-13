"""Probe the shipped server inside a network-disabled production container."""
import json
import os
import subprocess
import sys
from pathlib import Path
import time
import urllib.request

root = Path(__file__).resolve().parents[2]
os.environ.update(APS_OFFLINE_TESTS='1', OFFLINE_MODE='1', APS_PROVIDER_MAX_CALLS='200',
                  PYTHONPATH=f'{root}/tools/offline:/app:{root}')
server = subprocess.Popen([sys.executable, '-m', 'gunicorn', '--config', str(root / 'tools/e2e/runtime_gunicorn.py'),
                           '--bind', '127.0.0.1:8080', '--workers', '1', '--threads', '2', 'wsgi:app'])
try:
    for _ in range(100):
        try:
            urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=1)
            break
        except OSError:
            if server.poll() is not None:
                raise RuntimeError('Production server exited before health check')
            time.sleep(.1)
    for route in ('/', '/healthz', '/readyz'):
        assert urllib.request.urlopen('http://127.0.0.1:8080'+route).status == 200
    query = 'mutation($input: RunExecutionInput!){runExecution(input:$input){status finalOutput errorMessage}}'
    for mode in ('collaborative', 'advanced_pipeline'):
        payload = {'query': query, 'variables': {'input': {'mode': mode, 'taskPrompt': 'Write a concise answer', 'agents': 0, 'maxSteps': 4}}}
        req = urllib.request.Request('http://127.0.0.1:8080/graphql', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
        response = json.load(urllib.request.urlopen(req, timeout=60))
        assert 'errors' not in response, response
        assert response['data']['runExecution']['status'] == 'completed', response
    print('Production runtime: health, readiness, SPA, collaborative and advanced pipeline passed')
finally:
    server.terminate()
    server.wait(timeout=15)
