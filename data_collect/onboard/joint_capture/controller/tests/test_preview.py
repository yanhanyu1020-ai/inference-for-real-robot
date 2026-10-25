import http.client
import json
import pytest
from jointctl.preview import preview_config
from test_console_http import server, request


def test_preview_configuration_and_static_asset(server):
    code, data = request(server, 'GET', '/api/preview')
    value = json.loads(data)
    assert code == 200 and value['url'] == 'ws://192.168.1.11:9090'
    assert value['fps'] == 5 and value['topic'].endswith('/compressed')
    assert request(server, 'GET', '/preview.js')[0] == 200
    assert server[1].job is None
    conn = http.client.HTTPConnection('127.0.0.1', server[0].server_port)
    conn.request('GET', '/')
    response = conn.getresponse()
    assert "connect-src 'self' ws://192.168.1.11:9090;" in response.getheader('Content-Security-Policy')
    response.read(); conn.close()


@pytest.mark.parametrize('url', ['ws://evil.example:9090', 'ws://8.8.8.8:9090',
    'ws://user:password@192.168.1.11:9090', 'ws://192.168.1.11:9090/?query=1',
    "ws://192.168.1.11:9090; script-src *", 'http://192.168.1.11:9090'])
def test_invalid_address_cannot_extend_csp(url):
    assert not preview_config({'p450': {'preview': {'url': url}}})['enabled']


@pytest.mark.parametrize('preview', [{'fps': 100}, {'fps': True}, {'topic': '/control'},
                                   {'topic': '/image/compressed;evil'}, None])
def test_invalid_preview_settings_are_disabled(preview):
    assert not preview_config({'p450': {'preview': preview}})['enabled']
