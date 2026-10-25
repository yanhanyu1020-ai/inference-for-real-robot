import http.client
import json
import threading
import pytest
from test_console_state import make_state, idle


@pytest.fixture
def server(tmp_path):
    from jointctl.console import make_server
    state=make_state(tmp_path,runner=lambda argv,folder:0)
    idle(state)
    httpd=make_server(state,port=0)
    thread=threading.Thread(target=httpd.serve_forever,daemon=True); thread.start()
    yield httpd,state
    httpd.shutdown(); httpd.server_close(); state.close(); thread.join()


def request(server,method,path,body=None,headers=None):
    httpd,_=server
    conn=http.client.HTTPConnection('127.0.0.1',httpd.server_port,timeout=3)
    conn.request(method,path,body=body,headers=headers or {})
    r=conn.getresponse(); data=r.read(); conn.close()
    return r.status,data


def test_foreign_host_cannot_read_token(server):
    assert request(server,'GET','/api/session',headers={'Host':'attacker.example'})[0]==403


def test_foreign_origin_and_missing_token_cannot_start(server):
    data=json.dumps({'action':'start','payload':{'instruction':'test','task':'joint'}})
    assert request(server,'POST','/api/actions',data,{'Content-Type':'application/json'})[0]==403
    token=json.loads(request(server,'GET','/api/session')[1])['token']
    assert request(server,'POST','/api/actions',data,{'Content-Type':'application/json',
        'X-Console-Token':token,'Origin':'https://evil.example'})[0]==403


def test_state_is_read_only_and_json_content_is_safe(server):
    code,data=request(server,'GET','/api/state')
    assert code==200 and json.loads(data)['hosts']['p450']['status']['state']=='idle'
    assert server[1].job is None


def test_valid_action_is_async_accepted(server):
    token=json.loads(request(server,'GET','/api/session')[1])['token']
    code,data=request(server,'POST','/api/actions',json.dumps({'action':'start','payload':{
        'instruction':'test','task':'joint'}}),{'Content-Type':'application/json','X-Console-Token':token,
        'Origin':f'http://127.0.0.1:{server[0].server_port}'})
    assert code==202 and json.loads(data)['job']['id']


def test_arbitrary_file_and_invalid_json_rejected(server):
    assert request(server,'GET','/../../config/default.json')[0]==404
    token=json.loads(request(server,'GET','/api/session')[1])['token']
    headers={'Content-Type':'application/json','X-Console-Token':token,
             'Origin':f'http://127.0.0.1:{server[0].server_port}'}
    assert request(server,'POST','/api/actions','[]',headers)[0]==400
    assert request(server,'POST','/api/actions','x'*17000,headers)[0]==413
