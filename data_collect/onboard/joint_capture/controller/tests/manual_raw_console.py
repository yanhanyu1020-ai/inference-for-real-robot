"""Read-only UI fixture on loopback; never calls robot or accepts actions."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

web=Path(__file__).resolve().parents[1]/'src/jointctl/console_web'
mode='running'


def snapshot():
    live=mode!='stopped'
    stream=dict(received=300,written=299,durable=295,received_fps=30,written_fps=29.9,
        pending_bytes=(48 if mode=='backlog' else 1)*1024**2,capacity_bytes=64*1024**2,
        write_bytes_per_s=12*1024**2,rejected=0,write_errors=0)
    state=dict(active=live,episode=dict(episode_id='UI_ONLY',state='recording' if live else 'complete',
        t0_desktop_ns='1780000000000000000'),hosts={},allowed={},clock={})
    for host in ('p450','unitree'):
        state['hosts'][host]=dict(status=dict(stale=mode=='disconnected',active=live,
            reachable=True,episode_id='UI_ONLY' if live else None),session_prepared=True)
    state['hosts']['unitree']['raw_capture']=dict(format_version=2,stale=mode=='disconnected',
        quality_ok=True,durable_complete=not live,streams=dict(front=stream,wrist=stream),
        disk_available_bytes=100*1024**3,remaining_minutes=45,fault=[])
    return state


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global mode
        url=urlsplit(self.path)
        if url.path=='/':
            mode=parse_qs(url.query).get('mode',['running'])[0]
            data=(web/'index.html').read_bytes(); mime='text/html'
        elif url.path=='/api/state':
            data=json.dumps(snapshot()).encode(); mime='application/json'
        elif url.path=='/api/session':
            data=b'{"token":"fixture-no-actions"}'; mime='application/json'
        elif url.path in ('/app.js','/style.css','/styles.css') and (web/url.path[1:]).is_file():
            data=(web/url.path[1:]).read_bytes()
            mime='text/javascript' if url.path.endswith('.js') else 'text/css'
        else:
            self.send_error(404); return
        self.send_response(200); self.send_header('Content-Type',mime+'; charset=utf-8')
        self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
    def log_message(self,*args): pass


if __name__=='__main__':
    print('Read-only raw UI fixture: http://127.0.0.1:18766',flush=True)
    HTTPServer(('127.0.0.1',18766),Handler).serve_forever()
