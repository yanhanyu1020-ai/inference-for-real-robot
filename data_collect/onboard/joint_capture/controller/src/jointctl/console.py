"""Loopback-only HTTP boundary and detached desktop launcher."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

from .console_state import ConsoleState
from .telemetry import safe_json
from .preview import preview_config

IDENTITY = 'joint-capture-console-v1'
WEB_ROOT = Path(__file__).with_name('console_web')


def make_server(state, port=8766):
    token = secrets.token_urlsafe(32)
    preview = preview_config(getattr(state, 'config', {}))
    preview_origin = ' ' + preview['url'] if preview['enabled'] else ''

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, fmt, *args):
            # Do not record request bodies or security tokens.
            pass

        def _trusted(self, mutation=False):
            expected = f'127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host') != expected:
                return False
            origin = self.headers.get('Origin')
            if origin is not None and origin != 'http://' + expected:
                return False
            if mutation:
                return origin == 'http://' + expected and secrets.compare_digest(
                    self.headers.get('X-Console-Token', ''), token)
            return True

        def _reply(self, code, value, mime='application/json; charset=utf-8'):
            payload = value if isinstance(value, bytes) else json.dumps(safe_json(value), ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; connect-src 'self'" + preview_origin + "; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (ConnectionError, OSError):
                pass

        def do_GET(self):
            if not self._trusted():
                return self._reply(403, {'error': '仅允许本机同源访问'})
            if self.path == '/api/health':
                return self._reply(200, {'identity': IDENTITY, 'root': str(state.root),
                                         'config': str(state.config_path), 'pid': os.getpid()})
            if self.path == '/api/session':
                return self._reply(200, {'token': token})
            if self.path == '/api/preview':
                return self._reply(200, preview)
            if self.path == '/api/state':
                try:
                    return self._reply(200, state.snapshot())
                except Exception as exc:
                    return self._reply(500, {'error': f'读取状态失败：{exc}'})
            static = {'/': ('index.html', 'text/html; charset=utf-8'),
                      '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                      '/preview.js': ('preview.js', 'text/javascript; charset=utf-8'),
                      '/style.css': ('style.css', 'text/css; charset=utf-8')}
            if self.path not in static:
                return self._reply(404, {'error': 'not found'})
            filename, mime = static[self.path]
            try:
                content = (WEB_ROOT/filename).read_bytes()
            except OSError:
                return self._reply(503, {'error': '网页资源尚未安装'})
            return self._reply(200, content, mime)

        def do_POST(self):
            if not self._trusted(mutation=True):
                return self._reply(403, {'error': '操作来源或令牌无效，请刷新本机页面'})
            if self.path != '/api/actions':
                return self._reply(404, {'error': 'not found'})
            if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
                return self._reply(415, {'error': '需要 JSON 请求'})
            try:
                length = int(self.headers.get('Content-Length', '-1'))
            except ValueError:
                length = -1
            if length < 0 or self.headers.get('Transfer-Encoding'):
                return self._reply(400, {'error': '需要固定长度请求'})
            if length > 16384:
                # Closing with an unread upload can reset the connection on
                # Windows before the browser receives the rejection. Drain a
                # bounded amount; never allocate/read an arbitrary upload.
                if length <= 65536:
                    try:
                        self.rfile.read(length)
                    except OSError:
                        pass
                return self._reply(413, {'error': '请求过大'})
            try:
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or set(payload) != {'action','payload'}:
                    raise ValueError('操作格式无效')
                result = state.submit(payload['action'], payload['payload'])
                return self._reply(202, {'job': result})
            except (ValueError, TypeError) as exc:
                return self._reply(400, {'error': str(exc)})
            except RuntimeError as exc:
                return self._reply(409, {'error': str(exc)})
            except Exception as exc:
                return self._reply(500, {'error': f'提交失败：{exc}'})

        def do_OPTIONS(self):
            self._reply(403, {'error': 'cross-origin access disabled'})

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    return server


def health(port):
    # Never use a system proxy for local control.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f'http://127.0.0.1:{port}/api/health', timeout=1) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def launch(config, root, port, *, open_browser=True):
    config, root = Path(config).resolve(), Path(root).resolve()
    def matches(info):
        return isinstance(info, dict) and info.get('identity') == IDENTITY and info.get('root') == str(root) and info.get('config') == str(config)
    existing = health(port)
    if existing is not None and not matches(existing):
        raise RuntimeError('端口正在由其他服务或其他采集配置使用；不会复用。')
    if not matches(existing):
        folder = root/'.console'
        folder.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, '-m', 'jointctl.console', '--serve', '--config', str(config),
                '--manifest-root', str(root), '--port', str(port)]
        env = dict(os.environ)
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get('PYTHONPATH','')
        env['PYTHONIOENCODING'] = 'utf-8'
        options = {'start_new_session': True} if os.name != 'nt' else {
            'creationflags': subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
        with (folder/'server.log').open('ab') as output:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=output, stderr=output, env=env, **options)
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            if matches(health(port)):
                break
            if process.poll() is not None:
                # Another simultaneous launcher may have won the bind race.
                if not matches(health(port)):
                    raise RuntimeError(f'后台启动失败，查看 {folder / "server.log"}')
                break
            time.sleep(.2)
        else:
            raise RuntimeError(f'后台就绪超时，查看 {folder / "server.log"}')
    url = f'http://127.0.0.1:{port}/'
    if open_browser:
        webbrowser.open(url)
    return url


def main(argv=None):
    parser = argparse.ArgumentParser(description='P450 / Unitree 本机采集控制台')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[2]/'config/default.json')
    parser.add_argument('--manifest-root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--serve', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('port must be 1..65535')
    try:
        if not args.serve:
            print(launch(args.config, args.manifest_root, args.port))
            return 0
        state = ConsoleState(args.config, args.manifest_root)
        server = make_server(state, args.port)
        state.start_polling()
        try:
            server.serve_forever()
        finally:
            server.server_close()
            state.close()
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
