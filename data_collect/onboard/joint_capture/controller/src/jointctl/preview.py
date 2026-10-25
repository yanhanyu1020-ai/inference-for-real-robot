"""Validated read-only rosbridge preview settings for the local console."""
import ipaddress
import re
from urllib.parse import urlsplit


def preview_config(config):
    try:
        value = config.get('p450', {}).get('preview', {})
        url = value.get('url', 'ws://192.168.1.11:9090')
        parsed = urlsplit(url)
        address = ipaddress.IPv4Address(parsed.hostname)
        if (parsed.scheme not in ('ws', 'wss') or not address.is_private
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ('', '/') or not parsed.port):
            raise ValueError('invalid preview address')
        topic = value.get('topic', '/uav1/camera/color/image_raw/compressed')
        if not isinstance(topic, str) or not re.fullmatch(r'/[A-Za-z0-9_/]+/compressed', topic):
            raise ValueError('invalid compressed image topic')
        fps = value.get('fps', 5)
        if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 10:
            raise ValueError('preview fps must be 1..10')
        # Reconstruct rather than interpolate untrusted text into CSP.
        origin = f'{parsed.scheme}://{address}:{parsed.port}'
        return {'enabled': True, 'url': origin, 'topic': topic, 'fps': fps}
    except (ValueError, TypeError, AttributeError):
        return {'enabled': False, 'error': '无人机预览配置无效，请检查地址、话题和帧率。'}
