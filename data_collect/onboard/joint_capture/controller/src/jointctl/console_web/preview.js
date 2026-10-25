(() => {
  'use strict';
  const button = document.getElementById('preview-toggle');
  const image = document.getElementById('preview-image');
  const placeholder = document.getElementById('preview-placeholder');
  const status = document.getElementById('preview-state');
  let config, socket, wanted = false, reconnect, generation = 0;
  let lastFrame = 0, previousStamp = null, verified = false, retry = 1000;
  const now = () => performance.now();
  function clear(message) {
    image.hidden = true;
    image.removeAttribute('src');
    placeholder.hidden = false;
    placeholder.textContent = message;
    status.textContent = message;
    status.className = 'pill neutral';
  }
  function disconnect() {
    ++generation;
    clearTimeout(reconnect);
    const old = socket;
    socket = null;
    if (old) {
      if (old.readyState === WebSocket.OPEN) {
        old.send(JSON.stringify({op: 'unsubscribe', id: 'joint-preview', topic: config.topic}));
      }
      old.close();
    }
  }
  function connect() {
    if (!wanted || document.hidden || socket) return;
    const mine = ++generation;
    previousStamp = null;
    verified = false;
    lastFrame = 0;
    clear('正在连接相机…');
    const ws = socket = new WebSocket(config.url);
    const timeout = setTimeout(() => { if (mine === generation) ws.close(); }, 8000);
    ws.onopen = () => {
      if (mine !== generation) return;
      clearTimeout(timeout);
      clear('等待相机画面…');
      ws.send(JSON.stringify({op: 'subscribe', id: 'joint-preview', topic: config.topic,
        type: 'sensor_msgs/CompressedImage', throttle_rate: Math.ceil(1000/config.fps),
        queue_length: 1, compression: 'none'}));
    };
    ws.onmessage = event => {
      if (mine !== generation || typeof event.data !== 'string' || event.data.length > 3000000) return;
      let packet;
      try { packet = JSON.parse(event.data); } catch (_) { return; }
      if (packet.op !== 'publish' || packet.topic !== config.topic) return;
      const msg = packet.msg, stamp = msg?.header?.stamp;
      if (!Number.isSafeInteger(stamp?.secs) || !Number.isSafeInteger(stamp?.nsecs)
          || stamp.nsecs < 0 || stamp.nsecs >= 1000000000) return;
      const current = [stamp.secs, stamp.nsecs];
      if (previousStamp) {
        if (current[0] === previousStamp[0] && current[1] === previousStamp[1]) return;
        if (current[0] < previousStamp[0] || (current[0] === previousStamp[0] && current[1] < previousStamp[1])) {
          previousStamp = current;
          verified = false;
          lastFrame = 0;
          clear('相机时间已重置，等待新画面…');
          return;
        }
        verified = true;
      }
      previousStamp = current;
      // Require advancing source timestamps before calling the image live.
      if (!verified || typeof msg.data !== 'string' || msg.data.length > 2800000
          || !/^[A-Za-z0-9+/]+={0,2}$/.test(msg.data)) return;
      const format = String(msg.format || '').toLowerCase();
      const mime = format.includes('jpeg') ? 'image/jpeg' : format.includes('png') ? 'image/png' : null;
      if (!mime) return;
      lastFrame = now();
      retry = 1000;
      image.src = `data:${mime};base64,${msg.data}`;
    };
    ws.onerror = () => { if (mine === generation) clear('画面连接失败，正在重连…'); };
    ws.onclose = () => {
      clearTimeout(timeout);
      if (mine !== generation) return;
      socket = null;
      clear('画面连接断开，正在重连…');
      if (wanted && !document.hidden) {
        reconnect = setTimeout(connect, retry);
        retry = Math.min(retry * 2, 10000);
      }
    };
  }
  image.onload = () => {
    if (!wanted || document.hidden || !lastFrame || now() - lastFrame > 2000) return;
    image.hidden = false;
    placeholder.hidden = true;
    status.textContent = '实时';
    status.className = 'pill green';
  };
  image.onerror = () => clear('图像解码失败，等待下一帧…');
  setInterval(() => {
    if (wanted && lastFrame && now() - lastFrame > 2000) clear('画面已中断，等待新画面…');
  }, 500);
  button.addEventListener('click', () => {
    wanted = !wanted;
    button.textContent = wanted ? '关闭预览' : '开启预览';
    if (wanted) connect();
    else { disconnect(); clear('预览已关闭'); }
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) { disconnect(); clear('预览已暂停'); }
    else connect();
  });
  window.addEventListener('pagehide', () => { wanted = false; disconnect(); });
  fetch('/api/preview', {cache: 'no-store'}).then(r => {
    if (!r.ok) throw new Error();
    return r.json();
  }).then(value => {
    config = value;
    if (config.enabled) button.disabled = false;
    else clear(config.error);
  }).catch(() => clear('读取预览配置失败，请刷新页面'));
})();
