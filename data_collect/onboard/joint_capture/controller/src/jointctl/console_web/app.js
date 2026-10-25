(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  let token = null;
  let online = false;
  let submitting = false;
  let lastState = null;
  let lastSuccess = null;
  let polling = false;
  let submittedJobId = null;
  const actions = ['start', 'stop', 'recover', 'finalize', 'align', 'prepare'];
  const labels = {start: '开始采集', stop: '停止并保存', recover: '核对遗留会话', finalize: '校验待处理数据', align: '重新校验本条', prepare: '初始化设备'};
  const expandedDevices = new Set();
  const phases = {running: '执行中', succeeded: '已完成', failed: '失败', interrupted: '曾中断', validating: '校验中'};
  const states = {recording: '采集中', active: '采集中', idle: '空闲', stopped: '已停止', cleanup_pending: '待停止确认', unknown: '未知'};
  const gib = (n) => n !== null && n !== undefined && n !== '' && Number.isFinite(Number(n)) ? `${(Number(n) / 1073741824).toFixed(1)} GiB` : '未知';
  const count = (n) => n === null || n === undefined ? '未知' : Number(n).toLocaleString('zh-CN');
  const time = (s) => Number.isFinite(Number(s)) && s !== null ? new Date(Number(s) * 1000).toLocaleString('zh-CN', {hour12: false}) : '未知';
  const duration = (s) => { if (!Number.isFinite(Number(s)) || s === null) return '--:--:--'; const v = Math.max(0, Math.floor(Number(s))); return [Math.floor(v / 3600), Math.floor(v % 3600 / 60), v % 60].map(x => String(x).padStart(2, '0')).join(':'); };
  function intervalDuration(interval) {
    try { const start = BigInt(interval.start_inclusive_ns); const end = BigInt(interval.end_exclusive_ns); return end > start ? `${(Number(end - start) / 1e9).toFixed(2)} 秒` : '无有效区间'; }
    catch { return '未知'; }
  }
  const set = (id, value) => { $(id).textContent = value === null || value === undefined || value === '' ? '未知' : String(value); };
  function badge(id, label, tone) { const el = $(id); el.textContent = label; el.className = `pill ${tone}`; }
  function node(tag, cls, value) { const el = document.createElement(tag); if (cls) el.className = cls; if (value !== undefined) el.textContent = value; return el; }
  function row(parent, label, value) { const el = node('div', 'data-row'); el.append(node('span', '', label), node('strong', '', value)); parent.append(el); }
  function path(parent, label, value, current) {
    const line = node('div', 'path-line'); line.append(node('span', 'field-label', label));
    const box = node('div', 'path-value'); const code = node('code', '', value || '尚未取得');
    const copy = node('button', 'copy', '复制'); copy.type = 'button'; copy.disabled = !value; copy.dataset.value = value || '';
    box.append(code, copy); line.append(box);
    if (value && !current) line.append(node('span', 'historical', '上次采集路径'));
    parent.append(line);
  }
  function resource(parent, title, data, stale, warning) {
    const card = node('div', `resource ${warning ? 'warning' : ''}`);
    card.append(node('div', 'resource-title', title));
    if (!data || data.total_bytes === undefined) { card.append(node('strong', 'resource-number', '未知')); }
    else {
      const main = title === 'RAM / 整机内存' ? `${gib(data.used_bytes)} / ${gib(data.total_bytes)}` : `${gib(data.available_bytes)} 可用`;
      card.append(node('strong', 'resource-number', main));
      card.append(node('div', 'resource-sub', `${title.startsWith('RAM') ? '可用 ' + gib(data.available_bytes) : '总计 ' + gib(data.total_bytes)} · 已用 ${Number(data.used_percent).toFixed(1)}%`));
      const track = node('div', 'meter'); const bar = node('span'); bar.style.width = `${Math.min(100, Math.max(0, Number(data.used_percent) || 0))}%`; track.append(bar); card.append(track);
    }
    if (stale) card.append(node('div', 'stale-note', '资源信息过期 · 显示上次数据'));
    if (warning) card.append(node('div', 'stale-note', '磁盘可用空间低于警戒值'));
    parent.append(card);
  }
  function rawHealth(parent, raw, stale) {
    if (raw?.format_version !== 2) return;
    const section = node('div', 'raw-health');
    section.append(node('div', 'field-label', `MCAP 原始数据${stale || raw.stale ? ' · 上次状态（过期）' : ''}`));
    const decimal = value => value != null && Number.isFinite(Number(value)) ? Number(value).toFixed(1) : '—';
    const grid = node('div', 'raw-stream-grid'); section.append(grid);
    for (const [key, label] of [['front', '前置'], ['wrist', '腕部']]) {
      const camera = node('div', 'raw-stream'); grid.append(camera);
      const s = raw.streams?.[key] || {};
      camera.append(node('strong', '', `${label}相机`));
      row(camera, '接收/写入 FPS', `${decimal(s.received_fps)} / ${decimal(s.written_fps)}`);
      row(camera, '写入/持久化', `${count(s.written)} / ${count(s.durable)}`);
      const ratio = s.capacity_bytes > 0 ? Math.min(100, Math.max(0, Number(s.pending_bytes) / Number(s.capacity_bytes) * 100)) : null;
      row(camera, '队列/写盘', `${ratio == null ? '未知' : ratio.toFixed(0) + '%'} · ${decimal(s.write_bytes_per_s == null ? null : s.write_bytes_per_s / 1048576)} MiB/s`);
      const track = node('div', 'meter'); const bar = node('span');
      bar.style.width = `${ratio || 0}%`; if (ratio >= 70) bar.style.background = 'var(--amber, #b57719)';
      track.append(bar); camera.append(track);
      row(camera, '拒收/错误', `${count(s.rejected)} / ${count(s.write_errors)}`);
    }
    row(section, '磁盘预算（估计）', `${gib(raw.disk_available_bytes)} · 约 ${decimal(raw.remaining_minutes)} 分钟`);
    row(section, '文件收尾 / 数据质量', `${raw.durable_complete === true ? '已落盘' : '待收尾确认'} / ${raw.quality_ok === true ? '未报告错误' : raw.quality_ok === false ? '异常' : '未知'}`);
    section.append(node('div', 'muted', '容量按未压缩速率估计并预留 10 GiB；对时结果见独立验收区。'));
    for (const error of raw.fault || []) section.append(node('div', 'error-note', String(error)));
    parent.append(section);
  }
  function host(id, info, threshold, recording) {
    const target = $(`${id}-content`); target.replaceChildren();
    const status = info?.status || {}; const resources = info?.resources || {};
    const stale = !online || status.stale !== false;
    const active = !stale && status.active === true;
    const idle = !stale && status.active === false;
    badge(`${id}-badge`, stale ? '离线 / 过期' : active ? recording ? '采集中' : '活动 · 待核对' : idle ? '已连接 · 空闲' : '状态未知', stale ? 'amber' : active ? recording ? 'green' : 'amber' : idle ? 'subtle' : 'amber');
    row(target, '连接 / 采集状态', stale ? '未知（数据过期）' : `${status.reachable ? '已连接' : '未知'} / ${states[status.state] || status.state || '未知'}`);
    if (id === 'p450') {
      const v = info?.vehicle;
      row(target, 'MID360 定位', stale ? '未知（状态过期）' : v?.odom_valid === true ? '定位有效' : '尚未就绪');
      for (const [key, label] of [['mid360_driver', '雷达驱动'], ['fast_lio', '定位算法'], ['d435i', '相机驱动']]) row(target, label, stale ? '未知' : info?.components?.[key]?.running ? '运行中' : '未运行');
    } else {
      const devices = resources.joysticks;
      row(target, '手柄识别', !stale && info?.gamepad ? info.gamepad.connected ? '已连接' : '未连接' : resources.stale === false && Array.isArray(devices) ? devices.length ? `系统已识别 ${devices.join(', ')}（待遥操确认）` : '未识别，请检查手柄 / 接收器' : '未知');
      const motors = info?.motor_enabled;
      row(target, '关节电机', stale || !Array.isArray(motors) || motors.length !== 6 ? '未确认' : motors.every(v => v === true) ? '已使能 · 保持位置' : motors.every(v => v === false) ? '已失能' : '部分使能 · 检查机械臂');
      row(target, '手柄遥操', stale || info?.arm_enabled == null ? '未确认' : info.arm_enabled && !info.command_inhibited ? '已启用' : '未启用');
      row(target, '遥操模式', stale ? '未知' : info?.teleop_mode === 'pose' ? '末端坐标移动' : info?.teleop_mode === 'joint' ? '关节控制' : '未知');
      if (!stale && info?.command_inhibited) row(target, '遥操保护', '已禁止发送目标 · 重连后松开摇杆并按 Home 确认');
      if (!stale && info?.teleop_error) target.append(node('div', 'error-note', String(info.teleop_error)));
      if (!stale && info?.feedback_error && info.feedback_error !== info.teleop_error) target.append(node('div', 'error-note', String(info.feedback_error)));
      row(target, '常驻设备', stale ? '未知' : info?.session_prepared ? '已初始化 · 可连续采集' : '需要初始化 / 检查设备');
      for (const [key, label] of [['front', '前置相机'], ['wrist', '腕部相机']]) {
        const camera = info?.camera_health?.[key];
        row(target, label, stale || !camera ? '未确认' : camera.ready ? '持续出帧' : camera.error ? '异常 · 检查 USB' : '未出帧');
      }
      const speedKnown = !stale && info?.gamepad?.connected === true;
      row(target, '摇杆倍率', speedKnown && Number.isFinite(info?.speed_factor) ? `${info.speed_factor}×` : '未知');
      row(target, '机械臂指令速度', speedKnown && Number.isFinite(info?.movement_speed) ? `${info.movement_speed}%` : '未知');
      if (info?.raw_capture?.format_version === 2) {
        const raw = info.raw_capture;
        row(target, active ? '原始数据质量' : '上条原始数据质量', raw.quality_ok === true ? '未报告错误' : raw.quality_ok === false ? '异常 · 展开详情' : '未知');
        for (const error of raw.fault || []) target.append(node('div', 'error-note', String(error)));
      }
    }
    row(target, '状态查询耗时', status.query_ms == null ? '未知' : `${Number(status.query_ms).toFixed(0)} ms${stale ? '（上次）' : ''}`);
    if (info?.write_stalled) target.append(node('div', 'error-note', `写入已连续 ${Math.floor(status.progress_unchanged_s)} 秒没有增长，请检查传感器 / 相机；连接成功不等于数据正常。`));
    if (status.error) target.append(node('div', 'error-note', String(status.error)));
    if (id === 'unitree' && !stale && info?.session_error) target.append(node('div', 'error-note', '设备初始化未成功，请检查相机后重新初始化。'));
    if (status.progress_name) row(target, id === 'p450' ? '本次写入' : '已保存帧数', id === 'p450' && status.progress_name === 'session_bytes' ? gib(status.progress_value) : count(status.progress_value));
    else row(target, id === 'p450' ? '本次写入' : '已保存帧数', '未知');
    const details = node('details', 'device-details');
    details.open = expandedDevices.has(id);
    details.append(node('summary', '', '展开数据路径、帧数与资源详情'));
    details.addEventListener('toggle', () => { if (details.isConnected) { if (details.open) expandedDevices.add(id); else expandedDevices.delete(id); } });
    target.append(details);
    if (id === 'unitree') rawHealth(details, info?.raw_capture, stale || !active);
    row(details, '最后成功采样', time(status.updated_at));
    if (id === 'unitree') {
      row(details, '丢弃帧数', count(info?.frames_dropped)); row(details, '保存错误', count(info?.save_errors));
      if (Number(info?.frames_dropped) > 0) target.prepend(node('div', 'error-note', `采集器正在丢帧 / Dropped: ${count(info.frames_dropped)} 组；本条不能按完整数据验收。`));
      if (Number(info?.save_errors) > 0) target.prepend(node('div', 'error-note', '存在保存错误，请展开设备详情检查。'));
    }
    path(details, '远端数据路径', info?.path, info?.path_is_current);
    const grid = node('div', 'resource-grid');
    resource(grid, 'RAM / 整机内存', resources.memory, !online || resources.stale !== false, false);
    const disk = resources.disk; resource(grid, 'DISK / 数据盘', disk, !online || resources.stale !== false, !!disk && Number(disk.available_bytes) < Number(threshold));
    details.append(grid);
    const sampled = node('div', 'resource-sample', `资源采样 ${time(resources.updated_at)}${resources.error ? ' · ' + resources.error : ''}`); details.append(sampled);
  }
  function render(state) {
    lastState = state;
    const ep = state.episode || {}; const job = state.job || null;
    const ownedAndFresh = ['p450', 'unitree'].every(id => { const remote = state.hosts?.[id]?.status; return remote?.stale === false && remote.active === true && remote.episode_id === ep.episode_id; });
    const stalled = ['p450', 'unitree'].some(id => state.hosts?.[id]?.write_stalled);
    const raw = state.hosts?.unitree?.raw_capture;
    const rawUnhealthy = raw?.format_version === 2 && (raw.stale || raw.quality_ok !== true || raw.fault?.length > 0);
    const legacyUnhealthy = Number(state.hosts?.unitree?.frames_dropped) > 0 || Number(state.hosts?.unitree?.save_errors) > 0;
    const recording = online && state.active === true && ep.state === 'recording' && ep.t0_desktop_ns != null && !!ep.episode_id && ownedAndFresh && !stalled && !rawUnhealthy && !legacyUnhealthy;
    const stage = ep.state;
    const allIdle = ['p450', 'unitree'].every(id => { const remote = state.hosts?.[id]?.status; return remote?.stale === false && remote.active === false; });
    const interruptedStart = stage === 'starting' && state.starter_alive === false;
    const title = !online ? '连接中断 · 采集状态未确认' : recording ? '联合采集中' : interruptedStart ? '启动已中断 · 请停止清理' : stage === 'starting' ? '准备中，请等待' : stage === 'stopping' ? '正在停止录制' : stage === 'recording' || stage === 'partial' || stage === 'stop_failed' || stage === 'recovered' ? '采集状态待核对' : job?.state === 'running' ? '操作执行中' : stage === 'complete' && allIdle ? '待命 · 上次采集已结束' : '等待设备就绪';
    set('session-title', title);
    set('session-subtitle', recording ? '两端正在采集，共同开始时间已记录。结束时请点击“停止并保存”。' : interruptedStart ? '启动进程已经退出，远端状态不一致。请点击“停止并保存”清理本次残留。' : stage === 'starting' ? '正在准备两端设备；只有双方就绪后才开始共同采集。' : stage === 'stopping' ? '正在停止两端写入并保存结束信息；验收稍后进行。' : state.active ? '本机有活动会话，但远端状态未共同确认；请核对设备状态。' : job?.state === 'running' ? '任务已受理，正在执行。请等待状态更新。' : '开始前请确认两台设备均已连接并空闲。');
    badge('session-pill', recording ? 'LIVE · 采集中' : !online ? '连接中断 · 上次状态' : state.active || state.operation_busy ? '未确认 / 执行中' : '待命', recording ? 'green' : state.active || state.operation_busy || !online ? 'amber' : 'neutral');
    set('episode-id', ep.episode_id || '尚无'); set('episode-label', ep.mode || '尚无'); set('episode-phase', ({starting:'启动中',recording:'采集中',stopping:'停止中',complete:'已停止',start_failed:'启动失败',partial:'需要核对',stop_failed:'停止失败',recovered:'已恢复待核对'})[ep.state] || ep.state || '尚无');
    set('elapsed', duration(state.elapsed_s));
    set('interval-start', ep.t0_desktop_ns == null ? '尚未共同开始' : time(Number(ep.t0_desktop_ns) / 1e9));
    set('interval-end', ep.t1_desktop_ns == null ? state.active ? '尚未停止' : '尚无' : time(Number(ep.t1_desktop_ns) / 1e9));
    const blockers = [];
    if (state.operation_busy) blockers.push('操作正在执行，请等待完成，不要重复点击。');
    else if (state.active) blockers.push('已有会话：需要结束时点击“停止并保存”。清理未完成时，恢复连接后再次停止。');
    else for (const id of ['p450', 'unitree']) {
      const s = state.hosts?.[id]?.status;
      if (s?.stale !== false) blockers.push(`${id} 状态未确认：检查电源、网络及 SSH 连接。`);
      else if (s.active || s.episode_id) blockers.push(`${id} 有遗留会话，请先“核对遗留会话”，再停止归档。`);
      else if (id === 'unitree' && (state.hosts[id].session_error || Object.values(state.hosts[id].camera_health || {}).some(c => c.ready === false))) blockers.push('Unitree 相机未就绪：检查 USB 后点击“初始化设备”。');
    }
    set('start-hint', blockers.length ? blockers.join(' ') : '两端通信正常且空闲。首次使用请先初始化并检查定位 / 手柄；任务名可留空。');
    $('start-hint').hidden = blockers.length === 0;
    $('session-subtitle').hidden = recording || (!state.active && job?.state !== 'running');
    const pending = state.postprocess || {};
    set('postprocess-hint', pending.pending_count ? `待校验 ${pending.pending_count} 条。设备空闲时点击“校验待处理数据”，每次处理一条；不会导出视频。` : '暂无待校验数据。停止后可继续下一条，校验可稍后进行。');
    $('postprocess-hint').hidden = true;
    $('finalize').textContent = `校验待处理数据${pending.pending_count ? '（' + pending.pending_count + '）' : ''}`;
    set('remaining', `单条上限 30 分钟${state.remaining_s === null || state.remaining_s === undefined ? '' : ' · 剩余约 ' + duration(state.remaining_s)} · 不自动停止`);
    host('p450', state.hosts?.p450, state.disk_warning_bytes, recording); host('unitree', state.hosts?.unitree, state.disk_warning_bytes, recording);
    const clock = state.clock || {}; badge('clock-badge', !online ? '状态过期' : clock.degraded === false ? '探测正常' : '异常 / 未确认', !online ? 'amber' : clock.degraded === false ? 'green' : 'amber');
    set('clock-error', !online || clock.degraded !== false || clock.estimated_error_ms == null ? '未确认' : `${Number(clock.estimated_error_ms).toFixed(2)} ms · 估计`);
    const clockAge = id => { const age = clock.health?.[id]?.sample_age_ns; return age !== null && age !== undefined && Number.isFinite(Number(age)) ? `${(Number(age) / 1e9).toFixed(1)} s` : '未知'; };
    set('clock-time', !online ? '状态过期' : clock.health ? `P450 ${clockAge('p450')} · Unitree ${clockAge('unitree')}` : !state.active && ep.episode_id ? '采集已结束' : '未知');
    const reasons = $('clock-reasons'); reasons.replaceChildren(); (clock.reasons || []).forEach(x => reasons.append(node('div', '', String(x))));
    for (const id of ['p450', 'unitree']) {
      const rtt = clock.health?.[id]?.minimum_rtt_ns;
      if (rtt != null) reasons.append(node('div', '', `${id} 时钟探测最小 RTT：${(Number(rtt) / 1e6).toFixed(2)} ms`));
    }
    const validation = state.validation || {}; const vstate = validation.state;
    badge('validation-badge', ({passed:'验收通过',failed:'未通过',running:'校验中',pending:'待后处理'})[vstate] || '未验收', vstate === 'passed' ? 'green' : vstate === 'failed' || vstate === 'pending' ? 'amber' : 'neutral');
    const body = $('validation-body'); body.replaceChildren();
    const report = validation.report;
    if (!report) {
      body.append(node('p', 'muted', vstate === 'running' ? '正在生成验收报告。' : vstate === 'failed' ? '验收失败；原始数据仍保留。' : vstate === 'pending' ? '两端录制已停，原始数据和对时记录已保存；设备空闲时点击“校验待处理数据”。' : '尚无验收报告。'));
      if (vstate === 'failed' && validation.error) body.append(node('div', 'error-note', String(validation.error).slice(-1200)));
    }
    else { row(body, '有效区间', report.valid_interval ? intervalDuration(report.valid_interval) : '无有效区间');
      if (report.valid_interval) { const exact = node('details', 'interval-detail'); exact.append(node('summary', '', '查看精确纳秒边界')); const bounds = node('div', 'interval-bounds'); row(bounds, '起点（含）', String(report.valid_interval.start_inclusive_ns ?? '未知')); row(bounds, '终点（不含）', String(report.valid_interval.end_exclusive_ns ?? '未知')); exact.append(bounds); body.append(exact); }
      const quality = report.quality || {}; row(body, '数据校验', quality.data_validated === true ? '已校验' : '未通过');
      const warnings = [...(quality.degradation_reasons || []), ...(report.warnings || [])]; warnings.forEach(x => body.append(node('div', 'error-note', String(x))));
    }
    set('report-path', validation.path || '尚未取得'); $('report-path').dataset.value = validation.path || '';
    document.querySelector('[data-copy="report-path"]').disabled = !validation.path;
    badge('job-badge', job ? (phases[job.phase] || phases[job.state] || job.state) : '无任务', job?.state === 'succeeded' ? 'green' : job?.state === 'failed' ? 'amber' : 'neutral');
    set('job-summary', job ? `${labels[job.action] || job.action} · ${phases[job.phase] || phases[job.state] || job.state} · ${time(job.started_at)}${job.error ? ' · ' + job.error : ''}` : '尚无操作任务。');
    set('job-log', job?.log || '尚无日志');
    if (job && submittedJobId === job.id && job.state !== 'running') {
      set('action-feedback', `${labels[job.action] || job.action}任务${job.state === 'succeeded' ? '已完成' : job.state === 'failed' ? '失败' : '中断'}${job.error ? '：' + job.error : '；请查看最终状态与验收结果。'}`);
      submittedJobId = null;
    }
    actions.forEach(action => { $(action).disabled = !online || submitting || state.allowed?.[action] !== true; });
    $('instruction').disabled = submitting || state.active || state.operation_busy;
  }
  function connection(ok, message) {
    online = ok; $('browser-light').className = `light ${ok ? 'green' : 'amber'}`;
    set('browser-state', message); set('sample-time', ok ? `最近读取 ${new Date().toLocaleTimeString('zh-CN', {hour12:false})}` : lastSuccess ? `最后成功读取 ${lastSuccess.toLocaleTimeString('zh-CN', {hour12:false})}` : '尚无有效数据');
    if (lastState) render(lastState); else actions.forEach(a => $(a).disabled = true);
  }
  async function requestJson(url, options, timeoutMs) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, {...options, signal: controller.signal});
      return {response, data: await response.json()};
    } finally { clearTimeout(timer); }
  }
  async function poll() {
    if (polling) return;
    polling = true;
    try { const {response, data: state} = await requestJson('/api/state', {cache:'no-store'}, 4000); if (!response.ok) throw new Error(`HTTP ${response.status}`); lastSuccess = new Date(); connection(true, '本机服务已连接'); render(state); }
    catch (error) { connection(false, '本机服务离线 · 状态未更新'); }
    finally { polling = false; }
  }
  async function session() { try { const {response, data} = await requestJson('/api/session', {cache:'no-store'}, 4000); if (!response.ok) throw new Error(); token = data.token; } catch { token = null; } }
  async function submit(action) {
    if (!online || submitting || lastState?.allowed?.[action] !== true) return;
    const payload = action === 'start' ? {instruction:$('instruction').value.trim()} : {};
    submitting = true; render(lastState); set('action-feedback', '正在提交操作…');
    let posted = false;
    try { if (!token) await session(); if (!token) throw new Error('无法获取本机操作令牌，请刷新页面。');
      posted = true;
      const {response, data: result} = await requestJson('/api/actions', {method:'POST',headers:{'Content-Type':'application/json','X-Console-Token':token},body:JSON.stringify({action,payload})}, 8000);
      posted = false;
      if (response.status === 403) token = null;
      if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
      submittedJobId = result.job?.id || null;
      set('action-feedback', `${labels[action]}任务已受理；等待设备和任务状态更新。`); await poll();
    } catch (error) {
      if (posted) { connection(false, '操作结果未确认 · 请核对状态'); set('action-feedback', '操作请求超时或连接中断，结果未知。请等待状态刷新并核对任务记录，确认后再手动重试；不会自动重发。'); poll(); }
      else set('action-feedback', `提交失败：${error.message}`);
    }
    finally { submitting = false; if (lastState) render(lastState); }
  }
  $('capture-form').addEventListener('submit', e => { e.preventDefault(); submit('start'); });
  try { $('instruction').value = localStorage.getItem('joint-instruction') || ''; } catch {}
  $('instruction').addEventListener('input', () => { try { localStorage.setItem('joint-instruction', $('instruction').value); } catch {} });
  for (const action of actions.slice(1)) $(action).addEventListener('click', () => submit(action));
  document.addEventListener('click', async e => { const button = e.target.closest('button.copy'); if (!button || button.disabled) return; const value = button.dataset.copy ? $(button.dataset.copy).dataset.value : button.dataset.value; if (!value) return;
    try { await navigator.clipboard.writeText(value); button.textContent = '已复制'; } catch { button.textContent = '复制失败'; } setTimeout(() => { button.textContent = '复制'; }, 2000);
  });
  window.addEventListener('offline', () => connection(false, '浏览器离线 · 状态未更新'));
  window.addEventListener('online', poll);
  window.JointConsole = {render};
  session(); poll(); setInterval(poll, 1000);
})();
