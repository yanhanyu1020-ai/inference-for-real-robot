// Minimal DOM boundary; executes the real renderer and checks visible text.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
class Element {
  constructor() { this.children=[]; this.style={}; this.dataset={}; this.own=''; }
  set textContent(v) { this.own=String(v); this.children=[]; }
  get textContent() { return this.own+this.children.map(c=>c.textContent||'').join(' '); }
  append(...c) { this.children.push(...c); }
  prepend(...c) { this.children.unshift(...c); }
  replaceChildren(...c) { this.own=''; this.children=c; }
  addEventListener() {}
}
const nodes=new Map();
const get=id=>{ if(!nodes.has(id))nodes.set(id,new Element()); return nodes.get(id); };
const document={getElementById:get,createElement:()=>new Element(),querySelector:get,addEventListener(){}};
const state={active:true,episode:{episode_id:'test',state:'recording',t0_desktop_ns:'1780000000000000000'},hosts:{},allowed:{stop:true}};
for(const id of ['p450','unitree']) state.hosts[id]={status:{stale:false,active:true,reachable:true,episode_id:'test'}};
const stream={received:300,written:299,durable:295,received_fps:30,written_fps:29.9,
  pending_bytes:48*1024**2,capacity_bytes:64*1024**2,write_bytes_per_s:12*1024**2,rejected:0,write_errors:0};
state.hosts.unitree.raw_capture={format_version:2,stale:false,quality_ok:true,durable_complete:false,
  streams:{front:stream,wrist:stream},remaining_minutes:45,disk_available_bytes:100*1024**3,fault:[]};
const window={addEventListener(){}};
vm.runInNewContext(fs.readFileSync(process.argv[2],'utf8'),{document,window,localStorage:{getItem:()=>null},
  fetch:async url=>({ok:true,json:async()=>url==='/api/state'?state:{token:'test'}}),
  AbortController,setTimeout,clearTimeout,setInterval:()=>0,console});
setImmediate(()=>{
  assert.equal(get('session-title').textContent,'联合采集中');
  let text=get('unitree-content').textContent;
  assert.match(text,/30\.0.*29\.9/);
  assert.match(text,/75%/);
  assert.match(text,/45.*分钟/);
  assert.match(text,/299.*295/);
  state.hosts.unitree.raw_capture.fault=['front writer: disk error'];
  state.hosts.unitree.raw_capture.quality_ok=false;
  window.JointConsole.render(state);
  assert.notEqual(get('session-title').textContent,'联合采集中');
  assert.match(get('unitree-content').textContent,/disk error/);
  state.hosts.unitree.status.stale=true;
  window.JointConsole.render(state);
  assert.match(get('unitree-content').textContent,/过期/);
  state.active=false; state.episode.state='complete';
  for(const id of ['p450','unitree'])state.hosts[id].status={stale:false,active:false};
  state.hosts.unitree.raw_capture.durable_complete=true;
  window.JointConsole.render(state);
  assert.match(get('unitree-content').textContent,/已落盘/);
  assert.notEqual(get('validation-badge').textContent,'验收通过');
  delete state.hosts.unitree.raw_capture;
  window.JointConsole.render(state);
  assert.doesNotMatch(get('unitree-content').textContent,/MCAP/);
  console.log('raw console running/backlog/fault/stopped/disconnected/legacy passed');
});
