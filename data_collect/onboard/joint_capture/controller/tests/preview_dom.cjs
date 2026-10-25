const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const elements = {};
for (const id of ['preview-toggle','preview-image','preview-placeholder','preview-state']) {
  elements[id] = {hidden:true, disabled:true, textContent:'', handlers:{},
    addEventListener(name, fn) {this.handlers[name]=fn;},
    removeAttribute(name) {delete this[name];}};
}
const events = {}, windowEvents = {}, timers = new Map(), intervals = [];
let time = 100, timerId = 0;
const sockets = [];
class Socket {
  static OPEN = 1;
  constructor(url) {this.url=url;this.readyState=0;this.sent=[];sockets.push(this);}
  send(data) {this.sent.push(JSON.parse(data));}
  close() {this.readyState=3;this.onclose?.();}
  open() {this.readyState=1;this.onopen();}
  frame(secs, data='aGVsbG8=') {this.onmessage({data:JSON.stringify({op:'publish',
    topic:'/uav1/camera/color/image_raw/compressed', msg:{header:{stamp:{secs,nsecs:0}},format:'jpeg',data}})});}
}
const document = {hidden:false, getElementById:id=>elements[id],
  addEventListener:(event,fn)=>events[event]=fn};
vm.runInNewContext(fs.readFileSync(process.argv[2],'utf8'), {
  document, window:{addEventListener:(event,fn)=>windowEvents[event]=fn},
  WebSocket:Socket, performance:{now:()=>time},
  setTimeout:(fn,delay)=>{timers.set(++timerId,{fn,delay});return timerId;},
  clearTimeout:id=>timers.delete(id), setInterval:fn=>intervals.push(fn),
  fetch:async()=>({ok:true,json:async()=>({enabled:true,url:'ws://192.168.1.11:9090',
    topic:'/uav1/camera/color/image_raw/compressed',fps:5})})
});
setImmediate(()=>{
  const button=elements['preview-toggle'], image=elements['preview-image'];
  assert.equal(button.disabled,false);
  button.handlers.click();
  const first=sockets[0];first.open();
  assert.equal(first.sent[0].op,'subscribe');
  assert.equal(first.sent[0].queue_length,1);
  assert.equal(first.sent[0].throttle_rate,200);
  first.frame(1);assert.equal(image.src,undefined);
  first.frame(2);image.onload();assert.equal(image.hidden,false);
  time+=2100;
  first.frame(2);intervals[0]();
  assert.equal(image.hidden,true);assert.equal(image.src,undefined);
  first.frame(3);image.onload();assert.equal(image.hidden,false);
  first.frame(0);assert.equal(image.hidden,true);
  first.frame(1);image.onload();assert.equal(image.hidden,false);
  document.hidden=true;events.visibilitychange();
  assert.equal(first.sent.at(-1).op,'unsubscribe');assert.equal(image.hidden,true);
  document.hidden=false;events.visibilitychange();
  assert.equal(sockets.length,2);
  const second=sockets[1];second.open();
  second.frame(10);second.frame(11);image.onload();
  button.handlers.click();
  assert.equal(image.hidden,true);
  assert.equal(second.sent.at(-1).op,'unsubscribe');
  first.frame(100);assert.equal(image.src,undefined);
  assert.ok(sockets.every(s=>s.sent.every(p=>['subscribe','unsubscribe'].includes(p.op))));
  assert.equal(timers.size,0);
  console.log('Preview: advancing frames, frozen timestamps, restart, pause and unsubscribe passed');
});
