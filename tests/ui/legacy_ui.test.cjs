const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');

class Element {
  constructor(tag='div') {
    this.tag=tag; this.children=[]; this.listeners={}; this._value=''; this.textContent='';
    this.disabled=false; this.checked=true; this.hidden=new Set(['hidden']);
    this.classList={add:name=>this.hidden.add(name),remove:name=>this.hidden.delete(name)};
  }
  get value() {
    if (this.tag==='select' && !this._value) return this.children[0]?.children[0]?.value || '';
    return this._value;
  }
  set value(value){this._value=value;}
  set innerHTML(value){throw new Error('Dynamic innerHTML is prohibited');}
  appendChild(child){this.children.push(child);return child;}
  replaceChildren(...children){this.children=children;this._value='';}
  removeAttribute(name){delete this[name];}
  addEventListener(name,callback){this.listeners[name]=callback;}
}
const trips=[
  {trip_id:'00000015',partition:'evaluator_test',points:100,nodes:20,arcs:30,reference_available:true,reference_valid:true},
  {trip_id:'00000000',partition:'student_test',points:100,nodes:20,arcs:30,reference_available:true,reference_valid:true}
];
const flush=()=>new Promise(resolve=>setImmediate(resolve));
function fixture(){
  const elements=new Map();
  const document={getElementById:id=>{
    if(!elements.has(id))elements.set(id,new Element(id==='trip-select'?'select':'div'));
    return elements.get(id);
  },createElement:tag=>new Element(tag),createTextNode:text=>({textContent:text})};
  const pending=[];
  const context=vm.createContext({document,window:{location:{origin:'http://localhost:8000'}},URL,AbortController,setTimeout,clearTimeout,console,
    fetch:async(url)=>{
      if(url.endsWith('/health'))return {ok:true,json:async()=>({model_loaded:true,model_name:'MLP'})};
      if(url.endsWith('/trips'))return {ok:true,json:async()=>({trips})};
      return new Promise(resolve=>pending.push(resolve));
    }});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../../ui/app.js'),'utf8'),context);
  return {context,elements,pending,document};
}
const success=()=>({ok:true,json:async()=>({metrics:{point_accuracy:null,top1_accuracy:null,route_continuity:1,candidate_miss_rate:null,inference_time_ms:4,evaluation_observations:0,evaluation_coverage:0,sequence_length:2},predicted_route_preview:[1,2],map_url:'/map/00000015?partition=evaluator_test'})});

test('teacher trips are present and nullable metrics remain unavailable',async()=>{
  const f=fixture();await flush();
  const selector=f.elements.get('trip-select');
  assert.equal(selector.children[0].label,'evaluator_test (1 trips)');
  const request=f.context.runEvaluation();await flush();f.pending[0](success());await request;
  assert.equal(f.elements.get('val-accuracy').textContent,'N/A');
  assert.match(f.elements.get('sub-accuracy').textContent,/0 aligned labels/);
  assert.equal(f.elements.get('route-tags').children.length,2);
});

test('stale prediction cannot alter the newer request spinner or button',async()=>{
  const f=fixture();await flush();
  const old=f.context.runEvaluation();await flush();
  await f.context.checkHealthAndLoadTrips();
  const current=f.context.runEvaluation();await flush();
  f.pending[0](success());await old;
  assert.equal(f.elements.get('btn-evaluate').disabled,true);
  assert.equal(f.elements.get('eval-spinner').hidden.has('hidden'),false);
  f.pending[1](success());await current;
  assert.equal(f.elements.get('btn-evaluate').disabled,false);
});

test('server error is displayed as text and clears old result',async()=>{
  const f=fixture();await flush();
  const request=f.context.runEvaluation();await flush();
  f.pending[0]({ok:false,status:500,json:async()=>({detail:'<img src=x onerror=alert(1)>'})});
  await request;
  assert.match(f.elements.get('map-placeholder').textContent,/<img src=x/);
  assert.equal(f.elements.get('map-frame').src,undefined);
  assert.equal(f.elements.get('val-accuracy').textContent,'—');
});

test('changing decoder while pending allows a new request',async()=>{
  const f=fixture();await flush();
  f.elements.get('trip-select').value='student_test/00000000';
  f.context.updateMetadata();
  const request=f.context.runEvaluation();await flush();
  f.elements.get('chk-viterbi').listeners.change();
  assert.equal(f.elements.get('btn-evaluate').disabled,false);
  f.pending[0](success());await request;
  assert.equal(f.elements.get('btn-evaluate').disabled,false);
});
