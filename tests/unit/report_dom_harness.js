// Minimal independent DOM harness. No browser, network, or rendering engine.
const assert = require("node:assert/strict"), vm = require("node:vm"), fs = require("node:fs");
const payload = JSON.parse(fs.readFileSync(0, "utf8"));
class Element {
  constructor(tag) { this.tagName=tag; this.children=[]; this.style={}; this.attrs={}; this.textContent=""; this.value=""; this.scrollLeft=0; }
  append(...children) { for(const child of children){this.children.push(child);if(this.tagName==="select" && this.children.length===1)this.value=child.value;} }
  replaceChildren() { this.children=[]; if(this.tagName==="select")this.value=""; }
  setAttribute(key,value) { this.attrs[key]=value; }
  click() { this.onclick?.(); }
}
const elements=new Map();
const selectIds=new Set(["run","resource","job","series","speed"]);
const document={getElementById(id){if(!elements.has(id))elements.set(id,new Element(selectIds.has(id)?"select":"div"));return elements.get(id);},createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag),createTextNode:text=>({textContent:text})};
document.getElementById("data").textContent=JSON.stringify(payload.data);
document.getElementById("speed").value="250";
let tick;
const context={document,setInterval:callback=>{tick=callback;return 1;},clearInterval:()=>{tick=null;},setTimeout:()=>{},window:{print(){}},console};
vm.runInNewContext(payload.script, context);
const get=id=>document.getElementById(id);
assert.match(get("position").textContent,/event 4 \/ 4/);
get("first").click();assert.match(get("position").textContent,/event 1 \/ 4 · tick 0/);
get("next").click();assert.match(get("position").textContent,/event 2 \/ 4 · tick 0/);
get("previous").click();assert.match(get("position").textContent,/event 1 \/ 4/);
get("cursor").value=3;get("cursor").oninput();
get("resource").value="machine:M1";get("resource").onchange();
assert.equal(get("events").children.length,2);
get("job").value="B";get("job").onchange();assert.equal(get("events").children.length,0);
get("job").value="A";get("job").onchange();assert.equal(get("events").children.length,2);
get("job").value="";get("job").onchange();
const before=get("gantt").attrs.viewBox;
get("zoomIn").click();assert.notEqual(get("gantt").attrs.viewBox,before);assert.equal(get("gantt").style.width,"2136px");assert.equal(get("zoomLevel").textContent,"2×");
get("zoomIn").click();assert.equal(get("gantt").style.width,"4108px");
get("zoomOut").click();assert.equal(get("gantt").style.width,"2136px");
get("ganttViewport").scrollLeft=100;get("zoomReset").click();assert.equal(get("ganttViewport").scrollLeft,0);assert.equal(get("gantt").attrs.viewBox,before);
get("resource").value="";get("resource").onchange();
get("play").click();assert.equal(get("play").textContent,"Pause");assert.match(get("position").textContent,/event 1 \/ 4/);
tick();assert.match(get("position").textContent,/event 2 \/ 4/);
get("play").click();assert.equal(get("play").textContent,"Play");assert.equal(tick,null);
get("play").click();tick();tick();assert.equal(get("play").textContent,"Play");assert.equal(tick,null);
console.log("offline DOM controls passed");
