"""Small dependency-free browser editor for :mod:`runtime_control.scene_builder`."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser

from .scene_builder import SceneSpec, export_scene_map


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMO_DIR = PACKAGE_ROOT / "eg"
DEFAULT_DOG_XML = DEFAULT_DEMO_DIR / "dog" / "xml" / "dog_terrain.xml"
DEFAULT_DOG_PLAYER = DEFAULT_DEMO_DIR / "play.py"
DEFAULT_DOG_POLICY = DEFAULT_DEMO_DIR / "model_3400.onnx"


def build_scene_editor_html():
    return r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>MuJoCo 场景编辑器</title><style>
:root{--bg:#07101d;--panel:#101c2d;--line:#2b4160;--text:#edf6ff;--blue:#43a2ff;--cyan:#48ddd0}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}
header{height:64px;padding:12px 20px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between}
h1{font-size:20px;margin:0}small,#status{color:#91a8c5}.layout{display:grid;grid-template-columns:220px 1fr 300px;gap:14px;padding:14px;height:calc(100vh - 64px)}
section{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:14px;overflow:auto}h2{font-size:14px;margin:0 0 12px;color:var(--cyan)}
button,select,input{width:100%;border:1px solid var(--line);background:#15263b;color:var(--text);border-radius:7px;padding:8px;margin:4px 0}
input[type=number]{appearance:textfield;-moz-appearance:textfield}input[type=number]::-webkit-inner-spin-button,input[type=number]::-webkit-outer-spin-button{-webkit-appearance:none;margin:0}
button{cursor:pointer}button:hover,.active{border-color:var(--blue);background:#173758}.danger{border-color:#88404d}.primary{background:#14609e;border-color:#2789d7}
label{display:block;margin-top:9px;color:#a9bad0;font-size:12px}.tools{display:grid;grid-template-columns:1fr 1fr;gap:6px}.canvas-wrap{position:relative;padding:0;overflow:hidden}
canvas{width:100%;height:100%;display:block;background:#dce6e9;cursor:crosshair}.hint{position:absolute;left:12px;bottom:10px;background:#07101dcc;padding:7px 10px;border-radius:7px}
#list button{text-align:left}.row{display:grid;grid-template-columns:1fr 1fr;gap:7px}#status{padding:9px 14px;border:1px solid var(--line);border-radius:8px;background:#15263b}.status-ok{color:#55e1a1;border-color:#338566!important}.status-bad{color:#ff7887;border-color:#a74755!important}
</style><header><div><small>RUNTIME CONTROL</small><h1>MuJoCo 可视化地图编辑器</h1></div><span id="status">就绪</span></header>
<div class="layout"><section><h2>已有地形</h2><button id="newScene">＋ 新建地形</button><button id="refreshScenes">刷新列表</button><div id="sceneList"></div><h2 style="margin-top:18px">添加障碍</h2><div id="tools"></div><h2 style="margin-top:18px">对象列表</h2><div id="list"></div></section>
<section class="canvas-wrap"><canvas id="canvas"></canvas><div class="hint">单击添加 · 拖动移动 · Delete 删除 · R 旋转</div></section>
<section><h2>场景与地形</h2><label>场景名称</label><input id="sceneName" value="visual_course">
<div class="row"><div><label>场地长度 X</label><input id="terrainLength" type="number" value="12" step="0.5"></div><div><label>场地宽度 Y</label><input id="terrainWidth" type="number" value="8" step="0.5"></div></div>
<label>基础地形</label><select id="terrainKind"><option>flat</option><option>slope</option><option>stairs</option><option>noise</option></select>
<div class="row"><div><label>最大高度</label><input id="terrainHeight" type="number" value="0.25" step="0.05"></div><div><label>随机种子</label><input id="terrainSeed" type="number" value="42"></div></div>
<h2 style="margin-top:20px">选中障碍</h2><div id="empty">尚未选中障碍</div><div id="properties" hidden>
<label>名称</label><input id="objName"><div class="row"><div><label>X</label><input id="objX" type="number" step="0.1"></div><div><label>Y</label><input id="objY" type="number" step="0.1"></div></div>
<label>旋转（度）</label><input id="objYaw" type="number" step="5"><div id="params"></div>
<div class="tools"><button id="rotate">旋转 15°</button><button id="remove" class="danger">删除</button></div></div>
<h2 style="margin-top:20px">场景操作</h2><button id="export" class="primary">保存/导出当前地形</button><button id="preview" class="primary">用演示机器人在 MuJoCo 展示</button><button id="deleteScene" class="danger">删除当前已保存地形</button><button id="clear">清空障碍</button><button id="exitEditor" class="danger">退出编辑器</button>
<p><small>导出结果保存在启动命令指定的目录，可以直接交给 RuntimeScene。</small></p></section></div>
<script>
const kinds=['platform','wall','stairs','gap','stepping_stones','slalom','ramp','side_slope','speed_bumps','hurdles','narrow_bridge','wave_ground','uneven_stairs','random_blocks','seesaw','rotating_bar'];
const labels={platform:'高台',wall:'高墙',stairs:'楼梯',gap:'沟壑',stepping_stones:'梅花桩',slalom:'绕杆',ramp:'斜坡',side_slope:'侧坡',speed_bumps:'减速带',hurdles:'跨栏',narrow_bridge:'窄桥',wave_ground:'波浪地形',uneven_stairs:'不等高楼梯',random_blocks:'随机碎石',seesaw:'跷跷板',rotating_bar:'旋转杆'};
const defaults={platform:{length:1,width:1.4,height:.3},wall:{length:.15,width:2,height:.8},stairs:{count:5,run:.3,width:1.2,rise:.12},gap:{gap:.35,bank_length:1.5,width:1.5,height:.25},stepping_stones:{count:7,size:.22,spacing:.45,height:.2,stagger:.25},slalom:{count:6,spacing:.8,offset:.5,radius:.04,height:.9},ramp:{length:2,width:1.5,angle:20,thickness:.08},side_slope:{length:2.5,width:1.8,angle:15,thickness:.08},speed_bumps:{count:6,spacing:.45,width:1.5,radius:.07},hurdles:{count:4,spacing:1,width:1.4,height:.28,thickness:.045},narrow_bridge:{length:3,width:.45,height:.35},wave_ground:{length:4,width:1.5,amplitude:.12,waves:3,segments:32},uneven_stairs:{count:7,run:.32,width:1.3,min_rise:.07,max_rise:.17,seed:0},random_blocks:{length:4,width:2,density:3,min_size:.12,max_size:.32,max_height:.18,seed:0},seesaw:{length:2.2,width:.8,thickness:.08,pivot_height:.28,amplitude:18,speed:1},rotating_bar:{length:2.5,height:.35,thickness:.055,speed:1.2}};
const paramSchemas={
 platform:{length:['高台长度','m',.1,20,.05],width:['高台宽度','m',.1,20,.05],height:['高台高度','m',.01,3,.01]},
 wall:{length:['墙体厚度','m',.02,2,.01],width:['墙体宽度','m',.1,20,.05],height:['墙体高度','m',.05,5,.05]},
 stairs:{count:['台阶数量','级',1,50,1],run:['单阶长度','m',.05,2,.01],width:['台阶宽度','m',.1,10,.05],rise:['单阶高度','m',.01,.8,.01]},
 gap:{gap:['沟壑宽度','m',.05,3,.01],bank_length:['单侧平台长度','m',.2,10,.1],width:['通过区域宽度','m',.2,10,.1],height:['平台高度','m',.02,2,.01]},
 stepping_stones:{count:['石桩数量','个',1,100,1],size:['石桩边长','m',.05,1,.01],spacing:['前后间距','m',.1,3,.01],height:['石桩高度','m',.02,2,.01],stagger:['左右错位','m',0,2,.01]},
 slalom:{count:['杆子数量','根',1,50,1],spacing:['前后间距','m',.1,5,.05],offset:['左右偏移','m',.05,3,.05],radius:['杆子半径','m',.01,.3,.005],height:['杆子高度','m',.1,3,.05]},
 ramp:{length:['斜坡长度','m',.2,20,.1],width:['斜坡宽度','m',.2,10,.1],angle:['坡度角','°',1,60,1],thickness:['板厚','m',.02,.5,.01]},
 side_slope:{length:['侧坡长度','m',.2,20,.1],width:['侧坡宽度','m',.2,10,.1],angle:['横向倾角','°',1,45,1],thickness:['板厚','m',.02,.5,.01]},
 speed_bumps:{count:['减速带数量','条',1,50,1],spacing:['前后间距','m',.1,3,.05],width:['减速带宽度','m',.2,10,.1],radius:['凸起半径','m',.01,.5,.01]},
 hurdles:{count:['跨栏数量','组',1,30,1],spacing:['前后间距','m',.2,5,.05],width:['横杆宽度','m',.2,10,.1],height:['横杆高度','m',.03,2,.01],thickness:['横杆粗细','m',.01,.3,.005]},
 narrow_bridge:{length:['桥面长度','m',.2,20,.1],width:['桥面宽度','m',.1,3,.05],height:['桥面高度','m',.02,3,.01]},
 wave_ground:{length:['波浪区域长度','m',.5,20,.1],width:['波浪区域宽度','m',.2,10,.1],amplitude:['波浪振幅','m',.01,1,.01],waves:['波浪数量','个',.5,20,.5],segments:['细分数量','段',4,256,1]},
 uneven_stairs:{count:['台阶数量','级',1,50,1],run:['单阶长度','m',.05,2,.01],width:['台阶宽度','m',.1,10,.05],min_rise:['最小单阶高度','m',.01,.8,.01],max_rise:['最大单阶高度','m',.01,1,.01],seed:['随机种子','',0,99999,1]},
 random_blocks:{length:['碎石区域长度','m',.5,20,.1],width:['碎石区域宽度','m',.5,20,.1],density:['密集程度','块/m²',.1,20,.1],min_size:['最小石块尺寸','m',.03,1,.01],max_size:['最大石块尺寸','m',.04,2,.01],max_height:['最大石块高度','m',.02,1,.01],seed:['随机种子','',0,99999,1]},
 seesaw:{length:['板面长度','m',.3,10,.1],width:['板面宽度','m',.2,5,.05],thickness:['板面厚度','m',.02,.5,.01],pivot_height:['支点高度','m',.05,2,.01],amplitude:['摆动幅度','°',1,45,1],speed:['摆动速度','rad/s',.1,10,.1]},
 rotating_bar:{length:['旋转杆长度','m',.2,10,.1],height:['旋转中心高度','m',.05,2,.01],thickness:['杆子半径','m',.01,.3,.005],speed:['旋转速度','rad/s',.1,10,.1]}
};
const byId=id=>document.getElementById(id),canvas=byId('canvas'),ctx=canvas.getContext('2d'),tools=byId('tools'),list=byId('list'),sceneList=byId('sceneList'),status=byId('status'),terrainKind=byId('terrainKind'),sceneName=byId('sceneName'),empty=byId('empty'),properties=byId('properties'),objName=byId('objName'),objX=byId('objX'),objY=byId('objY'),objYaw=byId('objYaw'),params=byId('params'),rotate=byId('rotate'),remove=byId('remove');
let scene={obstacles:[]},sceneId=null,tool='platform',selected=-1,drag=false;
function val(id){return Number(document.querySelector('#'+id).value)} function terrain(){return {kind:terrainKind.value,length:val('terrainLength'),width:val('terrainWidth'),height:val('terrainHeight'),rows:128,cols:128,seed:val('terrainSeed'),roughness:.18,smoothness:8,stair_count:8}}
function worldToView(x,y){return [canvas.width*(x/val('terrainLength')+.5),canvas.height*(.5-y/val('terrainWidth'))]}
function viewToWorld(x,y){return [(x/canvas.width-.5)*val('terrainLength'),(.5-y/canvas.height)*val('terrainWidth')]}
function resize(){const r=canvas.getBoundingClientRect();canvas.width=Math.max(500,r.width*devicePixelRatio);canvas.height=Math.max(400,r.height*devicePixelRatio);draw()}
function footprint(o){let p=o.params;if(['wall','platform','ramp','side_slope','narrow_bridge','wave_ground','random_blocks','seesaw'].includes(o.kind))return [p.length,p.width];if(['stairs','uneven_stairs'].includes(o.kind))return [p.count*p.run,p.width];if(o.kind==='gap')return [p.bank_length*2+p.gap,p.width];if(o.kind==='stepping_stones')return [p.count*p.spacing,p.size+p.stagger*2];if(['slalom','speed_bumps','hurdles'].includes(o.kind))return [p.count*p.spacing,p.width||p.offset*2+p.radius*2];if(o.kind==='rotating_bar')return [p.length,p.thickness*4];return [1,1]}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);let L=val('terrainLength'),W=val('terrainWidth');let step=1;ctx.strokeStyle='#afc0c5';ctx.lineWidth=1;for(let x=-L/2;x<=L/2;x+=step){let a=worldToView(x,-W/2),b=worldToView(x,W/2);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}for(let y=-W/2;y<=W/2;y+=step){let a=worldToView(-L/2,y),b=worldToView(L/2,y);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}
 ctx.fillStyle='#e64b55aa';let s=worldToView(0,0);ctx.beginPath();ctx.arc(s[0],s[1],Math.max(8,canvas.width/L*.3),0,Math.PI*2);ctx.fill();
 scene.obstacles.forEach((o,i)=>{let c=worldToView(o.x,o.y),f=footprint(o),sx=canvas.width/L,sy=canvas.height/W;ctx.save();ctx.translate(...c);ctx.rotate(-o.yaw);ctx.fillStyle=i===selected?'#1689e7cc':'#365d7ecc';ctx.strokeStyle=i===selected?'#003d70':'#17364c';ctx.lineWidth=i===selected?4:2;if(o.kind==='slalom'||o.kind==='stepping_stones'){ctx.beginPath();ctx.ellipse(0,0,f[0]*sx/2,f[1]*sy/2,0,0,Math.PI*2);ctx.fill();ctx.stroke()}else{ctx.fillRect(-f[0]*sx/2,-f[1]*sy/2,f[0]*sx,f[1]*sy);ctx.strokeRect(-f[0]*sx/2,-f[1]*sy/2,f[0]*sx,f[1]*sy)}ctx.fillStyle='white';ctx.textAlign='center';ctx.font=`${13*devicePixelRatio}px system-ui`;ctx.fillText(labels[o.kind],0,5*devicePixelRatio);ctx.restore()})}
function hit(x,y){for(let i=scene.obstacles.length-1;i>=0;i--){let o=scene.obstacles[i],c=worldToView(o.x,o.y),f=footprint(o);if(Math.abs(x-c[0])<f[0]*canvas.width/val('terrainLength')/2+10&&Math.abs(y-c[1])<f[1]*canvas.height/val('terrainWidth')/2+10)return i}return -1}
function refresh(){document.querySelector('#list').innerHTML=scene.obstacles.map((o,i)=>`<button data-i="${i}" class="${i===selected?'active':''}">${i+1}. ${labels[o.kind]} (${o.x.toFixed(1)}, ${o.y.toFixed(1)})</button>`).join('')||'<small>暂无障碍</small>';document.querySelectorAll('#list button').forEach(b=>b.onclick=()=>select(Number(b.dataset.i)));draw()}
function normalizeParams(o){if(o.kind==='random_blocks'&&o.params.count!==undefined&&o.params.density===undefined){o.params.density=+(o.params.count/Math.max(.01,o.params.length*o.params.width)).toFixed(2);delete o.params.count}return o}
async function refreshLibrary(){try{let data=await(await fetch('/api/scenes')).json();sceneList.innerHTML=data.scenes.map(s=>`<button data-scene="${s.id}" class="${s.id===sceneId?'active':''}">${s.name}<small><br>${s.terrain} · ${s.obstacles} 个障碍</small></button>`).join('')||'<small>暂无已保存地形</small>';sceneList.querySelectorAll('button').forEach(b=>b.onclick=()=>loadScene(b.dataset.scene))}catch(e){status.textContent='读取地形列表失败';status.className='status-bad'}}
async function loadScene(id){try{let response=await fetch('/api/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})}),data=await response.json();if(!response.ok)throw Error(data.error);sceneId=data._scene_id;scene={obstacles:(data.obstacles||[]).map(normalizeParams)};sceneName.value=data.name;terrainKind.value=data.terrain.kind;byId('terrainLength').value=data.terrain.length;byId('terrainWidth').value=data.terrain.width;byId('terrainHeight').value=data.terrain.height;byId('terrainSeed').value=data.terrain.seed;select(-1);refreshLibrary();status.textContent='已载入：'+data.name;status.className='status-ok'}catch(e){alert('载入失败：\n'+e.message)}}
function select(i){selected=i;empty.hidden=i>=0;properties.hidden=i<0;if(i<0){refresh();return}let o=scene.obstacles[i],schema=paramSchemas[o.kind]||{};objName.value=o.name||'';objX.value=o.x;objY.value=o.y;objYaw.value=Math.round(o.yaw*180/Math.PI);params.innerHTML=Object.entries(o.params).map(([k,v])=>{let s=schema[k]||[k,'',-1e6,1e6,'any'];return `<label>${s[0]}${s[1]?'（'+s[1]+'）':''}</label><input data-param="${k}" type="number" min="${s[2]}" max="${s[3]}" step="${s[4]}" value="${v}">`}).join('');document.querySelectorAll('[data-param]').forEach(e=>{e.onchange=()=>{let value=Number(e.value),minimum=Number(e.min),maximum=Number(e.max);if(Number.isFinite(value)&&value>=minimum&&value<=maximum){o.params[e.dataset.param]=value;refresh()}else{alert('请输入 '+minimum+' 到 '+maximum+' 之间的数值');e.value=o.params[e.dataset.param]}};e.onwheel=event=>{event.preventDefault();e.blur()}});refresh()}
tools.innerHTML=kinds.map(k=>`<button data-kind="${k}" class="${k===tool?'active':''}">${labels[k]}<small> ${k}</small></button>`).join('');tools.querySelectorAll('button').forEach(b=>b.onclick=()=>{tool=b.dataset.kind;tools.querySelectorAll('button').forEach(x=>x.classList.toggle('active',x===b))});
canvas.onmousedown=e=>{let r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)*devicePixelRatio,y=(e.clientY-r.top)*devicePixelRatio,i=hit(x,y);if(i>=0){select(i);drag=true}else{let w=viewToWorld(x,y);scene.obstacles.push({kind:tool,x:+w[0].toFixed(2),y:+w[1].toFixed(2),z:0,yaw:0,params:{...defaults[tool]},name:''});select(scene.obstacles.length-1)}};
canvas.onmousemove=e=>{if(!drag||selected<0)return;let r=canvas.getBoundingClientRect(),w=viewToWorld((e.clientX-r.left)*devicePixelRatio,(e.clientY-r.top)*devicePixelRatio),o=scene.obstacles[selected];o.x=+w[0].toFixed(2);o.y=+w[1].toFixed(2);objX.value=o.x;objY.value=o.y;refresh()};onmouseup=()=>drag=false;
function bindObj(id,key,convert=Number){document.querySelector('#'+id).oninput=e=>{if(selected<0)return;scene.obstacles[selected][key]=convert(e.target.value);refresh()}}bindObj('objName','name',String);bindObj('objX','x');bindObj('objY','y');objYaw.oninput=e=>{if(selected>=0){scene.obstacles[selected].yaw=Number(e.target.value)*Math.PI/180;refresh()}};
rotate.onclick=()=>{if(selected>=0){scene.obstacles[selected].yaw+=Math.PI/12;select(selected)}};remove.onclick=()=>{if(selected>=0){scene.obstacles.splice(selected,1);select(-1)}};byId('clear').onclick=()=>{scene.obstacles=[];select(-1)};
document.querySelectorAll('#terrainLength,#terrainWidth,#terrainKind,#terrainHeight,#terrainSeed').forEach(e=>e.oninput=draw);
document.querySelectorAll('input[type=number]').forEach(e=>e.onwheel=event=>{event.preventDefault();e.blur()});
document.onkeydown=e=>{if(['INPUT','SELECT'].includes(e.target.tagName))return;if(e.key==='Delete'&&selected>=0)remove.click();if(e.key.toLowerCase()==='r'&&selected>=0)rotate.click()};
async function saveScene(showMessage=true){status.textContent='正在保存…';status.className='';let payload={_scene_id:sceneId,name:sceneName.value,terrain:terrain(),obstacles:scene.obstacles,spawn_position:[0,0,.45],spawn_quaternion:[1,0,0,0],metadata:{editor:'browser'}};let response=await fetch('/api/export',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),data=await response.json();if(!response.ok)throw Error(data.error);sceneId=data.id;status.textContent='✓ 已保存';status.className='status-ok';await refreshLibrary();if(showMessage)alert('保存成功！\n\nMJCF: '+data.xml+'\n场景配置: '+data.scene+(data.heightfield?'\n高度图: '+data.heightfield:''));return data}
byId('export').onclick=async()=>{try{await saveScene(true)}catch(e){status.textContent='保存失败';status.className='status-bad';alert('保存失败：\n'+e.message)}};
byId('preview').onclick=async()=>{let previewWindow=window.open('about:blank','mujoco_preview');try{await saveScene(false);status.textContent='正在启动演示机器人仿真…';let response=await fetch('/api/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:sceneId})}),data=await response.json();if(!response.ok)throw Error(data.error);status.textContent='✓ MuJoCo 仿真已启动';status.className='status-ok';if(previewWindow)previewWindow.location=data.url;else window.location.href=data.url}catch(e){if(previewWindow)previewWindow.close();status.textContent='仿真启动失败';status.className='status-bad';alert('仿真启动失败：\n'+e.message)}};
byId('deleteScene').onclick=async()=>{if(!sceneId){alert('当前是未保存的新地形。');return}if(!confirm('确定删除当前地形及其 terrain.xml / terrain.png 吗？'))return;try{let response=await fetch('/api/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:sceneId})}),data=await response.json();if(!response.ok)throw Error(data.error);newScene();await refreshLibrary();status.textContent='已删除';status.className='status-ok'}catch(e){alert('删除失败：\n'+e.message)}};
byId('exitEditor').onclick=async()=>{if(!confirm('确定退出地图编辑器吗？正在运行的 MuJoCo 预览也会停止。'))return;try{let response=await fetch('/api/shutdown',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}),data=await response.json();if(!response.ok)throw Error(data.error);document.body.innerHTML='<main style="max-width:560px;margin:15vh auto;padding:32px;color:#edf6ff;background:#101c2d;border:1px solid #2b4160;border-radius:14px;font:16px system-ui"><h1>地图编辑器已退出</h1><p>MuJoCo 预览和本地服务已停止，现在可以关闭此页面。</p></main>'}catch(e){status.textContent='退出失败';status.className='status-bad';alert('退出失败：\n'+e.message)}};
function newScene(){scene={obstacles:[]};sceneId=null;sceneName.value='visual_course_'+new Date().toISOString().slice(11,19).replaceAll(':','');terrainKind.value='flat';select(-1);refreshLibrary()}
byId('newScene').onclick=newScene;byId('refreshScenes').onclick=refreshLibrary;
window.onresize=resize;resize();refresh();refreshLibrary();
</script></html>'''


class SceneEditorServer:
    """Own the local editor HTTP server and its export directory."""

    def __init__(
        self, output_dir="generated/visual_course", host="127.0.0.1", port=0,
        policy_path=DEFAULT_DOG_POLICY, robot_xml=DEFAULT_DOG_XML,
        python_executable=sys.executable, preview_port=8766,
        preview_player=DEFAULT_DOG_PLAYER,
    ):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.policy_path = Path(policy_path).expanduser().resolve()
        self.robot_xml = Path(robot_xml).expanduser().resolve()
        self.preview_player = Path(preview_player).expanduser().resolve()
        self.python_executable = str(python_executable)
        self.preview_port = int(preview_port)
        self.preview_process = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, body, content_type="application/json", status=200):
                if isinstance(body, str):
                    payload = body.encode("utf-8")
                else:
                    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type + "; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self._reply(build_scene_editor_html(), "text/html")
                elif self.path == "/api/scenes":
                    self._reply({"scenes": owner.list_scenes()})
                else:
                    self._reply({"error": "not found"}, status=404)

            def do_POST(self):
                if self.path not in {
                    "/api/export", "/api/load", "/api/delete", "/api/preview",
                    "/api/shutdown",
                }:
                    self._reply({"error": "not found"}, status=404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 2_000_000:
                        raise ValueError("invalid request size")
                    payload = json.loads(self.rfile.read(length))
                    if self.path == "/api/shutdown":
                        self._reply({"shutdown": True})
                        threading.Thread(
                            target=owner.request_shutdown, daemon=True
                        ).start()
                    elif self.path == "/api/load":
                        self._reply(owner.load_scene(payload.get("id")))
                    elif self.path == "/api/delete":
                        owner.delete_scene(payload.get("id"))
                        self._reply({"deleted": payload.get("id")})
                    elif self.path == "/api/preview":
                        self._reply(owner.preview_scene(payload.get("id")))
                    else:
                        scene_id = payload.pop("_scene_id", None)
                        scene = SceneSpec.from_dict(payload)
                        target, scene_id = owner.export_target(scene, scene_id)
                        paths = export_scene_map(scene, target)
                        result = {key: str(value) for key, value in paths.items()}
                        result["id"] = scene_id
                        self._reply(result)
                except Exception as error:
                    self._reply({"error": str(error)}, status=400)

            def log_message(self, *_):
                pass

        try:
            self.server = ThreadingHTTPServer((host, int(port)), Handler)
        except OSError as error:
            if getattr(error, "errno", None) == 98:
                raise OSError(
                    "地图编辑器端口 %s:%s 已被占用；请关闭已有服务或使用 --port 更换端口"
                    % (host, port)
                ) from error
            raise
        self.thread = None

    def _scene_directory(self, scene_id):
        if scene_id in (None, "", "."):
            return self.output_dir
        candidate = (self.output_dir / str(scene_id)).resolve()
        if candidate == self.output_dir or self.output_dir not in candidate.parents:
            raise ValueError("scene id escapes the editor output directory")
        return candidate

    def list_scenes(self):
        if not self.output_dir.exists():
            return []
        documents = []
        root_scene = self.output_dir / "scene.json"
        if root_scene.is_file():
            documents.append((".", root_scene))
        documents.extend(
            (str(path.parent.relative_to(self.output_dir)), path)
            for path in self.output_dir.glob("*/scene.json")
        )
        result = []
        for scene_id, path in documents:
            try:
                scene = json.loads(path.read_text(encoding="utf-8"))
                result.append({
                    "id": scene_id,
                    "name": scene.get("name", scene_id),
                    "terrain": scene.get("terrain", {}).get("kind", "flat"),
                    "obstacles": len(scene.get("obstacles", [])),
                    "modified": path.stat().st_mtime,
                })
            except (OSError, ValueError):
                continue
        return sorted(result, key=lambda item: item["modified"], reverse=True)

    def load_scene(self, scene_id):
        path = self._scene_directory(scene_id) / "scene.json"
        if not path.is_file():
            raise FileNotFoundError("scene does not exist: %s" % scene_id)
        result = json.loads(path.read_text(encoding="utf-8"))
        result["_scene_id"] = scene_id or "."
        return result

    def export_target(self, scene, scene_id=None):
        if scene_id:
            return self._scene_directory(scene_id), scene_id
        scene_id = scene.name
        return self._scene_directory(scene_id), scene_id

    def delete_scene(self, scene_id):
        target = self._scene_directory(scene_id)
        scene_path = target / "scene.json"
        if not scene_path.is_file():
            raise FileNotFoundError("scene does not exist: %s" % scene_id)
        if target == self.output_dir:
            for name in ("scene.json", "terrain.xml", "terrain.png"):
                path = target / name
                if path.is_file():
                    path.unlink()
        else:
            shutil.rmtree(target)

    def preview_scene(self, scene_id):
        terrain_xml = self._scene_directory(scene_id) / "terrain.xml"
        for path, label in (
            (terrain_xml, "terrain XML"), (self.robot_xml, "robot XML"),
            (self.policy_path, "ONNX policy"), (self.preview_player, "preview player"),
        ):
            if not path.is_file():
                raise FileNotFoundError("%s not found: %s" % (label, path))
        if self.preview_process is not None and self.preview_process.poll() is None:
            self.preview_process.terminate()
        command = [
            self.python_executable, str(self.preview_player),
            "--onnx", str(self.policy_path), "--xml", str(self.robot_xml),
            "--custom-map", str(terrain_xml), "--map", "custom", "--gui",
            "--gui-port", str(self.preview_port), "--no-open-browser",
        ]
        self.preview_process = subprocess.Popen(
            command, cwd=str(self.preview_player.parent)
        )
        preview_url = "http://127.0.0.1:%d/" % self.preview_port
        deadline = time.monotonic() + 10.0
        last_error = None
        while time.monotonic() < deadline:
            return_code = self.preview_process.poll()
            if return_code is not None:
                raise RuntimeError(
                    "MuJoCo preview exited during startup (code %d)" % return_code
                )
            try:
                with urlopen(preview_url, timeout=0.25) as response:
                    if response.status == 200:
                        break
            except (OSError, URLError) as error:
                last_error = error
            time.sleep(0.1)
        else:
            self.preview_process.terminate()
            raise RuntimeError(
                "MuJoCo preview did not become ready in 10 seconds: %s" % last_error
            )
        return {
            "url": preview_url,
            "policy": str(self.policy_path), "robot": str(self.robot_xml),
        }

    @property
    def url(self):
        host, port = self.server.server_address[:2]
        return "http://%s:%d/" % (host, port)

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
        return self

    def _stop_preview(self):
        if self.preview_process is not None and self.preview_process.poll() is None:
            self.preview_process.terminate()
            try:
                self.preview_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.preview_process.kill()

    def request_shutdown(self):
        """Stop child preview and ask the HTTP serving loop to exit."""
        self._stop_preview()
        self.server.shutdown()

    def close(self):
        self._stop_preview()
        if self.thread is not None:
            self.server.shutdown()
            self.thread.join(timeout=2)
            self.thread = None
        self.server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="MuJoCo browser scene editor")
    parser.add_argument("--output", default="generated/visual_course")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--policy", default=str(DEFAULT_DOG_POLICY))
    parser.add_argument("--robot-xml", default=str(DEFAULT_DOG_XML))
    parser.add_argument("--preview-player", default=str(DEFAULT_DOG_PLAYER))
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--preview-port", type=int, default=8766)
    args = parser.parse_args(argv)
    editor = SceneEditorServer(
        args.output, args.host, args.port, args.policy, args.robot_xml,
        args.python_executable, args.preview_port, args.preview_player,
    ).start()
    print("MuJoCo scene editor: %s" % editor.url)
    print("Export directory: %s" % editor.output_dir)
    if not args.no_browser:
        webbrowser.open(editor.url)
    try:
        editor.thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        editor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
