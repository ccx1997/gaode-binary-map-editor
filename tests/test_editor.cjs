const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function editor() {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {value: id === 'roadWidth' ? '9' : '5', checked: id === 'orthogonalLines', style: {}, dataset: {}, classList: {add(){},remove(){},toggle(){}}, addEventListener(){},focus(){}});
    return elements.get(id);
  }
  element('canvas').getContext = () => ({});
  const document = {getElementById: element, querySelectorAll: () => [], activeElement: {tagName: 'BODY'}};
  const context = {document, window: {addEventListener(){}}, Image: class {}, setTimeout(){}, clearTimeout(){}, console, alert(message){throw Error(message);}};
  vm.createContext(context);
  const html = fs.readFileSync('map_converter.py', 'utf8');
  let script = html.split('<script>')[1].split('</script>')[0];
  script = script.replace("  init().catch(err=>{console.error(err);alert('启动失败：'+err.message);});", `
    state.config={width:400,height:300};
    this.editor={state,mapCache,handleEntranceClick,saveEntrance,moveEntrance,entranceRoadPoint,normalizedSpecialPoints,drawScene,paintLayer,pointerDown,pointerMove,pointerUp,screenPoint,setTool,addRect,addLine,hitTest,applyFields,deleteSelected,exportedAnnotations,forceMapPalette,importJson,snapshot,restore,buildPlanningGrid,planningGridFromPixels,nearestPlanningPoint,planAStar,routeSegmentClear,handleRouteClick,invalidateMap,thickenRoadPixels,recolorRoadPixels,thickenRoads,changeRoadColor,normalizedRoadStyle,roadSelectionFromPixels,deleteRoads,eraseRoadPixels,paintOutline,defaultOutlineWidth,normalizedOutlineWidth,selectOutline,changeOutlineWidth,matchOutlineWidth,element:$,document};
  `);
  vm.runInContext(script, context);
  return context.editor;
}

test('road drawing, selection, width editing, undo/redo and deletion use the road layer', () => {
  const e=editor();e.state.tool='road';
  e.addLine({x:10,y:20},{x:110,y:80});
  const road=e.state.annotations.roadLines[0];
  assert.equal(road.y2,80, 'roads allow diagonals by default');
  assert.equal(e.state.annotations.obstacleLines.length,0);
  assert.equal(e.hitTest({x:60,y:50}).type,'road');
  assert.equal(e.element('lineFields').hidden,false);
  assert.equal(e.element('rectFields').hidden,true);
  e.element('selectedLineWidth').value='17';e.applyFields();
  assert.equal(road.widthPx,17);
  e.element('undoBtn').onclick();
  assert.equal(e.state.annotations.roadLines[0].widthPx,9);
  e.element('redoBtn').onclick();
  assert.equal(e.state.annotations.roadLines[0].widthPx,17);
  e.state.selected={type:'road',id:road.id};e.deleteSelected();
  assert.equal(e.state.annotations.roadLines.length,0);
});

test('cropped road coordinates and width survive export', () => {
  const e=editor();e.state.tool='road';e.addLine({x:10,y:50},{x:200,y:50});
  const road=e.exportedAnnotations({x:40,y:30,width:100,height:100}).roadLines[0];
  assert.deepEqual([road.x1,road.y1,road.x2,road.y2,road.widthPx],[0,20,99,20,9]);
});

test('palette export retains 127 and removes display antialiasing', () => {
  const e=editor(), data={data:new Uint8ClampedArray([0,0,0,255,127,127,127,255,255,255,255,255,130,130,130,255])};
  e.forceMapPalette({getImageData:()=>data,putImageData(){}},4,1);
  assert.deepEqual(Array.from(data.data).filter((_,i)=>i%4===0),[0,127,255,127]);
});

test('new and legacy annotation projects import without losing roads', async () => {
  const e=editor();e.state.tool='road';e.addLine({x:10,y:20},{x:110,y:80});
  const data={source:{width:400,height:300},project:{annotations:e.state.annotations}};
  await e.importJson({text:async()=>JSON.stringify(data)});
  assert.equal(e.state.annotations.roadLines.length,1);
  assert.equal(e.state.annotations.roadLines[0].y2,80);
  delete data.project.annotations.roadLines;
  await e.importJson({text:async()=>JSON.stringify(data)});
  assert.equal(e.state.annotations.roadLines.length,0);
});

test('route validation only accepts semantic roads, regardless of display color', () => {
  const e=editor();e.state.config={width:12,height:4};e.element('routeClearance').value='0';
  const data=new Uint8ClampedArray(12*4*4);
  for(let y=0;y<4;y++) for(let x=0;x<12;x++) {
    const i=(y*12+x)*4,v=x<4?0:x<8?127:255;
    data[i]=data[i+1]=data[i+2]=v;data[i+3]=255;
  }
  const context={save(){},restore(){},translate(){},drawImage(){},getImageData:()=>({data}),putImageData(){}};
  e.document.createElement=()=>({getContext:()=>context});
  for(const color of [0,127,192,255]){
    e.state.roadStyle.color=color;e.invalidateMap();
    const grid=e.buildPlanningGrid();
    assert.equal(grid.cellSize,1);
    assert.deepEqual(Array.from(grid.free.slice(0,12)),[0,0,0,0,1,1,1,1,0,0,0,0]);
    assert.equal(grid.usableRoadPixels,16);
  }
});

test('thickening expands roads while preserving black building edges', () => {
  const e=editor(), width=9,height=5,pixels=new Uint8ClampedArray(width*height*4).fill(255);
  for(let y=0;y<height;y++)for(const [x,value] of [[3,127],[4,0]]){
    const i=(y*width+x)*4;pixels[i]=pixels[i+1]=pixels[i+2]=value;
  }
  e.thickenRoadPixels(pixels,width,height,2);
  assert.deepEqual(Array.from(pixels.slice(width*2*4,width*3*4)).filter((_,i)=>i%4===0),[255,255,127,127,0,255,255,255,255]);
});

test('all slider values stay exact, including black and white endpoints', () => {
  const e=editor();
  for(let color=0;color<=255;color++){
    const pixels=new Uint8ClampedArray([0,0,0,255,127,127,127,255,255,255,255,255]);
    e.recolorRoadPixels(pixels,color);
    assert.deepEqual(Array.from(pixels).filter((_,i)=>i%4===0),[0,color,255]);
  }
});

test('button and one slider gesture can be undone and redone independently', () => {
  const e=editor();e.thickenRoads();e.thickenRoads();
  assert.equal(e.state.roadStyle.extraWidthPx,4);
  e.element('roadColor').value='80';e.changeRoadColor();
  e.element('roadColor').value='0';e.changeRoadColor();e.element('roadColor').onchange();
  assert.equal(e.state.undo.length,3);
  assert.equal(e.state.roadStyle.color,0);
  e.element('undoBtn').onclick();assert.equal(e.state.roadStyle.color,127);
  e.element('undoBtn').onclick();assert.equal(e.state.roadStyle.extraWidthPx,2);
  e.element('redoBtn').onclick();assert.equal(e.state.roadStyle.extraWidthPx,4);
  e.element('redoBtn').onclick();assert.equal(e.state.roadStyle.color,0);
});

test('project import restores road style and old projects use defaults', async () => {
  const e=editor(), project={annotations:e.state.annotations,roadStyle:{color:255,extraWidthPx:6}};
  await e.importJson({text:async()=>JSON.stringify({project})});
  assert.equal(e.state.roadStyle.color,255);assert.equal(e.state.roadStyle.extraWidthPx,6);
  delete project.roadStyle;
  await e.importJson({text:async()=>JSON.stringify({project})});
  assert.equal(e.state.roadStyle.color,127);assert.equal(e.state.roadStyle.extraWidthPx,0);
});

test('road selection and deletion preserve black buildings and white pixels', () => {
  const e=editor(),pixels=new Uint8ClampedArray([0,0,0,255,127,127,127,255,127,127,127,255,255,255,255,255]);
  const selection=e.roadSelectionFromPixels(pixels,{x:10,y:20,width:4,height:1});
  assert.equal(selection.area,2);
  assert.equal(JSON.stringify(selection.runs),'[[20,11,12]]');
  assert.equal(e.eraseRoadPixels(pixels),2);
  assert.deepEqual(Array.from(pixels).filter((_,i)=>i%4===0),[0,255,255,255]);
});

test('boxed road deletion is undoable and later road strokes follow it in edit order', () => {
  const e=editor();e.state.roadStyle.color=0;
  e.state.roadSelection={x:10,y:20,width:100,height:50,area:100,runs:[]};
  e.deleteRoads();assert.equal(e.state.annotations.roadErases.length,1);
  const erase=e.state.annotations.roadErases[0];
  assert.equal(e.state.roadSelection,null);
  e.element('undoBtn').onclick();assert.equal(e.state.annotations.roadErases.length,0);
  e.element('redoBtn').onclick();assert.equal(e.state.annotations.roadErases.length,1);
  e.state.tool='road';e.addLine({x:10,y:25},{x:110,y:25});
  assert.ok(e.state.annotations.roadLines[0].drawOrder>erase.drawOrder);
  const clipped=e.exportedAnnotations({x:30,y:30,width:100,height:100}).roadErases[0];
  assert.deepEqual([clipped.x,clipped.y,clipped.width,clipped.height],[0,0,80,40]);
});

test('rectangle outline leaves its interior untouched', () => {
  const e=editor(),width=20,pixels=new Uint8Array(width*20).fill(127);
  const target={fillRect(x,y,w,h){for(let row=y;row<y+h;row++)for(let col=x;col<x+w;col++)pixels[row*width+col]=0;}};
  e.paintOutline(target,{x:2,y:3,width:12,height:10,widthPx:2});
  assert.equal(pixels[3*width+5],0);assert.equal(pixels[8*width+2],0);
  assert.equal(pixels[8*width+8],127);assert.equal(pixels[0],127);
});

test('rectangle creation, size and width editing, crop metadata and JSON round trip', async () => {
  const e=editor();e.element('outlineWidth').value='3';
  e.addRect('outline',{x:10,y:20,width:100,height:60});
  assert.equal(e.state.annotations.outlineRects[0].widthPx,3);
  e.element('rectW').value='140';e.element('selectedOutlineWidth').value='7';e.applyFields();
  const outline=e.state.annotations.outlineRects[0];
  assert.equal(outline.width,140);assert.equal(outline.widthPx,7);
  const clipped=e.exportedAnnotations({x:40,y:30,width:100,height:100}).outlineRects[0];
  assert.deepEqual([clipped.x,clipped.y,clipped.width,clipped.height],[-30,-10,140,60]);
  e.state.roadSelection={x:50,y:40,width:20,height:20,area:10,runs:[]};e.deleteRoads();
  await e.importJson({text:async()=>JSON.stringify({project:{annotations:e.state.annotations}})});
  assert.equal(e.state.annotations.outlineRects[0].widthPx,7);assert.equal(e.state.annotations.roadErases.length,1);
  await e.importJson({text:async()=>JSON.stringify({project:{annotations:{obstacleLines:[{id:'legacy',x1:0,y1:0,x2:10,y2:0,widthPx:3}]}}})});
  assert.equal(e.state.annotations.outlineRects.length,0);assert.equal(e.state.annotations.roadErases.length,0);assert.equal(e.state.annotations.obstacleLines.length,1);
});

function planningFixture(rows,clearance=0,crop=null){
  const e=editor(),width=rows[0].length,height=rows.length,pixels=new Uint8ClampedArray(width*height*4);
  rows.forEach((row,y)=>{assert.equal(row.length,width);[...row].forEach((cell,x)=>{
    const i=(y*width+x)*4,value=cell==='r'?127:cell==='#'?0:255;
    pixels[i]=pixels[i+1]=pixels[i+2]=value;pixels[i+3]=255;
  });});
  const grid=e.planningGridFromPixels(pixels,width,height,clearance,crop);
  const at=(x,y)=>({row:y,col:x,x:x+.5,y:y+.5});
  return {e,pixels,grid,at,width,height};
}

function entranceFixture(){
  const e=editor(),width=100,height=80,pixels=new Uint8ClampedArray(width*height*4).fill(255);
  // A raster road with a black obstacle and a deleted gap.
  for(let x=10;x<90;x++)pixels.fill(x===50?0:x===51?255:127,(30*width+x)*4,(30*width+x)*4+3);
  e.state.config={width,height};
  const context={save(){},restore(){},translate(){},drawImage(){},putImageData(){},getImageData(x,y,w,h){
    const data=new Uint8ClampedArray(w*h*4);
    for(let row=0;row<h;row++)data.set(pixels.subarray(((y+row)*width+x)*4,((y+row)*width+x+w)*4),row*w*4);
    return {data};
  }};
  e.document.createElement=()=>({getContext:()=>context});
  e.element('canvas').getBoundingClientRect=()=>({left:0,top:0});
  e.element('canvas').setPointerCapture=()=>{};
  return e;
}

test('entrances snap to semantic road pixels, respecting distance, obstacles, deletion and crop',()=>{
  const e=entranceFixture();
  for(const color of [0,127,255]){
    e.state.roadStyle.color=color;e.invalidateMap();
    assert.deepEqual({...e.entranceRoadPoint({x:20.5,y:42.5})},{x:20,y:30});
    assert.equal(e.entranceRoadPoint({x:20.5,y:42.51}),null);
    assert.equal(e.entranceRoadPoint({x:50,y:30},0),null);
    assert.equal(e.entranceRoadPoint({x:51,y:30},0),null);
  }
  e.state.cropRect={x:10,y:0,width:20,height:20};e.state.cropPreview=true;
  assert.equal(e.entranceRoadPoint({x:20,y:19}),null);
  assert.equal(e.entranceRoadPoint({x:20,y:30}),null);
});

test('entry creation requires a label and supports safe arbitrary strings, selection, edits and undo',()=>{
  const e=entranceFixture();e.state.tool='entrance';
  e.handleEntranceClick({x:20,y:31});
  assert.equal(e.state.annotations.specialPoints.length,0);
  assert.equal(e.element('entrancePanel').hidden,false);
  e.element('entranceLabel').value='   ';e.saveEntrance();
  assert.equal(e.state.annotations.specialPoints.length,0);
  const label='3栋 <北门> & "入口" 🚪';
  e.element('entranceLabel').value=label;e.saveEntrance();
  const item=e.state.annotations.specialPoints[0];
  assert.equal(item.type,'buildingEntrance');assert.equal(item.label,label);
  assert.deepEqual([item.x,item.y],[20,30]);assert.equal(e.hitTest(item).type,'entrance');
  assert.match(e.element('layerList').innerHTML,/&lt;北门&gt; &amp; &quot;入口&quot;/);
  assert.equal(e.element('selectionFields').hidden,true);
  e.element('entranceLabel').value='东门';e.saveEntrance();
  assert.equal(item.label,'东门');
  e.element('undoBtn').onclick();assert.equal(e.state.annotations.specialPoints[0].label,label);
  e.element('redoBtn').onclick();assert.equal(e.state.annotations.specialPoints[0].label,'东门');
  e.state.selected={type:'entrance',id:item.id};e.deleteSelected();
  assert.equal(e.state.annotations.specialPoints.length,0);
  e.element('undoBtn').onclick();assert.equal(e.state.annotations.specialPoints[0].label,'东门');
});

test('entrance dragging is one undo step, keeps the last road position and works in a rotated view',()=>{
  const e=entranceFixture();e.handleEntranceClick({x:20,y:30});e.element('entranceLabel').value='入口';e.saveEntrance();
  e.state.tool='select';e.state.viewRotation=90;e.state.zoom=2;e.state.panX=10;e.state.panY=20;
  const event=(x,y)=>{const p=e.screenPoint({x,y});return {clientX:p.x,clientY:p.y,button:0,pointerId:1};};
  const count=e.state.undo.length;
  e.pointerDown(event(20,30));e.pointerUp(event(20,30));assert.equal(e.state.undo.length,count);
  e.pointerDown(event(20,30));e.pointerMove(event(30,35));e.pointerMove(event(40,35));
  e.pointerMove(event(40,70));e.pointerUp(event(40,70));
  assert.deepEqual([e.state.annotations.specialPoints[0].x,e.state.annotations.specialPoints[0].y],[40,30]);
  assert.equal(e.state.undo.length,count+1);
  e.element('undoBtn').onclick();assert.equal(e.state.annotations.specialPoints[0].x,20);
  e.element('redoBtn').onclick();assert.equal(e.state.annotations.specialPoints[0].x,40);
});

test('entrance JSON preserves source coordinates and clips only the exported point list',async()=>{
  const e=entranceFixture();
  for(const x of [20,70]){e.handleEntranceClick({x,y:30});e.element('entranceLabel').value=`${x}栋入口`;e.saveEntrance();}
  const project=e.snapshot(),crop={x:20,y:25,width:50,height:20};
  const exported=e.exportedAnnotations(crop).specialPoints;
  assert.equal(exported.length,1);assert.deepEqual([exported[0].x,exported[0].y],[0,5]);
  await e.importJson({text:async()=>JSON.stringify({project,exported:{annotations:{specialPoints:exported}}})});
  assert.deepEqual(Array.from(e.state.annotations.specialPoints,p=>p.x),[20,70]);
  assert.equal(e.state.annotations.specialPoints[0].label,'20栋入口');
  await e.importJson({text:async()=>JSON.stringify({project:{annotations:{}}})});
  assert.equal(e.state.annotations.specialPoints.length,0);
});

test('entrance overlays do not alter map pixels, exported imagery or route validation',()=>{
  const e=entranceFixture();e.handleEntranceClick({x:20,y:30});e.element('entranceLabel').value='入口';e.saveEntrance();
  // A point reaching the raster painter must perform no drawing at all.
  e.paintLayer(new Proxy({}, {get(){throw Error('Point was painted into map pixels');}}),'entrance',e.state.annotations.specialPoints[0]);
  e.state.route={startClick:{x:10,y:30},goalClick:{x:40,y:30},path:[{x:10,y:30},{x:40,y:30}],status:'道路连通'};
  e.element('entranceLabel').value='修改入口';e.saveEntrance();
  assert.equal(e.state.route.path.length,2);
  const draws=[];
  e.drawScene({save(){},restore(){},translate(){},drawImage(surface){draws.push(surface);}},0,0,false);
  assert.equal(draws.length,1);
});

test('entry drafts cancel cleanly and original preview blocks creation, editing and deletion',()=>{
  const e=entranceFixture();e.state.tool='entrance';e.handleEntranceClick({x:20,y:30});
  e.element('cancelEntranceBtn').onclick();assert.equal(e.state.entranceDraft,null);assert.equal(e.state.undo.length,0);
  e.handleEntranceClick({x:20,y:30});e.element('entranceLabel').value='入口';e.saveEntrance();
  e.state.showOriginal=true;
  e.element('entranceLabel').value='不能修改';e.saveEntrance();e.handleEntranceClick({x:70,y:30});e.deleteSelected();
  assert.equal(e.state.annotations.specialPoints.length,1);assert.equal(e.state.annotations.specialPoints[0].label,'入口');
  assert.equal(e.state.entranceDraft,null);
});

test('invalid special-point data is rejected rather than becoming an unusable annotation',()=>{
  const e=editor(),point={id:'valid',type:'buildingEntrance',label:'入口',x:10,y:20};
  for(const patch of [{x:NaN},{y:300},{x:-1},{label:''},{label:{}},{type:'unknown'}]){
    assert.throws(()=>e.normalizedSpecialPoints([{...point,...patch}]),/标签或坐标无效/);
  }
});

test('invalid entrance import leaves the active project and its undo history intact',async()=>{
  const e=entranceFixture();e.handleEntranceClick({x:20,y:30});e.element('entranceLabel').value='已有入口';e.saveEntrance();
  const before=JSON.stringify(e.snapshot()),undoCount=e.state.undo.length;
  await assert.rejects(e.importJson({text:async()=>JSON.stringify({project:{annotations:{specialPoints:[{type:'buildingEntrance',label:42,x:10,y:20}]}}})}),/标签或坐标无效/);
  assert.equal(JSON.stringify(e.snapshot()),before);assert.equal(e.state.undo.length,undoCount);
});

test('one-pixel roads stay connected while a one-pixel deletion breaks the route',()=>{
  const {e,pixels,grid,at,width,height}=planningFixture(['.'.repeat(72),'r'.repeat(72),'.'.repeat(72)]);
  assert.equal(e.planAStar(grid,at(1,1),at(70,1)).length,2);
  e.eraseRoadPixels(pixels.subarray((width+36)*4,(width+37)*4));
  const broken=e.planningGridFromPixels(pixels,width,height);
  assert.equal(e.planAStar(broken,at(1,1),at(70,1)).length,0);
  pixels.fill(127,(width+36)*4,(width+36)*4+3);
  assert.ok(e.planAStar(e.planningGridFromPixels(pixels,width,height),at(1,1),at(70,1)).length);
});

test('route simplification follows a road bend without cutting across white background',()=>{
  const {e,grid,at}=planningFixture(['.........','.rrrrrrr.','.......r.','.......r.','.......r.','.......r.','.........']);
  const path=e.planAStar(grid,at(1,1),at(7,5));
  assert.ok(path.length>=3);
  assert.equal(e.routeSegmentClear(grid,path[0],path.at(-1)),false);
  for(let i=1;i<path.length;i++)assert.ok(e.routeSegmentClear(grid,path[i-1],path[i]));
});

test('black rectangle boundaries interrupt roads and changing clearance removes unsafe road pixels',()=>{
  const {e,pixels,grid,at,width,height}=planningFixture(['..........','rrrrrrrrrr','rrrrrrrrrr','rrrrrrrrrr','..........']);
  assert.ok(e.planAStar(grid,at(0,2),at(9,2)).length);
  const painter={fillRect(x,y,w,h){for(let row=y;row<y+h;row++)for(let col=x;col<x+w;col++)pixels.fill(0,(row*width+col)*4,(row*width+col)*4+3);}};
  e.paintOutline(painter,{x:3,y:0,width:4,height:5,widthPx:1});
  const blocked=e.planningGridFromPixels(pixels,width,height);
  assert.equal(e.planAStar(blocked,at(0,2),at(9,2)).length,0);
  const margin=planningFixture(['#########','rrrrrrrrr','rrrrrrrrr','.........'],1).grid;
  assert.equal(margin.free[9+4],0);assert.equal(margin.free[18+4],1);
});

test('planning respects crop bounds and does not cross diagonal blocked corners',()=>{
  const {e,grid,at}=planningFixture(['rrrrr','rrrrr','rrrrr'],0,{x:1,y:1,width:3,height:1});
  assert.equal(grid.usableRoadPixels,3);
  assert.equal(e.planAStar(grid,at(0,1),at(3,1)).length,0);
  assert.equal(e.planAStar(grid,at(1,1),at(3,1)).length,2);
  const corners=planningFixture(['r#','#r']);
  assert.equal(corners.e.planAStar(corners.grid,corners.at(0,0),corners.at(1,1)).length,0);
});

test('endpoint snapping finds the nearest road and strictly enforces the 40 px radius',()=>{
  const {e,grid}=planningFixture(['.'.repeat(85),'r'+'.'.repeat(39)+'r'+'.'.repeat(44)]);
  const near=e.nearestPlanningPoint({x:39.5,y:1.5},grid);
  assert.equal(near.col,40);assert.equal(near.snapped,true);
  assert.equal(e.nearestPlanningPoint({x:40.5,y:1.5},grid).snapped,false);
  assert.equal(e.nearestPlanningPoint({x:80.5,y:1.5},grid).col,40);
  assert.equal(e.nearestPlanningPoint({x:80.51,y:1.5},grid),null);
  const diagonal=planningFixture(['r...','....','....','....']);
  assert.equal(diagonal.e.nearestPlanningPoint({x:3.5,y:3.5},diagonal.grid,4),null);
});

test('display color changes preserve a route but geometry edits invalidate it',()=>{
  const e=editor();e.state.route={startClick:{x:1,y:1},goalClick:{x:10,y:1},path:[{x:1,y:1},{x:10,y:1}],status:'道路连通'};
  for(const color of ['0','255']){
    e.element('roadColor').value=color;e.changeRoadColor();e.element('roadColor').onchange();
    assert.equal(e.state.route.path.length,2);
  }
  e.thickenRoads();assert.equal(e.state.route.path.length,0);assert.match(e.state.route.status,/地图已修改/);
});

test('empty roads and excessive clearance give actionable feedback without white-background fallback',()=>{
  const e=editor();e.state.config={width:3,height:3};
  const data=new Uint8ClampedArray(3*3*4).fill(255);
  const context={drawImage(){},getImageData:()=>({data}),putImageData(){}};
  e.document.createElement=()=>({getContext:()=>context});
  e.handleRouteClick({x:1.5,y:1.5});assert.match(e.state.route.status,/没有道路/);
  data.fill(0,0,3);data.fill(127,4,7);e.invalidateMap();e.element('routeClearance').value='1';
  e.handleRouteClick({x:1.5,y:.5});assert.match(e.state.route.status,/减小障碍安全边距/);
});


test('new rectangles use the measured building width without changing explicit old widths',async()=>{
  const e=editor();e.state.config.defaultOutlineWidth=7;e.element('outlineWidth').value='';
  e.addRect('outline',{x:10,y:10,width:100,height:60});
  assert.equal(e.state.annotations.outlineRects[0].widthPx,7);
  const old={project:{annotations:{outlineRects:[{id:'old',name:'旧矩形',x:10,y:10,width:100,height:60,widthPx:3}]}}};
  await e.importJson({text:async()=>JSON.stringify(old)});
  assert.equal(e.state.annotations.outlineRects[0].widthPx,3);
  e.selectOutline({x:20,y:20});e.matchOutlineWidth();
  assert.equal(e.state.annotations.outlineRects[0].widthPx,7);
});

test('rectangle width slider updates only the chosen border, groups undo, and round trips',async()=>{
  const e=editor();e.state.config.defaultOutlineWidth=5;e.element('outlineWidth').value='5';
  e.addRect('outline',{x:10,y:10,width:100,height:60});
  e.addRect('outline',{x:200,y:20,width:80,height:50});
  e.selectOutline({x:20,y:20});
  assert.equal(e.element('outlineWidthPanel').hidden,false);
  assert.equal(e.element('outlineWidthSlider').disabled,false);
  const original=e.state.annotations.outlineRects[0],count=e.state.undo.length;
  e.element('outlineWidthSlider').value='12';e.element('outlineWidthSlider').oninput();
  e.element('outlineWidthSlider').value='18';e.element('outlineWidthSlider').oninput();
  e.element('outlineWidthSlider').onchange();
  assert.equal(original.widthPx,18);assert.equal(e.element('selectedOutlineWidth').value,18);
  assert.deepEqual([original.x,original.y,original.width,original.height],[10,10,100,60]);
  assert.equal(e.state.annotations.outlineRects[1].widthPx,5);
  assert.equal(e.state.undo.length,count+1);
  e.element('undoBtn').onclick();assert.equal(e.state.annotations.outlineRects[0].widthPx,5);
  e.element('redoBtn').onclick();assert.equal(e.state.annotations.outlineRects[0].widthPx,18);
  const saved={project:{annotations:e.exportedAnnotations(null)}};
  await e.importJson({text:async()=>JSON.stringify(saved)});
  assert.equal(e.state.annotations.outlineRects[0].widthPx,18);
  e.selectOutline({x:20,y:20});e.matchOutlineWidth();assert.equal(e.state.annotations.outlineRects[0].widthPx,5);
  e.element('undoBtn').onclick();assert.equal(e.state.annotations.outlineRects[0].widthPx,18);
});

test('rectangle width selection ignores other layers and cannot modify original preview',()=>{
  const e=editor();e.addRect('outline',{x:10,y:10,width:100,height:60});
  e.state.tool='road';e.addLine({x:10,y:20},{x:100,y:20});
  e.selectOutline({x:20,y:20});assert.equal(e.state.selected.type,'outline');
  const before=e.state.annotations.outlineRects[0].widthPx;
  e.state.showOriginal=true;e.changeOutlineWidth(20);
  assert.equal(e.state.annotations.outlineRects[0].widthPx,before);
  e.state.showOriginal=false;e.selectOutline({x:300,y:200});
  assert.equal(e.state.selected,null);assert.equal(e.element('outlineWidthSlider').disabled,true);
  e.changeOutlineWidth(20);assert.equal(e.state.annotations.outlineRects[0].widthPx,before);
});
