#!/usr/bin/env python3
"""Han-Neuro GUI — live FastAPI server.

Serves the compile playground and runs *real* exports through
``export_for_npu`` (no mock data). The ``iree_turbine`` backend runs live in
venv-shark; ``torch_mlir`` returns a clear error until the torch-mlir toolchain
is built (ABI wall).

Run (inside venv-shark):
    python scripts/hanneuro_gui_server.py        # -> http://127.0.0.1:8808
"""
from __future__ import annotations

import time

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from torch_mlir_zoo import export_for_npu
from torch_mlir_zoo.ops import RMSNorm, ScaledDotProductAttention, SwiGLU, TopK

# name -> (factory, example_args) with the same shapes the zoo configs use
MODELS = {
    "attention": (ScaledDotProductAttention, (torch.randn(1, 32, 512),)),
    "rmsnorm": (RMSNorm, (torch.randn(1, 32, 512),)),
    "mlp": (SwiGLU, (torch.randn(1, 32, 512),)),
    "topk": (TopK, (torch.randn(1, 32000),)),
}

app = FastAPI(title="Han-Neuro GUI")


class CompileReq(BaseModel):
    model: str = "attention"
    backend: str = "iree_turbine"
    int8: bool = False


@app.post("/api/compile")
def compile_model(req: CompileReq):
    if req.model not in MODELS:
        return JSONResponse({"error": f"unknown model {req.model!r}"}, status_code=400)
    factory, args = MODELS[req.model]
    t0 = time.perf_counter()
    try:
        r = export_for_npu(
            factory(), args, backend=req.backend,
            quantize="int8" if req.int8 else None,
        )
    except ModuleNotFoundError as e:
        return {"error": str(e), "hint": "이 백엔드 툴체인 미설치 — iree_turbine로 전환하세요."}
    except Exception as e:  # surface export failures without a 500
        return {"error": f"{type(e).__name__}: {e}"}
    ms = (time.perf_counter() - t0) * 1000.0
    return {
        "mlir": r.mlir,
        "ok": r.ok,
        "backend": r.backend,
        "server_side_op_hits": r.summary["server_side_op_hits"],
        "op_counts": r.summary["op_counts"],
        "dtypes": r.summary["dtypes"],
        "n_lines": r.summary["n_lines"],
        "fused": "aten.linear" in r.mlir,
        "ms": round(ms, 1),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Han-Neuro GUI (live)</title>
<style>
  :root{--ui:"Apple SD Gothic Neo","Malgun Gothic",-apple-system,system-ui,sans-serif;--mono:ui-monospace,"SF Mono",Menlo,monospace;
    --bg:#eceff5;--card:#fff;--editor:#f5f7fb;--chrome:#eaeef4;--ink:#14181f;--sub:#414b5a;--muted:#7a8494;
    --line:rgba(17,24,39,.1);--lines:rgba(17,24,39,.16);--blue:#2f7cf6;--blues:#1a5fd6;--bsoft:rgba(47,124,246,.12);
    --acc:linear-gradient(130deg,#3b8bff,#6d5efc);--glow:rgba(78,104,250,.38);--pass:#0a9d70;--psoft:rgba(10,157,112,.13);
    --warn:#e07d00;--wsoft:rgba(224,125,0,.14);--danger:#ec3a49;--dsoft:rgba(236,58,73,.13);--code:#0f1420;--codei:#e7ecf5;}
  @media(prefers-color-scheme:dark){:root{--bg:#0b0e15;--card:#161a23;--editor:#0f131c;--chrome:#161b26;--ink:#f0f3f8;
    --sub:#bcc5d4;--muted:#808b9b;--line:rgba(255,255,255,.09);--lines:rgba(255,255,255,.18);--blue:#62a0ff;--blues:#9cc4ff;
    --bsoft:rgba(59,139,255,.18);--pass:#33d69f;--psoft:rgba(51,214,159,.16);--warn:#ffb84d;--wsoft:rgba(255,184,77,.16);
    --danger:#ff6b78;--dsoft:rgba(255,107,120,.16);--code:#0a0e16;--codei:#dfe6f1;}}
  *{box-sizing:border-box}body{margin:0;font-family:var(--ui);color:var(--ink);background:var(--bg);font-size:14.5px;line-height:1.5;
    -webkit-font-smoothing:antialiased}main{max-width:1180px;margin:0 auto;padding:26px 22px 44px;display:flex;flex-direction:column;gap:16px}
  .bar{display:flex;align-items:center;gap:12px}.g{width:44px;height:44px;border-radius:12px;background:var(--acc);display:grid;
    place-items:center;color:#fff;font-size:22px;box-shadow:0 8px 20px var(--glow)}
  .kick{font-size:11px;font-weight:700;color:var(--muted);letter-spacing:.08em;text-transform:uppercase}
  h1{margin:0;font-size:23px;font-weight:800;letter-spacing:-.03em}.n{background:var(--acc);-webkit-background-clip:text;
    background-clip:text;-webkit-text-fill-color:transparent}.live{margin-left:auto;font-family:var(--mono);font-size:12px;
    font-weight:700;color:var(--pass);background:var(--psoft);padding:6px 12px;border-radius:999px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:0 10px 28px rgba(16,24,40,.09)}
  .tool{padding:15px;display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap}.ctl{display:flex;flex-direction:column;gap:6px}
  .ctl .cl{font-size:10.5px;font-weight:800;color:var(--muted);letter-spacing:.05em;text-transform:uppercase;padding-left:2px}
  .ctl.grow{flex:1 1 200px}select{height:42px;width:100%;border:1px solid var(--lines);border-radius:11px;background:var(--editor);
    color:var(--ink);font-family:var(--ui);font-size:14px;font-weight:600;padding:0 12px;cursor:pointer}
  .seg{display:inline-grid;grid-auto-flow:column;gap:5px;padding:4px;background:var(--editor);border:1px solid var(--lines);
    border-radius:12px;height:42px}.seg button{border:0;background:transparent;color:var(--sub);font-family:var(--ui);font-size:13px;
    font-weight:700;padding:0 14px;border-radius:8px;cursor:pointer;line-height:1.1}.seg button[aria-pressed=true]{background:var(--card);
    color:var(--blues);box-shadow:0 1px 2px rgba(16,24,40,.06)}.seg button .sm{display:block;font-size:9.5px;color:var(--muted);font-family:var(--mono)}
  .i8{display:inline-flex;align-items:center;gap:9px;height:42px;padding:0 14px;border:1px solid var(--lines);border-radius:11px;
    background:var(--editor)}.i8 b{font-size:13px;color:var(--sub)}.sw{position:relative;width:44px;height:26px}.sw input{position:absolute;
    opacity:0;width:100%;height:100%;margin:0;cursor:pointer}.tr{position:absolute;inset:0;border-radius:999px;background:rgba(120,130,145,.3)}
  .kn{position:absolute;top:3px;left:3px;width:20px;height:20px;border-radius:50%;background:#fff;transition:transform .18s}
  .sw input:checked~.tr{background:var(--blue)}.sw input:checked~.kn{transform:translateX(18px)}
  .cta{margin-left:auto;height:42px;padding:0 22px;border:0;border-radius:11px;background:var(--acc);color:#fff;font-family:var(--ui);
    font-size:14.5px;font-weight:800;cursor:pointer;box-shadow:0 10px 24px var(--glow)}.cta:disabled{opacity:.6}
  .grid{display:grid;grid-template-columns:minmax(0,340px) minmax(0,1fr);gap:16px;align-items:start}@media(max-width:860px){.grid{grid-template-columns:1fr}}
  .pad{padding:18px}.ct{font-size:11px;font-weight:800;color:var(--muted);letter-spacing:.06em;text-transform:uppercase;margin-bottom:14px}
  .verd{display:flex;align-items:center;gap:11px;margin-bottom:14px}.vi{width:42px;height:42px;border-radius:11px;display:grid;
    place-items:center;font-size:21px;font-weight:800}.vi.ok{background:var(--psoft);color:var(--pass)}.vi.no{background:var(--dsoft);color:var(--danger)}
  .vt{font-size:17px;font-weight:800}.vs{font-size:12px;color:var(--sub);margin-top:1px}
  .chips{display:flex;flex-wrap:wrap;gap:7px}.chip{font-size:12px;font-weight:700;padding:6px 10px;border-radius:9px;border:1px solid var(--lines);
    background:var(--editor);color:var(--sub);font-variant-numeric:tabular-nums}.chip.p{background:var(--psoft);color:var(--pass);border-color:transparent}
  .chip.w{background:var(--wsoft);color:var(--warn);border-color:transparent}.chip .k{color:var(--muted);font-weight:600}.chip .m{font-family:var(--mono)}
  .ops{margin-top:18px;display:flex;flex-direction:column;gap:8px}.ops .l{font-size:11px;font-weight:800;color:var(--muted);letter-spacing:.05em;text-transform:uppercase}
  .op{display:grid;grid-template-columns:1fr 30px;gap:4px 10px;align-items:center}.op .nm{grid-column:1/-1;font-family:var(--mono);font-size:11px;color:var(--sub);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:-3px}
  .op .b{height:8px;border-radius:999px;background:var(--line);overflow:hidden}.op .b>i{display:block;height:100%;background:var(--acc);border-radius:999px}
  .op .v{font-size:12px;font-weight:700;color:var(--sub);text-align:right;font-variant-numeric:tabular-nums}
  .ed{overflow:hidden}.eh{display:flex;align-items:center;justify-content:space-between;padding:12px 14px;background:var(--chrome);border-bottom:1px solid var(--line)}
  .fn{font-family:var(--mono);font-size:13px;font-weight:600;color:var(--ink);display:inline-flex;align-items:center;gap:8px}
  .fn .d{width:8px;height:8px;border-radius:50%;background:var(--blue);box-shadow:0 0 0 3px var(--bsoft)}.gh{height:34px;padding:0 12px;
    border:1px solid var(--lines);border-radius:9px;background:var(--editor);color:var(--sub);font-family:var(--ui);font-size:12.5px;font-weight:700;cursor:pointer}
  pre{margin:0;padding:16px 18px;background:var(--code);color:var(--codei);font-family:var(--mono);font-size:13px;line-height:1.7;overflow:auto;max-height:460px;white-space:pre}
  .empty{padding:40px 18px;text-align:center;color:var(--muted);font-size:13px}.err{color:var(--danger);font-weight:700}
  footer{text-align:center;color:var(--muted);font-size:12px;padding-top:4px}footer .m{font-family:var(--mono)}
</style></head><body><main>
  <div class="bar"><div class="g">🧠</div><div><div class="kick">Torch-MLIR Model Zoo · Live Compiler</div>
    <h1><span class="n">Han-Neuro</span> GUI</h1></div><span class="live" id="live">● live · export_for_npu</span></div>
  <div class="card tool">
    <div class="ctl grow"><span class="cl">모델</span><select id="model">
      <option value="attention">attention — SDPA [1,32,512]</option><option value="rmsnorm">rmsnorm [1,32,512]</option>
      <option value="mlp">mlp — SwiGLU [1,32,512]</option><option value="topk">topk [1,32000]</option></select></div>
    <div class="ctl"><span class="cl">백엔드</span><div class="seg" id="be" role="group">
      <button data-be="iree_turbine" aria-pressed="true">iree_turbine<span class="sm">live</span></button>
      <button data-be="torch_mlir" aria-pressed="false">torch_mlir<span class="sm">join</span></button></div></div>
    <div class="ctl"><span class="cl">양자화</span><label class="i8"><b>INT8</b>
      <span class="sw"><input type="checkbox" id="i8"><span class="tr"></span><span class="kn"></span></span></label></div>
    <button class="cta" id="run">컴파일 ▶</button></div>
  <div class="grid">
    <aside class="card pad"><div class="ct">계약 검증</div><div id="contract"><div class="empty">컴파일을 눌러 실제 export를 실행하세요.</div></div></aside>
    <section class="card ed"><div class="eh"><span class="fn"><span class="d"></span><span id="fname">—.mlir</span></span>
      <button class="gh" id="copy">복사</button></div><pre id="mlir">// export 결과가 여기 표시됩니다.</pre></section>
  </div>
  <footer><span class="m">han-neuro</span> · 실동작 서버 · iree_turbine=라이브 · torch_mlir=툴체인 필요</footer>
</main><script>
  var S={model:"attention",backend:"iree_turbine",int8:false},$=function(i){return document.getElementById(i)};
  $("model").onchange=function(e){S.model=e.target.value};$("i8").onchange=function(e){S.int8=e.target.checked};
  [].forEach.call(document.querySelectorAll("#be button"),function(b){b.onclick=function(){S.backend=b.dataset.be;
    [].forEach.call(document.querySelectorAll("#be button"),function(x){x.setAttribute("aria-pressed",x===b)})}});
  function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}
  function render(d){
    $("fname").textContent=S.model+"."+(S.backend==="torch_mlir"?"torch":"linalg")+".mlir";
    if(d.error){$("contract").innerHTML='<div class="err">'+esc(d.error)+'</div>'+(d.hint?'<div class="vs" style="margin-top:8px">'+esc(d.hint)+'</div>':'');
      $("mlir").textContent="// "+d.error;return;}
    var ok=d.ok,c=[];
    c.push('<div class="verd"><div class="vi '+(ok?"ok":"no")+'">'+(ok?"✓":"!")+'</div><div><div class="vt">'+(ok?"On-device 적합":"서버측 op 검출")+
      '</div><div class="vs">'+(ok?"server_side_op_hits = {}":"paged/KV-cache op 발견")+'</div></div></div>');
    c.push('<div class="chips">');
    c.push('<span class="chip '+(ok?"p":"w")+'"><span class="k">server_side</span> <span class="m">'+(ok?"{}":JSON.stringify(d.server_side_op_hits))+'</span></span>');
    c.push(d.fused?'<span class="chip p">fused <span class="m">aten.linear</span></span>':'<span class="chip w">분해 → mm·bmm</span>');
    c.push('<span class="chip"><span class="k">lines</span> '+d.n_lines+'</span>');
    c.push('<span class="chip"><span class="k">dtypes</span> <span class="m">'+esc(Object.keys(d.dtypes||{}).join(", ")||"—")+'</span></span>');
    c.push('<span class="chip"><span class="k">'+d.ms+'ms</span></span>');
    if(S.int8)c.push('<span class="chip p">INT8</span>');
    c.push('</div>');
    var ops=Object.keys(d.op_counts||{}).map(function(k){return[k,d.op_counts[k]]}).sort(function(a,b){return b[1]-a[1]}).slice(0,8);
    if(ops.length){var mx=ops[0][1];c.push('<div class="ops"><span class="l">op 분포 (실측)</span>');
      ops.forEach(function(o){c.push('<div class="op"><span class="nm">'+esc(o[0])+'</span><span class="b"><i style="width:'+Math.round(o[1]/mx*100)+'%"></i></span><span class="v">'+o[1]+'</span></div>')});
      c.push('</div>');}
    $("contract").innerHTML=c.join("");$("mlir").textContent=d.mlir;
  }
  $("run").onclick=function(){var b=$("run");b.disabled=true;b.textContent="컴파일 중…";$("live").textContent="● 실행 중";
    fetch("/api/compile",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(S)})
    .then(function(r){return r.json()}).then(function(d){render(d);}).catch(function(e){$("contract").innerHTML='<div class="err">요청 실패: '+e+'</div>';})
    .finally(function(){b.disabled=false;b.textContent="컴파일 ▶";$("live").textContent="● live · export_for_npu";});};
  $("copy").onclick=function(){if(navigator.clipboard)navigator.clipboard.writeText($("mlir").textContent)};
</script></body></html>"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8808)
