#!/usr/bin/env python3
"""Sand Lab MPM rig: validate WGSL, run column collapse headless, assert physics."""
import re, struct, math, random, array, sys
import wgpu

HTML = "/home/claude/rainbow/sandlab.html"
src = open(HTML).read()
common = re.search(r"const WGSL_COMMON = /\* wgsl \*/`(.*?)`;", src, re.S).group(1)
def grab(name):
    m = re.search(r"const " + name + r"\s*=\s*(WGSL_COMMON \+ )?/\* wgsl \*/`(.*?)`;", src, re.S)
    return (common + m.group(2)) if m.group(1) else m.group(2)

adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync()
mods = {}
for name in ["WGSL_CLEAR", "WGSL_P2G", "WGSL_GRID", "WGSL_G2P", "WGSL_RENDER"]:
    try:
        mods[name] = device.create_shader_module(code=grab(name))
        print(f"VALIDATE {name}: OK")
    except Exception as e:
        print(f"VALIDATE {name}: FAIL\n{e}"); sys.exit(1)

# also validate the full render pipeline (vertex-buffer layout) headlessly
try:
    device.create_render_pipeline(
        layout="auto",
        vertex={"module": mods["WGSL_RENDER"], "entry_point": "vs", "buffers": [{
            "array_stride": 64, "step_mode": wgpu.VertexStepMode.instance,
            "attributes": [
                {"shader_location": 0, "offset": 0, "format": wgpu.VertexFormat.float32x2},
                {"shader_location": 1, "offset": 8, "format": wgpu.VertexFormat.float32x2},
                {"shader_location": 2, "offset": 48, "format": wgpu.VertexFormat.float32}]}]},
        fragment={"module": mods["WGSL_RENDER"], "entry_point": "fs",
                  "targets": [{"format": wgpu.TextureFormat.bgra8unorm}]},
        primitive={"topology": wgpu.PrimitiveTopology.triangle_list})
    print("VALIDATE render pipeline (vertex-buffer layout): OK")
except Exception as e:
    print("VALIDATE render pipeline: FAIL", e); sys.exit(1)

def _const(name):
    return float(re.search(r"const " + name + r" = ([0-9.e+-]+)", src).group(1))
NG = int(_const("NG")); DT = _const("DT"); GRAV = _const("GRAV")
print(f"shipped config: NG={NG} DT={DT} GRAV={GRAV}")
E, NU = _const("E_YOUNG"), 0.3
MU = E / (2*(1+NU)); LA = E*NU/((1+NU)*(1-2*NU))
PS = 16; SUB = 12

clear_p = device.create_compute_pipeline(layout="auto", compute={"module": mods["WGSL_CLEAR"], "entry_point": "main"})
p2g_p = device.create_compute_pipeline(layout="auto", compute={"module": mods["WGSL_P2G"], "entry_point": "main"})
grid_p = device.create_compute_pipeline(layout="auto", compute={"module": mods["WGSL_GRID"], "entry_point": "main"})
g2p_p = device.create_compute_pipeline(layout="auto", compute={"module": mods["WGSL_G2P"], "entry_point": "main"})

RHO_REST0 = (1/NG/0.006)**2
def alpha_from_phi(deg):
    s = math.sin(math.radians(deg))
    return math.sqrt(2/3) * 2 * s / (3 - s)

def run_collapse(phi_deg, frames=250):
    # tall column, jittered
    random.seed(11)
    pts = []
    y = 0.04
    while y < 0.66:
        x = 0.44
        while x < 0.56:
            pts.append((x + (random.random()-0.5)*0.004, y + (random.random()-0.5)*0.004))
            x += 0.006
        y += 0.006
    N = len(pts)
    pdata = array.array("f", [0.0]*(N*PS))
    for i,(x,yy) in enumerate(pts):
        b = i*PS; pdata[b]=x; pdata[b+1]=yy; pdata[b+8]=1.0; pdata[b+11]=1.0; pdata[b+14]=RHO_REST0
    pbuf = device.create_buffer(size=N*PS*4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
    device.queue.write_buffer(pbuf, 0, pdata.tobytes())
    ga = device.create_buffer(size=NG*NG*3*4, usage=wgpu.BufferUsage.STORAGE)
    gv = device.create_buffer(size=NG*NG*16, usage=wgpu.BufferUsage.STORAGE)
    uni = device.create_buffer(size=80, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
    pvol = 0.5; pmass = 1.0   # mass-normalized units (see sandlab.html)
    rho_rest = (1/NG/0.006)**2
    device.queue.write_buffer(uni, 0, struct.pack("20f",
        DT, NG, 1/NG, GRAV,  MU, LA, alpha_from_phi(phi_deg), pmass,
        0,0,0,0,  0, float(N), NG, pvol,  rho_rest, 0.0,0,0))
    def bg(pipe, bufs):
        return device.create_bind_group(layout=pipe.get_bind_group_layout(0),
            entries=[{"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(bufs)])
    bgs = { 'clear': bg(clear_p, [ga]), 'p2g': bg(p2g_p, [pbuf, ga, uni]),
            'grid': bg(grid_p, [ga, gv, uni]), 'g2p': bg(g2p_p, [pbuf, gv, uni]) }
    early = None
    for f in range(frames):
        enc = device.create_command_encoder()
        for s in range(SUB):
            for pipe, key, wg in [(clear_p,'clear',(NG*NG*3+255)//256), (p2g_p,'p2g',(N+63)//64),
                                  (grid_p,'grid',(NG*NG+63)//64), (g2p_p,'g2p',(N+63)//64)]:
                cp = enc.begin_compute_pass(); cp.set_pipeline(pipe); cp.set_bind_group(0, bgs[key])
                cp.dispatch_workgroups(wg); cp.end()
        device.queue.submit([enc.finish()])
        if f == 8:
            eo = array.array("f"); eo.frombytes(bytes(device.queue.read_buffer(pbuf)))
            xs8 = sorted(eo[i*PS] for i in range(N))
            early = xs8[int(N*0.99)] - xs8[int(N*0.01)]
    out = array.array("f"); out.frombytes(bytes(device.queue.read_buffer(pbuf)))
    xs = sorted(out[i*PS] for i in range(N))
    ys = sorted(out[i*PS+1] for i in range(N))
    vs = [math.hypot(out[i*PS+2], out[i*PS+3]) for i in range(N)]
    qs = [out[i*PS+12] for i in range(N)]
    bad = sum(1 for i in range(N) if not (math.isfinite(out[i*PS]) and math.isfinite(out[i*PS+1])))
    ke = sum(v*v for v in vs)/N
    p95h = ys[int(N*0.95)]
    runout = xs[int(N*0.99)] - xs[int(N*0.01)]   # robust extent
    return {"N":N, "bad":bad, "runout":runout, "height":p95h, "ke":ke, "early":early,
            "q_mean": sum(qs)/N, "q_max": max(qs),
            "inbox": sum(1 for i in range(N) if 0<=out[i*PS]<=1 and 0<=out[i*PS+1]<=1)}

fails = 0
def check(name, cond, detail=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  " + detail))
    if not cond: fails += 1

# ---- disturbance ratchet test: poke settled sand twice, density must recover ----
def disturb_cycle():
    random.seed(11)
    pts = []
    y = 0.04
    while y < 0.40:
        x = 0.30
        while x < 0.70:
            pts.append((x + (random.random()-0.5)*0.004, y + (random.random()-0.5)*0.004))
            x += 0.006
        y += 0.006
    N = len(pts)
    pdata = array.array("f", [0.0]*(N*PS))
    for i,(x,yy) in enumerate(pts):
        b = i*PS; pdata[b]=x; pdata[b+1]=yy; pdata[b+8]=1.0; pdata[b+11]=1.0; pdata[b+14]=RHO_REST0
    pbuf = device.create_buffer(size=N*PS*4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
    device.queue.write_buffer(pbuf, 0, pdata.tobytes())
    ga = device.create_buffer(size=NG*NG*3*4, usage=wgpu.BufferUsage.STORAGE)
    gv = device.create_buffer(size=NG*NG*16, usage=wgpu.BufferUsage.STORAGE)
    uni = device.create_buffer(size=80, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
    rho_rest = (1/NG/0.006)**2
    device.queue.write_buffer(uni, 0, struct.pack("20f",
        DT, NG, 1/NG, GRAV, MU, LA, alpha_from_phi(30), 1.0,
        0,0,0,0, 0, float(N), NG, 0.5, rho_rest, 0.0,0,0))
    def bg(pipe, bufs):
        return device.create_bind_group(layout=pipe.get_bind_group_layout(0),
            entries=[{"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(bufs)])
    bgs = { 'clear': bg(clear_p,[ga]), 'p2g': bg(p2g_p,[pbuf,ga,uni]),
            'grid': bg(grid_p,[ga,gv,uni]), 'g2p': bg(g2p_p,[pbuf,gv,uni]) }
    def run(frames):
        for f in range(frames):
            enc = device.create_command_encoder()
            for si in range(SUB):
                for pipe,key,wg in [(clear_p,'clear',(NG*NG*3+255)//256),(p2g_p,'p2g',(N+63)//64),
                                    (grid_p,'grid',(NG*NG+63)//64),(g2p_p,'g2p',(N+63)//64)]:
                    cp = enc.begin_compute_pass(); cp.set_pipeline(pipe); cp.set_bind_group(0,bgs[key])
                    cp.dispatch_workgroups(wg); cp.end()
            device.queue.submit([enc.finish()])
    def p95h():
        out = array.array("f"); out.frombytes(bytes(device.queue.read_buffer(pbuf)))
        ys = sorted(out[i*PS+1] for i in range(N))
        return ys[int(N*0.95)], out
    run(120)
    h0, out = p95h()
    heights = [h0]
    rndv = random.Random(99)
    for cyc in range(2):
        # disturbance: random velocity kick to every particle
        for i in range(N):
            out[i*PS+2] = (rndv.random()-0.5)*0.8
            out[i*PS+3] = rndv.random()*0.5
        device.queue.write_buffer(pbuf, 0, out.tobytes())
        run(160)
        h, out = p95h()
        heights.append(h)
    return heights

hs = disturb_cycle()
print(f"settled height, then after 2 disturbance cycles: {['%.3f'%h for h in hs]}")
check("PHYSICS: no density ratchet (height stable within 12% over cycles)",
      hs[1] < hs[0]*1.12 and hs[2] < hs[0]*1.12, str(hs))

# ---- sustained-stir fluff test: drive a synthetic finger, density must recover ----
def stir_test():
    random.seed(11)
    pts = []
    y = 0.04
    while y < 0.40:
        x = 0.30
        while x < 0.70:
            pts.append((x + (random.random()-0.5)*0.004, y + (random.random()-0.5)*0.004))
            x += 0.006
        y += 0.006
    N = len(pts)
    pdata = array.array("f", [0.0]*(N*PS))
    for i,(x,yy) in enumerate(pts):
        b = i*PS; pdata[b]=x; pdata[b+1]=yy; pdata[b+8]=1.0; pdata[b+11]=1.0; pdata[b+14]=RHO_REST0
    pbuf = device.create_buffer(size=N*PS*4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
    device.queue.write_buffer(pbuf, 0, pdata.tobytes())
    ga = device.create_buffer(size=NG*NG*3*4, usage=wgpu.BufferUsage.STORAGE)
    gv = device.create_buffer(size=NG*NG*16, usage=wgpu.BufferUsage.STORAGE)
    uni = device.create_buffer(size=80, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
    def set_uni(mx=0.0, my=0.0, mvx=0.0, mvy=0.0, on=0.0):
        rho_rest = (1/NG/0.006)**2
        device.queue.write_buffer(uni, 0, struct.pack("20f",
            DT, NG, 1/NG, GRAV, MU, LA, alpha_from_phi(20), 1.0,
            mx, my, mvx, mvy, on, float(N), NG, 0.5, rho_rest, 0.0,0,0))
    set_uni()
    def bg(pipe, bufs):
        return device.create_bind_group(layout=pipe.get_bind_group_layout(0),
            entries=[{"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(bufs)])
    bgs = { 'clear': bg(clear_p,[ga]), 'p2g': bg(p2g_p,[pbuf,ga,uni]),
            'grid': bg(grid_p,[ga,gv,uni]), 'g2p': bg(g2p_p,[pbuf,gv,uni]) }
    def run(frames):
        for f in range(frames):
            enc = device.create_command_encoder()
            for si in range(SUB):
                for pipe,key,wg in [(clear_p,'clear',(NG*NG*3+255)//256),(p2g_p,'p2g',(N+63)//64),
                                    (grid_p,'grid',(NG*NG+63)//64),(g2p_p,'g2p',(N+63)//64)]:
                    cp = enc.begin_compute_pass(); cp.set_pipeline(pipe); cp.set_bind_group(0,bgs[key])
                    cp.dispatch_workgroups(wg); cp.end()
            device.queue.submit([enc.finish()])
    def state():
        out = array.array("f"); out.frombytes(bytes(device.queue.read_buffer(pbuf)))
        ys = sorted(out[i*PS+1] for i in range(N))
        vc = sum(out[i*PS+13] for i in range(N))/N
        return ys[int(N*0.95)], vc
    run(120)
    h0, vc0 = state()
    # synthetic finger: circles through the pile for 180 frames
    for f in range(180):
        ang = f * 0.25
        mx = 0.5 + 0.22*math.cos(ang)
        my = 0.14 + 0.10*math.sin(ang)
        mvx = -math.sin(ang)*1.6
        mvy = math.cos(ang)*1.6
        set_uni(mx, my, mvx, mvy, 1.0)
        run(1)
    set_uni()
    run(400)
    h1, vc1 = state()
    return h0, h1, vc0, vc1

h0, h1, vc0, vc1 = stir_test()
print(f"stir test: settled {h0:.3f} -> after stir+settle {h1:.3f}   mean vc {vc0:.4f} -> {vc1:.4f}")
check("PHYSICS: sustained stirring does not fluff (height within 15%)", h1 < h0*1.15, f"{h0:.3f} -> {h1:.3f}")

# ---- boil test: deep pressurized fill + stir must decay to rest, no fountains ----
def boil_test():
    random.seed(9)
    pts=[]; y=0.04
    while y<0.38:
        x=0.06
        while x<0.94:
            pts.append((x+(random.random()-0.5)*0.003,y+(random.random()-0.5)*0.003)); x+=0.0045
        y+=0.0045
    N=len(pts)
    pd=array.array("f",[0.0]*(N*PS))
    rr=(1/NG/0.0045)**2
    for i,(x,yy) in enumerate(pts):
        b=i*PS; pd[b]=x; pd[b+1]=yy; pd[b+8]=1.0; pd[b+11]=1.0; pd[b+14]=rr
    pbuf=device.create_buffer(size=N*PS*4,usage=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_DST|wgpu.BufferUsage.COPY_SRC)
    device.queue.write_buffer(pbuf,0,pd.tobytes())
    ga=device.create_buffer(size=NG*NG*3*4,usage=wgpu.BufferUsage.STORAGE)
    gv=device.create_buffer(size=NG*NG*16,usage=wgpu.BufferUsage.STORAGE)
    uni=device.create_buffer(size=80,usage=wgpu.BufferUsage.UNIFORM|wgpu.BufferUsage.COPY_DST)
    def set_uni(mx=0.,my=0.,mvx=0.,mvy=0.,on=0.):
        device.queue.write_buffer(uni,0,struct.pack("20f",DT,NG,1/NG,GRAV,MU,LA,alpha_from_phi(45),1.0,mx,my,mvx,mvy,on,float(N),NG,0.5,rr,0.0,0,0))
    set_uni()
    def bg(pipe,bufs): return device.create_bind_group(layout=pipe.get_bind_group_layout(0),
        entries=[{"binding":i,"resource":{"buffer":b,"offset":0,"size":b.size}} for i,b in enumerate(bufs)])
    bgs={"clear":bg(clear_p,[ga]),"p2g":bg(p2g_p,[pbuf,ga,uni]),
         "grid":bg(grid_p,[ga,gv,uni]),"g2p":bg(g2p_p,[pbuf,gv,uni])}
    def run(frames):
        for f in range(frames):
            enc=device.create_command_encoder()
            for pp,k,wg in [(clear_p,'clear',(NG*NG*3+255)//256),(p2g_p,'p2g',(N+63)//64),
                            (grid_p,'grid',(NG*NG+63)//64),(g2p_p,'g2p',(N+63)//64)]:
                for _ in [0]:
                    pass
            for si in range(SUB):
                for pp,k,wg in [(clear_p,'clear',(NG*NG*3+255)//256),(p2g_p,'p2g',(N+63)//64),
                                (grid_p,'grid',(NG*NG+63)//64),(g2p_p,'g2p',(N+63)//64)]:
                    cp=enc.begin_compute_pass(); cp.set_pipeline(pp); cp.set_bind_group(0,bgs[k])
                    cp.dispatch_workgroups(wg); cp.end()
            device.queue.submit([enc.finish()])
    run(100)
    for f in range(60):
        ang=f*0.3
        set_uni(0.5+0.25*math.cos(ang),0.35+0.2*math.sin(ang),-math.sin(ang)*1.8,math.cos(ang)*1.8,1.0)
        run(1)
    set_uni()
    # settle to criterion, not stopwatch: chunks until calm or budget exhausted
    ke, hs = 1.0, 1.0
    for chunk in range(10):
        run(250)
        out=array.array("f"); out.frombytes(bytes(device.queue.read_buffer(pbuf)))
        ke=sum(out[i*PS+2]**2+out[i*PS+3]**2 for i in range(N))/N
        hi=[math.hypot(out[i*PS+2],out[i*PS+3]) for i in range(N) if out[i*PS+1]>0.80]
        hs=(sum(hi)/len(hi)) if hi else 0.0
        if ke < 0.0008: break
    return ke, hs

bke, bhs = boil_test()
print(f"boil test: tail ke {bke:.5f}, high-particle mean speed {bhs:.4f}")
check("PHYSICS: no boiling (deep stirred fill decays to rest)", bke < 0.001 and bhs < 0.03,
      f"ke {bke:.5f} speed {bhs:.4f}")

lo = run_collapse(15)
hi = run_collapse(45)
print(f"phi=15: runout {lo['runout']:.3f} height {lo['height']:.3f} ke {lo['ke']:.5f}")
print(f"phi=45: runout {hi['runout']:.3f} height {hi['height']:.3f} ke {hi['ke']:.5f}")
check("no launch explosion (width at frame 8 near initial 0.12)",
      lo["early"] < 0.30 and hi["early"] < 0.30, f"{lo['early']:.3f}/{hi['early']:.3f}")
check("no NaN/Inf positions", lo["bad"] == 0 and hi["bad"] == 0)
check("all particles in domain", lo["inbox"] == lo["N"] and hi["inbox"] == hi["N"])
check("column collapsed (p95 height well below initial)", lo["height"] < 0.35 and hi["height"] < 0.45,
      f"{lo['height']:.3f}/{hi['height']:.3f}")
check("comes to rest (mean v^2 small)", lo["ke"] < 0.02 and hi["ke"] < 0.02, f"{lo['ke']:.4f}/{hi['ke']:.4f}")
check("PHYSICS: low friction runs out farther", lo["runout"] > hi["runout"] + 0.03,
      f"15deg {lo['runout']:.3f} vs 45deg {hi['runout']:.3f}")
check("PHYSICS: high friction pile stands taller", hi["height"] > lo["height"] + 0.01,
      f"{hi['height']:.3f} vs {lo['height']:.3f}")
check("PHYSICS: flow accumulated plastic strain (jamming state)", lo["q_mean"] > 0.02 and lo["q_max"] <= 2.0,
      f"q mean {lo['q_mean']:.3f} max {lo['q_max']:.3f}")
check("PHYSICS: low-friction flow hardened more than high-friction", lo["q_mean"] > hi["q_mean"],
      f"{lo['q_mean']:.3f} vs {hi['q_mean']:.3f}")
print("\n" + ("ALL TESTS PASS" if fails == 0 else f"{fails} FAILURES"))
sys.exit(1 if fails else 0)
