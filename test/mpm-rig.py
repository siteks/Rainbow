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

NG = 64; DT = 3.0e-4; GRAV = 25.0
E, NU = 1200.0, 0.3
MU = E / (2*(1+NU)); LA = E*NU/((1+NU)*(1-2*NU))
PS = 12; SUB = 12

clear_p = device.create_compute_pipeline(layout="auto", compute={"module": mods["WGSL_CLEAR"], "entry_point": "main"})
p2g_p = device.create_compute_pipeline(layout="auto", compute={"module": mods["WGSL_P2G"], "entry_point": "main"})
grid_p = device.create_compute_pipeline(layout="auto", compute={"module": mods["WGSL_GRID"], "entry_point": "main"})
g2p_p = device.create_compute_pipeline(layout="auto", compute={"module": mods["WGSL_G2P"], "entry_point": "main"})

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
        b = i*PS; pdata[b]=x; pdata[b+1]=yy; pdata[b+8]=1.0; pdata[b+11]=1.0
    pbuf = device.create_buffer(size=N*PS*4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
    device.queue.write_buffer(pbuf, 0, pdata.tobytes())
    ga = device.create_buffer(size=NG*NG*3*4, usage=wgpu.BufferUsage.STORAGE)
    gv = device.create_buffer(size=NG*NG*8, usage=wgpu.BufferUsage.STORAGE)
    uni = device.create_buffer(size=64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
    pvol = 0.5; pmass = 1.0   # mass-normalized units (see sandlab.html)
    device.queue.write_buffer(uni, 0, struct.pack("16f",
        DT, NG, 1/NG, GRAV,  MU, LA, alpha_from_phi(phi_deg), pmass,
        0,0,0,0,  0, float(N), NG, pvol))
    def bg(pipe, bufs):
        return device.create_bind_group(layout=pipe.get_bind_group_layout(0),
            entries=[{"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(bufs)])
    bgs = { 'clear': bg(clear_p, [ga]), 'p2g': bg(p2g_p, [pbuf, ga, uni]),
            'grid': bg(grid_p, [ga, gv, uni]), 'g2p': bg(g2p_p, [pbuf, gv, uni]) }
    for f in range(frames):
        enc = device.create_command_encoder()
        for s in range(SUB):
            for pipe, key, wg in [(clear_p,'clear',(NG*NG*3+255)//256), (p2g_p,'p2g',(N+63)//64),
                                  (grid_p,'grid',(NG*NG+63)//64), (g2p_p,'g2p',(N+63)//64)]:
                cp = enc.begin_compute_pass(); cp.set_pipeline(pipe); cp.set_bind_group(0, bgs[key])
                cp.dispatch_workgroups(wg); cp.end()
        device.queue.submit([enc.finish()])
    out = array.array("f"); out.frombytes(bytes(device.queue.read_buffer(pbuf)))
    xs = sorted(out[i*PS] for i in range(N))
    ys = sorted(out[i*PS+1] for i in range(N))
    vs = [math.hypot(out[i*PS+2], out[i*PS+3]) for i in range(N)]
    bad = sum(1 for i in range(N) if not (math.isfinite(out[i*PS]) and math.isfinite(out[i*PS+1])))
    ke = sum(v*v for v in vs)/N
    p95h = ys[int(N*0.95)]
    runout = xs[int(N*0.99)] - xs[int(N*0.01)]   # robust extent
    return {"N":N, "bad":bad, "runout":runout, "height":p95h, "ke":ke,
            "inbox": sum(1 for i in range(N) if 0<=out[i*PS]<=1 and 0<=out[i*PS+1]<=1)}

fails = 0
def check(name, cond, detail=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  " + detail))
    if not cond: fails += 1

lo = run_collapse(15)
hi = run_collapse(45)
print(f"phi=15: runout {lo['runout']:.3f} height {lo['height']:.3f} ke {lo['ke']:.5f}")
print(f"phi=45: runout {hi['runout']:.3f} height {hi['height']:.3f} ke {hi['ke']:.5f}")
check("no NaN/Inf positions", lo["bad"] == 0 and hi["bad"] == 0)
check("all particles in domain", lo["inbox"] == lo["N"] and hi["inbox"] == hi["N"])
check("column collapsed (p95 height well below initial)", lo["height"] < 0.35 and hi["height"] < 0.45,
      f"{lo['height']:.3f}/{hi['height']:.3f}")
check("comes to rest (mean v^2 small)", lo["ke"] < 0.02 and hi["ke"] < 0.02, f"{lo['ke']:.4f}/{hi['ke']:.4f}")
check("PHYSICS: low friction runs out farther", lo["runout"] > hi["runout"] + 0.03,
      f"15deg {lo['runout']:.3f} vs 45deg {hi['runout']:.3f}")
check("PHYSICS: high friction pile stands taller", hi["height"] > lo["height"] + 0.01,
      f"{hi['height']:.3f} vs {lo['height']:.3f}")
print("\n" + ("ALL TESTS PASS" if fails == 0 else f"{fails} FAILURES"))
sys.exit(1 if fails else 0)
