#!/usr/bin/env python3
"""Rainbow Silt GPU test rig.
Extracts the WGSL from silt-gpu.html (tested code == shipped code), validates
every shader through naga, then executes the step kernel on the software
adapter and asserts physical behavior: mass conservation, gravity profile,
grounding, and toppling.
"""
import re, sys, os
os.environ.setdefault("WGPU_BACKEND_TYPE", "")
import wgpu

HTML = "/home/claude/rainbow/silt-gpu.html"
STEPS = 12  # must match STEPS_PER_FRAME in the HTML

# ---- extract WGSL exactly as the JS assembles it ----
src = open(HTML).read()
def grab(name):
    m = re.search(r"const " + name + r"\s*=\s*(WGSL_COMMON \+ )?/\* wgsl \*/`(.*?)`;", src, re.S)
    assert m, name
    return m.group(2), bool(m.group(1))
common = re.search(r"const WGSL_COMMON = /\* wgsl \*/`(.*?)`;", src, re.S).group(1)
shaders = {}
for name in ["WGSL_STEP", "WGSL_SPAWN", "WGSL_CLEAR", "WGSL_RENDER"]:
    body, uses_common = grab(name)
    shaders[name] = (common + body) if uses_common else body

adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync()
print("adapter:", adapter.info["device"])

# ---- 1. validation: every shader must compile ----
mods = {}
for name, code in shaders.items():
    try:
        mods[name] = device.create_shader_module(code=code)
        print(f"VALIDATE {name}: OK")
    except Exception as e:
        print(f"VALIDATE {name}: FAIL\n{e}")
        sys.exit(1)

# ---- harness for the step kernel ----
W, H = 16, 256
NCELLS = W * H
pipe = device.create_compute_pipeline(layout="auto", compute={"module": mods["WGSL_STEP"], "entry_point": "main"})

cells_buf = device.create_buffer(size=NCELLS * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
unis, bgs = [], []
for _ in range(STEPS):
    u = device.create_buffer(size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
    unis.append(u)
    bgs.append(device.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
        {"binding": 0, "resource": {"buffer": cells_buf, "offset": 0, "size": NCELLS * 4}},
        {"binding": 1, "resource": {"buffer": u, "offset": 0, "size": 32}},
    ]))

import struct, random
def run_frame(fidx):
    enc = device.create_command_encoder()
    for s_i in range(STEPS):
        device.queue.write_buffer(unis[s_i], 0, struct.pack("8I", W, H, s_i & 1, random.getrandbits(32), s_i, STEPS, fidx & 1, 0))
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipe)
        cp.set_bind_group(0, bgs[s_i])
        cp.dispatch_workgroups((W // 2 + 7) // 8, (H // 2 + 7) // 8)
        cp.end()
    device.queue.submit([enc.finish()])

def upload(grid):
    device.queue.write_buffer(cells_buf, 0, grid.tobytes())
def download():
    import array
    data = device.queue.read_buffer(cells_buf)
    a = array.array("I"); a.frombytes(bytes(data))
    return a

import array
fails = 0
def check(name, cond, detail=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  " + detail))
    if not cond: fails += 1

# ---- 2. gravity profile: single grain in an empty column ----
grid = array.array("I", [0] * NCELLS)
grid[2 * W + 8] = 3  # one green grain near the top, vel 0
upload(grid)
ys, prev_y = [2], 2
for f in range(70):
    run_frame(f)
    data = download()
    pos = [i for i, v in enumerate(data) if v & 0xFF]
    assert len(pos) == 1, f"grain count {len(pos)} at frame {f}"
    y = pos[0] // W
    ys.append(y)
    if y >= H - 1: break
dy = [ys[i+1] - ys[i] for i in range(len(ys) - 1)]
print("dy per frame:", dy)
early = sum(dy[:8]) / 8
late = sum(dy[16:28]) / max(1, len(dy[16:28]))
check("gravity: accelerates (late speed >> early speed)", late >= early + 3, f"early {early:.1f} late {late:.1f}")
check("gravity: starts slow (first-frame fall <= 6)", dy[0] <= 6, str(dy[0]))
check("gravity: approaches terminal (some frame falls >= 10)", max(dy) >= 10, str(max(dy)))

# ---- 2b. cohort dispersion: same-frame spawns must NOT fall in a band ----
grid = array.array("I", [0] * NCELLS)
for x in range(4, 12): grid[2 * W + x] = 4  # 8 grains, one row, same "frame"
upload(grid)
for f in range(20): run_frame(f)
data = download()
cohort_ys = sorted(i // W for i, v in enumerate(data) if (v & 0xFF) == 4)
spread = cohort_ys[-1] - cohort_ys[0] if cohort_ys else 0
check("dispersion: same-frame cohort spreads >= 5 cells", spread >= 5, f"ys {cohort_ys}")

# ---- 3. mass conservation under chaotic settling ----
random.seed(7)
grid = array.array("I", [0] * NCELLS)
for i in range(NCELLS):
    y = i // W
    if y > H // 3 and random.random() < 0.35:
        grid[i] = random.randint(1, 5)
# stone shelf (stamped over whatever was there -- count sand AFTER this)
for x in range(4, 12): grid[(H // 2) * W + x] = 6
n0 = sum(1 for v in grid if 1 <= (v & 0xFF) <= 5)
upload(grid)
for f in range(120): run_frame(f)
data = download()
n1 = sum(1 for v in data if 1 <= (v & 0xFF) <= 5)
stones = sum(1 for v in data if (v & 0xFF) == 6)
check("mass conservation: sand", n1 == n0, f"{n0} -> {n1}")
check("mass conservation: stone", stones == 8, str(stones))

# ---- 4. grounding + topple: a 1-wide tower must collapse into a pile ----
grid = array.array("I", [0] * NCELLS)
for y in range(H - 40, H): grid[y * W + 8] = 2
upload(grid)
for f in range(160): run_frame(f)
data = download()
bottom_row = [x for x in range(W) if (data[(H - 1) * W + x] & 0xFF)]
heights = {}
for i, v in enumerate(data):
    if v & 0xFF == 2:
        x = i % W; heights[x] = max(heights.get(x, 0), H - i // W)
check("topple: tower spread to >= 7 columns", len(heights) >= 7, str(sorted(heights)))
check("topple: bottom row grains participated", len(bottom_row) >= 5, str(bottom_row))
check("topple: max height dropped below 20", max(heights.values()) < 20, str(max(heights.values())))

# ---- 4b. spawn width follows the halfw uniform ----
spawn_pipe = device.create_compute_pipeline(layout="auto", compute={"module": mods["WGSL_SPAWN"], "entry_point": "main"})
spawn_uni = device.create_buffer(size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
spawn_bg = device.create_bind_group(layout=spawn_pipe.get_bind_group_layout(0), entries=[
    {"binding": 0, "resource": {"buffer": cells_buf, "offset": 0, "size": NCELLS * 4}},
    {"binding": 1, "resource": {"buffer": spawn_uni, "offset": 0, "size": 32}}])
for halfw, lo, hi in [(1, 1, 3), (20, 25, 41)]:
    upload(array.array("I", [0] * NCELLS))
    for rep in range(6):  # several dispatches to fill the span
        device.queue.write_buffer(spawn_uni, 0, struct.pack("8I", W if False else 84 if False else W, H, W // 2, 2, 64, random.getrandbits(32), halfw, 0))
        enc = device.create_command_encoder()
        cp = enc.begin_compute_pass(); cp.set_pipeline(spawn_pipe); cp.set_bind_group(0, spawn_bg)
        cp.dispatch_workgroups(1); cp.end()
        device.queue.submit([enc.finish()])
    data = download()
    xs = sorted({i % W for i, v in enumerate(data) if v & 0xFF})
    extent = (xs[-1] - xs[0] + 1) if xs else 0
    # W=16 clips the wide case; only assert what fits
    ok = (extent <= hi) and (extent >= min(lo, W))
    check(f"spawn width halfw={halfw}: extent {extent} within [{min(lo, W)},{hi}]", ok, str(xs))

# ---- 5. tick-flag hygiene: no stuck flags after a frame completes ----
stuck = sum(1 for v in data if v & 0x1000)
check("tick flags cleared at frame end", stuck == 0, str(stuck))

print("\n" + ("ALL TESTS PASS" if fails == 0 else f"{fails} FAILURES"))
sys.exit(1 if fails else 0)
