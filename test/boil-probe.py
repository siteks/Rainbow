import re, struct, math, random, array
import wgpu
src = open("/home/claude/rainbow/sandlab.html").read()
common = re.search(r"const WGSL_COMMON = /\* wgsl \*/`(.*?)`;", src, re.S).group(1)
def grab(name):
    m = re.search(r"const " + name + r"\s*=\s*(WGSL_COMMON \+ )?/\* wgsl \*/`(.*?)`;", src, re.S)
    return (common + m.group(2)) if m.group(1) else m.group(2)
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync()
P = {n: device.create_compute_pipeline(layout="auto", compute={"module": device.create_shader_module(code=grab("WGSL_"+n.upper())), "entry_point": "main"}) for n in ["clear","p2g","grid","g2p"]}
NG=64; DT=4e-4; GRAV=32.0; E,NU=4800.0,0.3
MU=E/(2*(1+NU)); LA=E*NU/((1+NU)*(1-2*NU)); PS=16; SUB=12
RHO_REST0 = (1/NG/0.006)**2
def alpha(deg):
    sn=math.sin(math.radians(deg)); return math.sqrt(2/3)*2*sn/(3-sn)
FL=wgpu.BufferUsage
# deep fill: most of the box
random.seed(9)
pts=[]; y=0.04
while y<0.72:
    x=0.05
    while x<0.95:
        pts.append((x+(random.random()-0.5)*0.004,y+(random.random()-0.5)*0.004)); x+=0.007
    y+=0.007
N=len(pts)
print("particles:", N)
pd=array.array("f",[0.0]*(N*PS))
for i,(x,yy) in enumerate(pts):
    b=i*PS; pd[b]=x; pd[b+1]=yy; pd[b+8]=1.0; pd[b+11]=1.0; pd[b+14]=RHO_REST0
pbuf=device.create_buffer(size=N*PS*4,usage=FL.STORAGE|FL.COPY_DST|FL.COPY_SRC)
device.queue.write_buffer(pbuf,0,pd.tobytes())
ga=device.create_buffer(size=NG*NG*3*4,usage=FL.STORAGE)
gv=device.create_buffer(size=NG*NG*16,usage=FL.STORAGE)
uni=device.create_buffer(size=80,usage=FL.UNIFORM|FL.COPY_DST)
rho_rest=(1/NG/0.007)**2
def set_uni(mx=0.,my=0.,mvx=0.,mvy=0.,on=0.):
    device.queue.write_buffer(uni,0,struct.pack("20f",DT,NG,1/NG,GRAV,MU,LA,alpha(45),1.0,mx,my,mvx,mvy,on,float(N),NG,0.5,rho_rest,0.02,0,0))
set_uni()
def bg(pipe,bufs): return device.create_bind_group(layout=pipe.get_bind_group_layout(0),
    entries=[{"binding":i,"resource":{"buffer":b,"offset":0,"size":b.size}} for i,b in enumerate(bufs)])
BG={"clear":bg(P["clear"],[ga]),"p2g":bg(P["p2g"],[pbuf,ga,uni]),
    "grid":bg(P["grid"],[ga,gv,uni]),"g2p":bg(P["g2p"],[pbuf,gv,uni])}
def run(frames):
    for f in range(frames):
        enc=device.create_command_encoder()
        for s_ in range(SUB):
            for pp,k,wg in [(P["clear"],"clear",(NG*NG*3+255)//256),(P["p2g"],"p2g",(N+63)//64),
                            (P["grid"],"grid",(NG*NG+63)//64),(P["g2p"],"g2p",(N+63)//64)]:
                cp=enc.begin_compute_pass(); cp.set_pipeline(pp); cp.set_bind_group(0,BG[k])
                cp.dispatch_workgroups(wg); cp.end()
        device.queue.submit([enc.finish()])
def stats():
    out=array.array("f"); out.frombytes(bytes(device.queue.read_buffer(pbuf)))
    ke=sum(out[i*PS+2]**2+out[i*PS+3]**2 for i in range(N))/N
    vmax=max(math.hypot(out[i*PS+2],out[i*PS+3]) for i in range(N))
    ys=sorted(out[i*PS+1] for i in range(N))
    high=sum(1 for i in range(N) if out[i*PS+1] > 0.80)  # ejecta above fill line
    return ke, vmax, ys[int(N*0.99)], high
run(100)
print("settled:      ke %.5f  vmax %.2f  p99h %.3f  ejecta %d" % stats())
for f in range(60):
    ang=f*0.3
    set_uni(0.5+0.25*math.cos(ang),0.35+0.2*math.sin(ang),-math.sin(ang)*1.8,math.cos(ang)*1.8,1.0)
    run(1)
set_uni()
for tag, n in [("post-stir+50", 50), ("+100", 50), ("+200", 100)]:
    run(n)
    print(f"{tag}:  ke %.5f  vmax %.2f  p99h %.3f  ejecta %d" % stats())
