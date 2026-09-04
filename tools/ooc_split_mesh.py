#!/usr/bin/env python3
"""
Out-of-core spatial splitter for RealityScan mesh PLY that is too big to load
whole (billions of faces; 32-bit-overflowed header counts).

- Recovers true vertex/face counts from file size (header counts may be negative
  due to int32 overflow).
- Loads the whole vertex block into RAM (positions + colour).
- Streams the face block once, buckets each face into a ground-plane (XY) grid
  cell by its centroid, routes face vertex-indices to per-cell temp files.
- Finalises each cell into a valid, recentred, vertex-coloured mesh PLY that
  mesh2splat can load as one loadable part.

Usage:
  python ooc_split.py <in.ply> --plan                      # cheap: grid + part plan
  python ooc_split.py <in.ply> <out_dir> [--budget 30]     # full split (M faces/part)
"""
import sys, os, struct, argparse, time, glob
import numpy as np

VZ_TYPES = {b'float':4,b'float32':4,b'double':8,b'uchar':1,b'uint8':1,b'char':1,
            b'int8':1,b'ushort':2,b'uint16':2,b'short':2,b'int16':2,b'uint':4,
            b'uint32':4,b'int':4,b'int32':4}

def parse_header(path):
    with open(path,'rb') as f:
        raw = f.read(8192)
    hend = raw.find(b'end_header\n')+len(b'end_header\n')
    lines = raw[:hend].split(b'\n')
    vprops=[]; fprops=[]; cur=None; nV=nF=None
    for ln in lines:
        t=ln.split()
        if not t: continue
        if t[0]==b'element':
            cur=t[1]
            if cur==b'vertex': nV=int(t[2])
            elif cur==b'face':  nF=int(t[2])
        elif t[0]==b'property':
            if cur==b'vertex': vprops.append((t[1],t[2]))          # (type,name)
            elif cur==b'face':  fprops.append(tuple(t[1:]))
    vstride=sum(VZ_TYPES[t] for t,_ in vprops)
    names=[n for _,n in vprops]
    # face record assumed: list <countType> <idxType> vertex_indices  (triangles)
    # => 1 (uchar count) + 3*4 (int idx) = 13
    face_stride=13
    return dict(hend=hend, nV_decl=nV, nF_decl=nF, vstride=vstride,
                vprops=vprops, names=names, face_stride=face_stride)

def recover_counts(path, h):
    sz=os.path.getsize(path)
    nV_u = h['nV_decl'] & 0xffffffff if h['nV_decl']<0 else h['nV_decl']
    # trust vertex count (unsigned), derive face count from remaining bytes
    vblock = nV_u * h['vstride']
    fblock = sz - h['hend'] - vblock
    if fblock <= 0 or fblock % h['face_stride'] != 0:
        # fall back: derive vertex count from face count if faces trustworthy
        nF_u = h['nF_decl'] & 0xffffffff if h['nF_decl']<0 else h['nF_decl']
        vblock = sz - h['hend'] - nF_u*h['face_stride']
        nV_u = vblock // h['vstride']
        fblock = nF_u*h['face_stride']
    nF_u = fblock // h['face_stride']
    return sz, nV_u, nF_u

def col_index(names):
    lut={n.decode():i for i,n in enumerate(names)}
    xi=lut.get('x',0); yi=lut.get('y',1); zi=lut.get('z',2)
    ri=lut.get('red', lut.get('r')); gi=lut.get('green',lut.get('g')); bi=lut.get('blue',lut.get('b'))
    return xi,yi,zi,ri,gi,bi

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out_dir', nargs='?')
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--budget', type=float, default=45.0, help='target M faces per part (VRAM-bound)')
    ap.add_argument('--grid', type=int, default=0, help='force NxN grid (0=auto)')
    ap.add_argument('--maxfaces', type=float, default=0, help='TEST: only process first N million faces')
    a=ap.parse_args()

    h=parse_header(a.inp)
    sz,nV,nF=recover_counts(a.inp,h)
    xi,yi,zi,ri,gi,bi=col_index(h['names'])
    print(f"file {sz/1e9:.1f} GB  header {h['hend']}B  vstride {h['vstride']}B")
    print(f"vertex props: {[n.decode() for _,n in h['vprops']]}")
    print(f"declared counts: vertex={h['nV_decl']} face={h['nF_decl']}")
    print(f"RECOVERED: vertices={nV:,}  faces={nF:,}")
    exp = h['hend']+nV*h['vstride']+nF*h['face_stride']
    print(f"layout check: {exp} vs {sz}  {'OK' if exp==sz else 'MISMATCH!'}")

    # --- load vertex positions (need xyz for bucketing) ---
    t0=time.time()
    dt=np.dtype([('v', '<f4', 6)])  # assumes 6 float props (x y z r g b)
    if h['vstride']!=24:
        print("WARN: vstride != 24; this quick loader assumes 6 float props.");
    if a.plan:
        # cheap bbox from a memmap sample; no full load in plan mode
        vmm=np.memmap(a.inp, dtype=dt, mode='r', offset=h['hend'], shape=(nV,))
        SN=2_000_000; picks=[0, nV//2-SN//2, max(0,nV-SN)]
        allp=np.concatenate([np.asarray(vmm['v'][p:p+SN])[:, :3] for p in picks])
        mn=allp.min(axis=0); mx=allp.max(axis=0); VB=None
    else:
        print(f"loading vertex block into RAM (~{nV*h['vstride']/1e9:.0f} GB)...", flush=True)
        with open(a.inp,'rb') as vf:
            vf.seek(h['hend']); VB=np.fromfile(vf, dtype=dt, count=nV)
        mn=VB['v'][:, :3].min(axis=0); mx=VB['v'][:, :3].max(axis=0)
    print(f"bbox min {mn}  max {mx}  span {mx-mn}  ({time.time()-t0:.1f}s)", flush=True)

    spanx, spany = float(mx[0]-mn[0]), float(mx[1]-mn[1])
    # auto grid: aim avg ~ budget/2 faces/cell so peaks stay under budget
    if a.grid>0:
        G=a.grid
    else:
        target_cells = max(1, int(np.ceil(nF/(a.budget*1e6*0.6))))
        G=int(np.ceil(np.sqrt(target_cells)))
    print(f"\nPLAN: grid {G}x{G} = {G*G} cells over XY  (cell {spanx/G:.1f} x {spany/G:.1f} units)")
    print(f"      ~{nF/(G*G)/1e6:.1f} M faces/cell avg, budget {a.budget} M/part")
    print(f"      est parts (occupied cells) <= {G*G};  est total part data ~ {sz/1e9:.0f} GB")

    if a.plan:
        print("\n[--plan] stopping before any heavy work.")
        return

    # ---- FULL SPLIT ----
    assert a.out_dir, "need out_dir for full split"
    os.makedirs(a.out_dir, exist_ok=True)
    tmpdir=os.path.join(a.out_dir,'_tmp_gids'); os.makedirs(tmpdir, exist_ok=True)
    origin=np.floor(mn).astype(np.float64)   # recenter origin (add back for georef)
    print(f"recenter origin (add back to georef): {origin.tolist()}")

    cellsx=lambda X: np.clip(((X-mn[0])/(spanx/G)).astype(np.int64),0,G-1)
    cellsy=lambda Y: np.clip(((Y-mn[1])/(spany/G)).astype(np.int64),0,G-1)

    # pass 1: stream faces, route (3 gids) to per-cell temp files
    face_dt=np.dtype([('c','u1'),('i','<i4',3)])   # 13B triangle record
    fmm=np.memmap(a.inp, dtype=face_dt, mode='r',
                  offset=h['hend']+nV*h['vstride'], shape=(nF,))
    xyz=VB['v']  # in-RAM: fancy index is fast (no page faults)
    handles={}
    def th(cid):
        if cid not in handles: handles[cid]=open(os.path.join(tmpdir,f'c{cid}.bin'),'wb')
        return handles[cid]
    CH=20_000_000
    nF_do = int(a.maxfaces*1e6) if a.maxfaces>0 else nF
    if nF_do<nF: print(f"[TEST] processing only first {nF_do:,} faces")
    t0=time.time()
    for s in range(0,nF_do,CH):
        e=min(nF_do,s+CH)
        idx=np.asarray(fmm['i'][s:e])                 # (m,3) int32 (unsigned via &)
        idxu=idx.astype(np.int64)&0xffffffff
        # centroid XY via vertex lookups
        p0=xyz[idxu[:,0]]; p1=xyz[idxu[:,1]]; p2=xyz[idxu[:,2]]
        cx=(p0[:,0]+p1[:,0]+p2[:,0])/3.0
        cy=(p0[:,1]+p1[:,1]+p2[:,1])/3.0
        cid=cellsx(cx)*G+cellsy(cy)
        order=np.argsort(cid,kind='stable'); cid_s=cid[order]; idxu_s=idxu[order]
        uc,starts=np.unique(cid_s,return_index=True)
        starts=list(starts)+[len(cid_s)]
        for k,c in enumerate(uc):
            block=idxu_s[starts[k]:starts[k+1]].astype('<i4')
            th(int(c)).write(block.tobytes())
        print(f"  routed {e/1e6:.0f}/{nF/1e6:.0f} M faces  ({time.time()-t0:.0f}s)", end='\r')
    for f in handles.values(): f.close()
    print(f"\nrouting done ({time.time()-t0:.0f}s). finalising parts...")

    # pass 2: each temp -> valid recentred vertex-coloured mesh PLY
    part=0
    for tf in sorted(glob.glob(os.path.join(tmpdir,'c*.bin'))):
        gids=np.fromfile(tf,dtype='<i4').astype(np.int64)&0xffffffff
        gids=gids.reshape(-1,3)
        if len(gids)==0: continue
        uniq,inv=np.unique(gids.reshape(-1),return_inverse=True)
        Vsub=VB['v'][uniq]                     # (nu,6) positions+colour
        pos=Vsub[:, :3]-origin.astype(np.float32)
        col=Vsub[:, 3:6]
        faces=inv.reshape(-1,3).astype('<i4')
        outp=os.path.join(a.out_dir,f'part_{part:04d}.ply')
        with open(outp,'wb') as o:
            o.write(b'ply\nformat binary_little_endian 1.0\n')
            o.write(f'element vertex {len(uniq)}\n'.encode())
            for nm in (b'x',b'y',b'z',b'red',b'green',b'blue'):
                o.write(b'property float '+nm+b'\n')
            o.write(f'element face {len(faces)}\n'.encode())
            o.write(b'property list uchar int vertex_indices\nend_header\n')
            vout=np.empty((len(uniq),6),dtype='<f4'); vout[:, :3]=pos; vout[:,3:]=col
            o.write(vout.tobytes())
            fr=np.empty((len(faces),),dtype=[('c','u1'),('i','<i4',3)])
            fr['c']=3; fr['i']=faces
            o.write(fr.tobytes())
        print(f"  part_{part:04d}: {len(faces):,} faces, {len(uniq):,} verts -> {os.path.getsize(outp)/1e9:.2f} GB")
        part+=1
        os.remove(tf)
    os.rmdir(tmpdir)
    print(f"\nDONE: {part} parts in {a.out_dir}  (origin {origin.tolist()} to add back for georef)")

if __name__=='__main__':
    main()
