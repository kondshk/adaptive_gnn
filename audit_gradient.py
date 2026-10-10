#!/usr/bin/env python3
"""Gradient audit (Sec. 5.3): gradient flow through TannerGNN message passing.

For an untrained model and for each of the five headline checkpoints
(results/headline_v2/seed*_best.pt), one focal-loss backward pass on the first
256 shots of the p = 0.04 test set gives the mean absolute gradient of the
input projection and of the readout. A ratio below 0.01 would mean the
gradient vanishes through the message-passing stack. Also reports the mean
|Delta_i| of the LLR corrections.

Usage:
    python audit_gradient.py      # writes results/diagnostics/gradient_audit.json
"""
from __future__ import annotations
import sys, json, pathlib
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gnn_pipeline.tanner_graph import build_tanner_graph
from gnn_pipeline.gnn_model import TannerGNN
from gnn_pipeline.loss_functions import focal_loss

torch.manual_seed(42); np.random.seed(42)

DATA_FILE = ROOT / 'data' / 'headline_v2' / 'test_p04.npz'; N = 256
CKPTS = sorted((ROOT / 'results' / 'headline_v2').glob('seed*_best.pt'))
OUT = ROOT / 'results' / 'diagnostics' / 'gradient_audit.json'

raw=np.load(DATA_FILE,allow_pickle=True)
hx=raw['hx'].astype(np.uint8); hz=raw['hz'].astype(np.uint8)
n=hx.shape[1]; mx=hx.shape[0]; mz=hz.shape[0]; num_nodes=n+mx+mz

syn=raw['syndromes'][:N].astype(np.float32)
pv=raw['p_values'][:N].astype(np.float32)
llr=np.log((1-pv)/pv).astype(np.float32)
ze=raw['z_errors'][:N].astype(np.float32)

node_type_np,edge_index_np,edge_type_np=build_tanner_graph(hx,hz)
nt_t=torch.from_numpy(node_type_np)
ei_t=torch.from_numpy(edge_index_np)
ety_t=torch.from_numpy(edge_type_np)

def mk(i):
    # same node features as audit_headline_v2.Harness.graphs
    x=torch.zeros(num_nodes,4)
    x[:n,0]=float(llr[i]); x[:n,1]=1.
    x[n:n+mx,0]=torch.from_numpy(syn[i,:mx]); x[n:n+mx,2]=1.
    x[n+mx:, 0]=torch.from_numpy(syn[i,mx:]); x[n+mx:, 3]=1.
    return Data(x=x,edge_index=ei_t,edge_type=ety_t,node_type=nt_t,
                y=torch.from_numpy(ze[i]))

batch=next(iter(DataLoader([mk(i) for i in range(N)],batch_size=N,shuffle=False)))

def build_gnn():
    return TannerGNN(node_feat_dim=4,hidden_dim=32,num_mp_layers=3,
                     dropout=0.,use_film=False,use_attention=False,
                     use_residual=True,use_layer_norm=True)

def measure(gnn,batch,desc):
    gnn.train()
    for p in gnn.parameters():
        if p.grad is not None: p.grad.zero_()
    out=gnn(batch)
    is_d=batch.node_type==0
    avg=batch.x[is_d,0]; d=out.view(-1)
    pred=torch.sigmoid(-(avg+d))
    tgt=batch.y.view(-1).clamp(0,1)
    loss=focal_loss(pred,tgt)
    loss.backward()

    grads={}
    for nm,p in gnn.named_parameters():
        if p.grad is not None:
            g=p.grad.detach().abs()
            grads[nm]=dict(mean=float(g.mean()),max=float(g.max()),norm=float(g.norm()))

    in_ms  =[grads[k]['mean'] for k in grads if 'input_proj' in k]
    out_ms  =[grads[k]['mean'] for k in grads if 'readout'    in k]
    mp_ms   =[grads[k]['mean'] for k in grads if 'mp_layers'  in k]

    mi=float(np.mean(in_ms))  if in_ms  else 0.
    mo=float(np.mean(out_ms)) if out_ms else 0.
    mm=float(np.mean(mp_ms))  if mp_ms  else 0.
    ratio=mi/(mo+1e-12)
    mean_d=float(d.detach().abs().mean())

    flag=('VANISHING' if ratio<0.01 else 'EXPLODING' if ratio>100 else 'HEALTHY')
    print(f"\n  {desc}")
    print(f"    input proj grad: {mi:.3e} | mp layers: {mm:.3e} | readout: {mo:.3e}")
    print(f"    ratio input/readout: {ratio:.4f} → {flag}")
    print(f"    mean |delta_i|: {mean_d:.6f}  loss={loss.item():.4f}")
    return dict(desc=desc,mi=mi,mm=mm,mo=mo,ratio=ratio,mean_delta=mean_d,
                loss=float(loss.item()),vanishing=ratio<0.01,exploding=ratio>100,flag=flag)

print("=== Gradient audit ===")
r0=measure(build_gnn(),batch,"Untrained")

trained=[]
for ck in CKPTS:
    gnn=build_gnn()
    gnn.load_state_dict(torch.load(ck,map_location='cpu'))
    trained.append(dict(measure(gnn,batch,f"Trained ({ck.name})"),checkpoint=str(ck.relative_to(ROOT))))

ratios=np.array([r['ratio'] for r in trained]); deltas=np.array([r['mean_delta'] for r in trained])
summary=dict(ratio_mean=float(ratios.mean()),ratio_std=float(ratios.std(ddof=1)),
             ratio_min=float(ratios.min()),ratio_max=float(ratios.max()),
             mean_delta_mean=float(deltas.mean()),mean_delta_std=float(deltas.std(ddof=1)),
             mean_delta_min=float(deltas.min()),mean_delta_max=float(deltas.max()),
             any_vanishing=bool(any(r['vanishing'] for r in trained)),
             any_exploding=bool(any(r['exploding'] for r in trained)))
print("\n=== Summary over checkpoints ===")
print(f"  Untrained: ratio={r0['ratio']:.4f} delta={r0['mean_delta']:.4f}")
print(f"  Trained:   ratio {summary['ratio_mean']:.3f} ± {summary['ratio_std']:.3f} "
      f"[{summary['ratio_min']:.3f}, {summary['ratio_max']:.3f}], "
      f"mean |delta| {summary['mean_delta_mean']:.3f} ± {summary['mean_delta_std']:.3f} "
      f"[{summary['mean_delta_min']:.3f}, {summary['mean_delta_max']:.3f}]")

OUT.parent.mkdir(parents=True,exist_ok=True)
results=dict(data=f"{DATA_FILE.relative_to(ROOT)} (first {N} shots)",
             untrained=r0,trained=trained,summary=summary)
OUT.write_text(json.dumps(results,indent=2))
print(f"Results saved to {OUT.relative_to(ROOT)}")
