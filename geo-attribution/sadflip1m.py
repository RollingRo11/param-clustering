import sys, json, torch
import torch.nn.functional as F
sys.path.insert(0, "/workspace/circuit-decomp/geo-attribution")
import geo1b, geo67
from geo67 import GatedRunner, is_write_side
from german67 import ENGLISH, chunks_from, ce_per_tok
SAD_EVAL = ["My laptop crashed and I lost my essay :(", "The flight got delayed by five hours :(", "I burned the cookies again :(", "My bike got a flat tire on the way home :(", "The museum was closed for renovations :(", "I dropped my ice cream on the sidewalk :(", "We had to cancel the picnic because of rain :(", "My team got knocked out of the tournament :(", "I forgot my best friend's birthday :(", "The store sold out right before my turn :("]
from pathlib import Path
D = Path("/dev/shm/geo1b/run1m"); device = "cuda"
SAD_ID, HAPPY_ID = 40624, 27046
target = geo67.load_target(device)
bk = torch.load(D/"banks_prop1m.pt", weights_only=True, map_location="cpu")
run = GatedRunner(target, bk, device)
res = json.loads((D/"emote1b_prop1m.json").read_text())
c = res["contrast_top16"][0]; tid = res["target_id"]
from tokenizers import Tokenizer
tok = Tokenizer.from_file("/dev/shm/geo1b/target_local/tokenizer.json")
eval_sad = chunks_from(SAD_EVAL, tok, 96, device)
eval_ctrl = chunks_from(ENGLISH[12:], tok, 96, device)
saved = {}
def additive(gain):
    ct = torch.tensor([c], device=device)
    d = target.get_submodule("hf.model.embed_tokens").weight[tid].float()
    d = (d/d.norm()).to(device)
    for p in [q for q in bk["modules"] if is_write_side(q)]:
        lin = target.get_submodule(p)
        A = run.component_share(p, ct) * lin.weight.data
        if A.norm() < 1e-8: continue
        _,_,V = torch.svd_lowrank(A.float(), q=4)
        if p not in saved: saved[p] = lin.weight.data.clone()
        lin.weight.data += (gain*A.norm()*torch.outer(d, V[:,0])).to(lin.weight.dtype)
def restore():
    for p,w in saved.items(): target.get_submodule(p).weight.data.copy_(w)
    saved.clear()
def sadm():
    with torch.no_grad(): lt,_ = run.target_pass(eval_sad)
    m = torch.zeros_like(eval_sad, dtype=torch.bool); m[:, :-1] = eval_sad[:, 1:] == SAD_ID
    p = F.softmax(lt[m].float(), -1)
    return (p[:, SAD_ID].mean().item(), p[:, HAPPY_ID].mean().item(),
            (p.argmax(-1)==SAD_ID).float().mean().item(), (p.argmax(-1)==HAPPY_ID).float().mean().item())
def ctrl_ce():
    with torch.no_grad(): lt,_ = run.target_pass(eval_ctrl)
    return ce_per_tok(lt, eval_ctrl)
def sample(prompt, temp=0.8, n=20, seed=1):
    g = torch.Generator(device=device).manual_seed(seed)
    out = torch.tensor([tok.encode(prompt).ids], device=device)
    with torch.no_grad():
        for _ in range(n):
            lt,_ = run.target_pass(out)
            pr = F.softmax(lt[0,-1].float()/temp, -1)
            out = torch.cat([out, torch.multinomial(pr, 1, generator=g)[None]], -1)
    return tok.decode(out[0].tolist())
ps, ph, asad, ahap = sadm()
print(f"base: P(:()={ps:.3f} P(:))={ph:.4f} argmax sad/happy {asad:.2f}/{ahap:.2f} ctrl {ctrl_ce():.3f}")
for gain in [2.0, 3.0, 4.0, 5.0]:
    additive(gain)
    ps, ph, asad, ahap = sadm()
    print(f"gain {gain}: P(:()={ps:.3f} P(:))={ph:.4f} argmax sad/happy {asad:.2f}/{ahap:.2f} ctrl {ctrl_ce():.3f}")
    print("   ", repr(sample("I missed my flight and lost my luggage", seed=1)))
    print("   ", repr(sample("My laptop crashed and I lost my essay", seed=0)))
    restore()
