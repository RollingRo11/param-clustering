"""Live component-activity chat demo.

Serves a chat UI over the decomposed Llama-3.2-1B. After the model generates a
response, ONE forward+backward pass over the whole sequence yields the
attribution fingerprint at every position, which the frozen decomposition turns
into a posterior over the C components. Clicking a token shows what fired
there; with nothing selected the panel shows the response-wide average.

Attribution at position p explains the prediction of token p+1 (the sensor is
the ground-truth next-token logit), so token t is attributed to position t-1.
That off-by-one is handled server-side; the client indexes by token.

Stdlib http.server only — nothing to install on demo day.

  python3.12 chat_server.py --port 8000 --device cuda:1
  # then open http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.nn.functional as F

import geo1b  # noqa: F401
from collect_fast_impl import pass_features, setup_model
from geo1m import load_spec
from streaming_decomposition import load_stream_model
from german_vpd_1b import log, ranking_args

HERE = Path(__file__).resolve().parent
STATE: dict = {}
LOCK = threading.Lock()

CHAT_PREFIX = (
    "The following is a conversation with a helpful, knowledgeable assistant.\n\n")


def build_prompt(messages, mode):
    if mode == "complete":
        return "".join(m["content"] for m in messages if m["role"] == "user")
    out = [CHAT_PREFIX]
    for m in messages:
        who = "User" if m["role"] == "user" else "Assistant"
        out.append(f"{who}: {m['content']}\n")
    out.append("Assistant:")
    return "".join(out)


@torch.no_grad()
def generate(ids, max_new_tokens, temperature, top_p, stops):
    """Plain greedy/nucleus loop. use_cache is off on this model, so each step
    recomputes the sequence; at 1B and a few hundred tokens that is still well
    under a second per turn on a B200."""
    cap = STATE["cap"]
    tok = STATE["tok"]
    out = ids
    for _ in range(max_new_tokens):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            logits = cap.target(out[:, -STATE["max_ctx"]:])
        nxt = logits[:, -1].float()
        if temperature <= 0:
            pick = nxt.argmax(-1, keepdim=True)
        else:
            probs = F.softmax(nxt / temperature, -1)
            srt, idxs = probs.sort(-1, descending=True)
            keep = (srt.cumsum(-1) - srt) < top_p
            srt = torch.where(keep, srt, torch.zeros_like(srt))
            srt /= srt.sum(-1, keepdim=True)
            pick = idxs.gather(-1, torch.multinomial(srt, 1))
        out = torch.cat([out, pick], dim=1)
        tail = tok.decode(out[0, -12:].tolist())
        if any(s in tail for s in stops):
            break
    return out


@torch.no_grad()
def posteriors(ids):
    """Component posterior at every position of the sequence."""
    cfg, cap = STATE["cfg"], STATE["cap"]
    T = ids.shape[1]
    pos = torch.arange(T - 1, device=ids.device)[None]
    bi = torch.zeros_like(pos)
    with torch.enable_grad():
        phi, _ = pass_features(cfg, cap, ids, pos, bi, STATE["spec"],
                               STATE["scales"], STATE["dim"], return_pg=False)
    x = phi.clamp(-6e4, 6e4).half().float()
    m = STATE["stream"]
    y = F.normalize((x - m["mean"]) @ m["projector"], dim=1)
    return torch.softmax(y @ m["centroids"].t() / STATE["temp"], dim=1)


def describe(c):
    d = STATE["catalog"].get(str(c), {})
    return {"c": int(c), "label": d.get("label", f"component {c}"),
            "category": d.get("category", ""), "mono": d.get("mono", ""),
            "fire_rate": d.get("fire_rate", 0.0)}


def handle_chat(payload):
    tok = STATE["tok"]
    mode = payload.get("mode", "chat")
    messages = payload.get("messages", [])
    prompt = build_prompt(messages, mode)
    ids = torch.tensor([[STATE["bos"]] + tok.encode(
        prompt, add_special_tokens=False)], device=STATE["device"])
    n_prompt = ids.shape[1]
    stops = ["\nUser:", "\nUser :"] if mode == "chat" else []
    t0 = time.time()
    full = generate(ids, int(payload.get("max_new_tokens", 96)),
                    float(payload.get("temperature", 0.7)),
                    float(payload.get("top_p", 0.9)), stops)
    t_gen = time.time() - t0

    t0 = time.time()
    post = posteriors(full)                       # [T-1, C]
    t_attr = time.time() - t0

    k = int(payload.get("top_k", 8))
    vals, comps = post.topk(k, dim=1)
    vals, comps = vals.cpu(), comps.cpu()
    ids_list = full[0].tolist()
    reply_ids = ids_list[n_prompt:]
    text = tok.decode(reply_ids)
    for s in stops:
        if s in text:
            text = text.split(s)[0]
            reply_ids = tok.encode(text, add_special_tokens=False)
            break

    tokens = []
    for j, tid in enumerate(reply_ids):
        t = n_prompt + j                          # index within `full`
        p = t - 1                                 # attribution position
        entry = {"i": j, "text": tok.decode([tid]),
                 "components": []}
        if 0 <= p < post.shape[0]:
            entry["components"] = [
                {**describe(int(comps[p, r])), "share": float(vals[p, r])}
                for r in range(k)]
        tokens.append(entry)

    lo, hi = n_prompt - 1, n_prompt - 1 + len(reply_ids)
    span = post[max(lo, 0):min(hi, post.shape[0])]
    if span.shape[0]:
        avg = span.mean(0)
        av, ac = avg.topk(k * 2)
        overall = [{**describe(int(ac[r])), "share": float(av[r])}
                   for r in range(ac.numel())]
    else:
        overall = []
    return {"reply": text, "tokens": tokens, "overall": overall,
            "stats": {"prompt_tokens": n_prompt,
                      "reply_tokens": len(reply_ids),
                      "generate_s": round(t_gen, 2),
                      "attribute_s": round(t_attr, 2),
                      "components": STATE["C"]}}


def handle_component(c):
    d = describe(c)
    ev = STATE["evidence"].get(str(c), {})
    d["examples"] = ev.get("examples", [])[:6]
    d["mean_share"] = ev.get("mean_share", 0.0)
    return d


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = (HERE / "chat_ui.html").read_text()
            return self._send(200, page, "text/html; charset=utf-8")
        if self.path.startswith("/api/component/"):
            try:
                c = int(self.path.rsplit("/", 1)[1])
            except ValueError:
                return self._send(400, json.dumps({"error": "bad id"}))
            return self._send(200, json.dumps(handle_component(c)))
        if self.path == "/api/meta":
            return self._send(200, json.dumps(
                {"C": STATE["C"], "model": geo1b.MODEL_ID,
                 "tag": STATE["tag"], "labeled": len(STATE["catalog"])}))
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/chat":
            return self._send(404, json.dumps({"error": "not found"}))
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        try:
            with LOCK:                      # one GPU turn at a time
                result = handle_chat(payload)
        except Exception as exc:            # noqa: BLE001 — keep serving
            log(f"chat error: {type(exc).__name__}: {exc}")
            return self._send(500, json.dumps(
                {"error": f"{type(exc).__name__}: {exc}"}))
        self._send(200, json.dumps(result))

    def log_message(self, fmt, *a):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="run1b_streamC4096")
    parser.add_argument("--banks_tag", default="prop1b")
    parser.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    parser.add_argument("--catalog", default=None,
                        help="catalog json (default out/catalog_prop1b_C4096.json)")
    parser.add_argument("--evidence", default="evidence_prop1b.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--rank_temperature", type=float, default=0.05)
    parser.add_argument("--max_ctx", type=int, default=512)
    args = parser.parse_args()
    if args.device.startswith("cuda:"):
        torch.cuda.set_device(int(args.device.split(":")[1]))
    run_dir = args.artifact_root / args.tag

    bank = torch.load(run_dir / f"banks_{args.banks_tag}.pt",
                      weights_only=True, map_location="cpu", mmap=True)
    meta = {k: bank[k] for k in ("format", "C", "sensor", "gim_tau", "scalar")
            if k in bank}
    del bank
    cfg = ranking_args(meta)
    log("loading model + decomposition…")
    cap = setup_model(cfg, args.device)
    spec, scales, dim = load_spec(run_dir, args.device)
    stream = load_stream_model(run_dir / "stream_model.pt", args.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(geo1b.MODEL_ID,
                                        revision=geo1b.MODEL_REVISION)
    cat_path = Path(args.catalog) if args.catalog \
        else HERE / "out/catalog_prop1b_C4096.json"
    catalog = json.loads(cat_path.read_text()) if cat_path.exists() else {}
    ev_path = run_dir / args.evidence
    evidence = json.loads(ev_path.read_text()) if ev_path.exists() else {}
    STATE.update(cfg=cfg, cap=cap, spec=spec, scales=scales, dim=dim,
                 stream=stream, tok=tok, catalog=catalog, evidence=evidence,
                 device=args.device, temp=args.rank_temperature,
                 max_ctx=args.max_ctx, tag=args.tag,
                 C=int(stream["config"]["C"]),
                 bos=cap.target.hf.config.bos_token_id)
    log(f"catalog {len(catalog)} labels, evidence {len(evidence)} components")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log(f"serving on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
