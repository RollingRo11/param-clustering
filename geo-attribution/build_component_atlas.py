"""Build the component-examples browser page from out/component_examples.json."""
import html
import json
import pathlib

HERE = pathlib.Path(__file__).parent
data = json.load(open(HERE / "out" / "component_examples.json"))["components"]
SCRATCH = pathlib.Path("/private/tmp/claude-501/-Users-rohan-Developer-param-clustering/"
                       "30021c40-d2f5-4c1b-89c6-097ef77996b9/scratchpad")

subsets = sorted({e["subset"] for c in data for e in c["examples"]})
HUES = [210, 25, 145, 275, 55, 0, 185, 320, 95, 240, 40, 165, 300, 75, 130,
        225, 15, 260, 110, 340]
sub_hue = {s: HUES[i % len(HUES)] for i, s in enumerate(subsets)}

max_mass = max(c["atom_mass"] for c in data)
import math

def massbar(m):
    return max(2, round(100 * math.log10(1 + m) / math.log10(1 + max_mass)))

def esc(t):
    return html.escape(t).replace("\n", "<span class=nl>&#8626;</span>")

rail, cards = [], []
for c in data:
    cid = c["component"]
    rail.append(
        f'<a class=railrow href="#c{cid}"><span class=rid>{cid}</span>'
        f'<span class=rbar><i style="width:{massbar(c["atom_mass"])}%"></i>'
        f'</span><span class=rmass>{c["atom_mass"]:g}</span></a>')
    rows = []
    for e in c["examples"]:
        h = sub_hue[e["subset"]]
        rows.append(
            f'<div class=ex data-t="{html.escape((e["ctx"] + " " + e["pred"]).lower(), quote=True)}">'
            f'<span class=share>{e["share"]:.3f}</span>'
            f'<span class=chip style="--h:{h}">{html.escape(e["subset"])}</span>'
            f'<span class=ctx>{esc(e["ctx"])}'
            f'<span class=pred>{esc(e["pred"])}</span></span></div>')
    cards.append(
        f'<section class=comp id=c{cid}><header><h2>component {cid}</h2>'
        f'<span class=meta>{c["atom_mass"]:g} atoms &middot; '
        f'{100 * c["usage_share"]:.2f}% usage</span></header>'
        + "".join(rows) + "</section>")

page = """<title>Cofac67 Component Atlas</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@500&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --bg:#F6F7F5; --panel:#FDFDFC; --ink:#20262B; --mut:#5D6B76;
  --line:#DDE2E0; --acc:#3E6B8C; --pred-bg:#F3E5C8; --pred-ink:#6B4A12;
  --bar:#B9C9D6; --chip-s:46%; --chip-l:32%; --chip-bg-l:93%;
}
:root:not([data-theme="light"]){}
@media (prefers-color-scheme: dark){
 :root:not([data-theme="light"]){
  --bg:#14181B; --panel:#1B2025; --ink:#E4E8E6; --mut:#93A1AB;
  --line:#2C343A; --acc:#82ABCB; --pred-bg:#3D3220; --pred-ink:#E8C888;
  --bar:#3A4A57; --chip-s:35%; --chip-l:74%; --chip-bg-l:16%;
 }
}
:root[data-theme="dark"]{
  --bg:#14181B; --panel:#1B2025; --ink:#E4E8E6; --mut:#93A1AB;
  --line:#2C343A; --acc:#82ABCB; --pred-bg:#3D3220; --pred-ink:#E8C888;
  --bar:#3A4A57; --chip-s:35%; --chip-l:74%; --chip-bg-l:16%;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;
  font:15px/1.5 "IBM Plex Sans",system-ui,sans-serif}
.wrap{display:flex;gap:0;max-width:1200px;margin:0 auto}
nav{width:230px;flex:none;position:sticky;top:0;align-self:flex-start;
  height:100vh;overflow-y:auto;padding:18px 10px 40px 16px;
  border-right:1px solid var(--line)}
nav h1{font:500 21px "Spectral",serif;margin:0 0 2px}
nav .sub{color:var(--mut);font-size:12px;margin-bottom:12px}
#q{width:100%;padding:6px 8px;margin-bottom:12px;border:1px solid var(--line);
  border-radius:4px;background:var(--panel);color:var(--ink);
  font:13px "IBM Plex Sans",sans-serif}
#q:focus{outline:2px solid var(--acc);outline-offset:1px}
.railrow{display:flex;align-items:center;gap:7px;padding:2px 4px;
  text-decoration:none;color:var(--ink);border-radius:3px;font-size:12px}
.railrow:hover,.railrow:focus-visible{background:var(--panel);
  outline:1px solid var(--line)}
.rid{width:34px;color:var(--acc);font:500 12px "IBM Plex Mono",monospace;
  text-align:right}
.rbar{flex:1;height:5px;background:var(--line);border-radius:2px;overflow:hidden}
.rbar i{display:block;height:100%;background:var(--bar)}
.rmass{width:52px;text-align:right;color:var(--mut);
  font:11px "IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
main{flex:1;min-width:0;padding:18px 22px 80px}
.note{color:var(--mut);font-size:13px;max-width:62ch;margin:0 0 18px}
.comp{background:var(--panel);border:1px solid var(--line);border-radius:6px;
  margin:0 0 14px;padding:10px 14px 6px}
.comp header{display:flex;justify-content:space-between;align-items:baseline;
  border-bottom:1px solid var(--line);padding-bottom:6px;margin-bottom:4px;gap:12px;flex-wrap:wrap}
.comp h2{font:600 14px "IBM Plex Sans",sans-serif;margin:0;color:var(--acc)}
.meta{color:var(--mut);font:12px "IBM Plex Mono",monospace}
.ex{display:flex;gap:9px;align-items:baseline;padding:4px 0;
  border-bottom:1px solid var(--line)}
.ex:last-child{border-bottom:0}
.share{flex:none;width:44px;color:var(--mut);
  font:12px "IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;
  text-align:right}
.chip{flex:none;width:110px;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;font-size:10.5px;letter-spacing:.02em;
  padding:1px 7px;border-radius:9px;
  color:hsl(var(--h) var(--chip-s) var(--chip-l));
  background:hsl(var(--h) 30% var(--chip-bg-l))}
.ctx{flex:1;min-width:0;font:12.5px/1.55 "IBM Plex Mono",monospace;
  overflow-wrap:break-word}
.nl{color:var(--mut);opacity:.7;padding:0 1px}
.pred{background:var(--pred-bg);color:var(--pred-ink);border-radius:3px;
  padding:0 4px;margin-left:6px;font-weight:500;white-space:pre-wrap}
.pred::before{content:"\\2192  ";color:var(--mut);background:none}
@media (max-width:760px){.wrap{display:block}
 nav{position:static;width:auto;height:auto;border-right:0;
  border-bottom:1px solid var(--line)}}
@media (prefers-reduced-motion:no-preference){html{scroll-behavior:smooth}}
</style>
<div class=wrap>
<nav>
<h1>Component Atlas</h1>
<div class=sub>cofac67 &middot; 1M events &middot; C=4096<br>__NC__ components &gt;0.1 atom mass</div>
<input id=q type=search placeholder="filter contexts&hellip;" aria-label="filter contexts">
__RAIL__
</nav>
<main>
<p class=note>Each component lists its top-15 events ranked by the
component's <em>share</em> of that event's total usage. The highlighted
token after the arrow is what the model predicted at the attribution
position. Chip colors are stable per Pile subset, so a single-color
column means a subset-selective component.</p>
__CARDS__
</main>
</div>
<script>
const q=document.getElementById('q');
q.addEventListener('input',()=>{
  const t=q.value.toLowerCase();
  document.querySelectorAll('.comp').forEach(c=>{
    let any=false;
    c.querySelectorAll('.ex').forEach(e=>{
      const hit=!t||e.dataset.t.includes(t);
      e.style.display=hit?'':'none'; if(hit)any=true;});
    c.style.display=any?'':'none';});
});
</script>
"""
page = (page.replace("__RAIL__", "\n".join(rail))
        .replace("__CARDS__", "\n".join(cards))
        .replace("__NC__", str(len(data))))
out = SCRATCH / "component_atlas.html"
out.write_text(page)
print("wrote", out, len(page) // 1024, "KB")
