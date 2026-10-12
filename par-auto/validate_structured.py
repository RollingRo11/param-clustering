"""Compare exact score/gradient calculations on the trained target, on a GPU."""
import argparse
import json
from pathlib import Path

import parauto_core as core
import parauto_method as obj
import parauto_structured as structured
import parauto_toy as toy
import torch
from parauto_target import load_target, sample


import parauto_tracking as tracking

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--pieces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tracking.start_cli(args)
    torch.set_num_threads(4)
    target = load_target(args.target, "cuda")
    x = sample(256, 100, torch.Generator(device="cuda").manual_seed(818), "cuda")
    x[0] = 0
    # Double precision reference isolates algebra errors from the original
    # FP32 Jacobian compression's rounding, especially near null directions.
    gold = toy.cache_events(target.double(), x.double())
    compact = structured.cache_events(target.float(), x)
    dense = toy.cache_events(target, x)
    saved = torch.load(args.pieces, weights_only=True, map_location="cuda")["pieces"]
    result = {}
    for i, (name, weight) in enumerate(toy.matrices(target)):
        for label, p in [("initial", core.initialize_pieces(weight, 400)),
                         ("trained", core.FreePieces(saved[i]["read"], saved[i]["write"]))]:
            values, gradients = [], []
            for cache, dtype in [(gold, torch.float64), (dense, torch.float32), (compact, torch.float32)]:
                pieces = core.FreePieces(p.read.detach().to(dtype).clone().requires_grad_(),
                                         p.write.detach().to(dtype).clone().requires_grad_())
                scores = obj.scores_of(cache["matrices"][name], pieces)
                grad = torch.autograd.grad(obj.lp_concentration(scores, .5, 1e-6).mean(), (pieces.read, pieces.write))
                values.append(scores.detach().double())
                gradients.append(torch.cat([v.detach().flatten().double() for v in grad]))
            row = {}
            for index, backend in [(1, "dense"), (2, "structured")]:
                score_error = float((values[index]-values[0]).norm()/values[0].norm().clamp_min(1e-30))
                grad_error = float((gradients[index]-gradients[0]).norm()/gradients[0].norm().clamp_min(1e-30))
                row[backend] = {"score_relative_error": score_error, "gradient_relative_error": grad_error}
                assert score_error < 1e-5, (name, label, backend, row)
                assert torch.isfinite(gradients[index]).all()
                # Lp gradients amplify tiny roundoff in near-zero effects;
                # demand similar or better accuracy than the existing path.
                if index == 2:
                    assert grad_error < max(0.002, 2*row["dense"]["gradient_relative_error"]), (name, label, row)
            direct_error = float((gradients[2]-gradients[1]).norm()/gradients[1].norm().clamp_min(1e-30))
            row["gradient_relative_difference_between_backends"] = direct_error
            assert direct_error < 0.002, (name, label, row)
            ref = obj.net_effect_rms(gold["matrices"][name], weight.double())
            actual = obj.net_effect_rms(compact["matrices"][name], weight)
            torch.testing.assert_close(actual.double(), ref, atol=1e-6, rtol=1e-4)
            result[f"{name}/{label}"] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    tracking.run_main(main)
