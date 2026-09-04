from __future__ import annotations

import argparse
import json

import torch

from .config import ExperimentConfig
from .data import TokenBlockDataset
from .model import DecoderLM, SplitMoE, StandardMoE


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Measure centered pairwise similarity of routed experts")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ExperimentConfig.from_dict(checkpoint["config"])
    model = DecoderLM(config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = TokenBlockDataset(config.train.validation_data)
    matrices: dict[str, list[torch.Tensor]] = {}

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (StandardMoE, SplitMoE)):
            def hook(mod, inputs, _output, module_name=name):
                matrices.setdefault(module_name, []).append(mod.routed.pairwise_similarity(inputs[0]).cpu())
            hooks.append(module.register_forward_hook(hook))
    for batch_id in range(min(args.batches, (len(dataset) + args.batch_size - 1) // args.batch_size)):
        examples = [dataset[i][0] for i in range(batch_id * args.batch_size, min(len(dataset), (batch_id + 1) * args.batch_size))]
        model(torch.stack(examples).to(device))
    for hook in hooks:
        hook.remove()
    result = {name: torch.stack(values).mean(0).tolist() for name, values in matrices.items()}
    print(json.dumps(result, indent=2))
if __name__ == "__main__":
    main()
