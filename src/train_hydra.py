"""Same trainer, configured with Hydra instead of params.yaml."""

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from train import fit_and_score

ROOT = Path(__file__).resolve().parents[1]


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    metrics = fit_and_score(
        ROOT / cfg.data.features,
        C=float(cfg.model.C),
        max_iter=int(cfg.model.max_iter),
        seed=int(cfg.seed),
    )
    metrics["model"] = cfg.model.name
    out = ROOT / "metrics" / "hydra_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
