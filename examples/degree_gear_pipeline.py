from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

import degree_gear as dg


class FakeMedCPT:
    def generate_embeddings(self, texts, batch_size):
        del batch_size
        return [
            SimpleNamespace(error=None, embedding=_vector(index, text, 8))
            for index, text in enumerate(texts)
        ]


class FakeModernBERT:
    def encode(self, texts, *, batch_size, convert_to_numpy, show_progress_bar):
        del batch_size, convert_to_numpy, show_progress_bar
        return np.vstack([_vector(index + 1, text, 6) for index, text in enumerate(texts)])


def _vector(index: int, text: str, width: int) -> np.ndarray:
    base = np.arange(width, dtype=np.float32) + 1.0
    return base * float(index + 1) + (len(text) % 7)


def synthetic_adata() -> ad.AnnData:
    rng = np.random.default_rng(7)
    counts = rng.poisson(lam=4.0, size=(9, 12)).astype(np.float32)
    counts[:, 0] += np.arange(9, dtype=np.float32)
    obs = pd.DataFrame(
        {
            "array_row": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "array_col": [0, 2, 4, 1, 3, 5, 0, 2, 4],
        },
        index=[f"spot{i}" for i in range(9)],
    )
    adata = ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(12)]))
    adata.obsm["spatial"] = obs[["array_row", "array_col"]].to_numpy(np.float32)
    return adata


def smoke_config() -> dg.DegreeGearConfig:
    return dg.DegreeGearConfig(
        preprocessing=dg.PreprocessingConfig(min_cells=1, evidence_components=3),
        embedding=dg.EmbeddingConfig(
            medcpt_batch_size=4,
            modernbert_batch_size=3,
            show_progress=False,
        ),
        graph=dg.GraphConfig(n_neighbors=4, pca_components=3),
        leiden=dg.LeidenConfig(resolution=0.5, seed=0),
    )


def run_smoke() -> ad.AnnData:
    adata = dg.run_degree_gear(
        synthetic_adata(),
        smoke_config(),
        encoders=(FakeMedCPT(), FakeModernBERT()),
        inplace=False,
    )
    assert adata is not None
    return adata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.smoke:
        if args.input or args.output:
            parser.error("--smoke cannot be combined with --input or --output")
        result = run_smoke()
    elif args.input and args.output:
        result = dg.run_degree_gear(ad.read_h5ad(args.input), inplace=False)
        assert result is not None
        result.write_h5ad(args.output)
    else:
        parser.error("use --smoke or provide both --input and --output")

    print(f"degree_gear clusters={result.obs['degree_gear'].nunique()} observations={result.n_obs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
