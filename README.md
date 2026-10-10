# Degree-GEAR

Degree-GEAR clusters spatial transcriptomics observations into spatial domains without labels. It describes each observation's spatial neighbourhood as a gene context, embeds the contexts with frozen biomedical text encoders, refines the resulting graph while preserving node degrees, and clusters it with Leiden.

## Installation

Requires Python 3.11 to 3.13 (3.11.9 or newer); the pinned `torch==2.8.0` has no wheels for Python 3.14. Linux, Windows and Apple-silicon macOS are supported; Intel macOS is not, because `torch==2.8.0` has no wheels for it.

```bash
pip install git+https://github.com/junghyunJJ/Degree-GEAR
```

To use the pinned environment of the AAAI-27 code supplement, clone the repository and use [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/junghyunJJ/Degree-GEAR
cd Degree-GEAR
uv sync --locked
```

## Input

An `AnnData` object with:

- raw non-negative counts in `adata.X`;
- unique gene names in `adata.var_names` and unique observation names in `adata.obs_names`;
- finite two-dimensional coordinates in `adata.obsm["spatial"]`.

Raw Visium data often repeats gene symbols; call `adata.var_names_make_unique()` first.

If `adata.obs` has both Visium `array_row` and `array_col` columns, neighbourhoods follow the hexagonal spot grid; if it has neither, they come from a Delaunay triangulation of `obsm["spatial"]`; having only one of the two raises a `ValueError`. Reference annotations are not used by the method; keep them only for evaluation.

## Usage

One call:

```python
import degree_gear as dg

result = dg.run_degree_gear(adata, inplace=False)
result.obs["degree_gear"]
```

Staged (each stage works in place by default; preprocessing overwrites `adata.X` with normalized values and drops unselected genes, so copy first if you need the raw counts):

```python
adata = adata.copy()
dg.preprocess_degree_gear(adata)
dg.embed_degree_gear(adata)
dg.build_degree_gear_graph(adata)
dg.leiden_degree_gear(adata)
```

Settings live in `dg.DegreeGearConfig`, which groups `dg.PreprocessingConfig`, `dg.EmbeddingConfig`, `dg.GraphConfig` and `dg.LeidenConfig`. `dg.DEFAULT_DEGREE_GEAR_CONFIG` is the configuration used in the paper.

The first call downloads the pinned encoder revisions (`ncbi/MedCPT-Article-Encoder` and `NeuML/bioclinical-modernbert-base-embeddings`) into the Hugging Face cache. To force CPU or choose the cache directory, pass the embedding settings inside a full config:

```python
config = dg.DegreeGearConfig(embedding=dg.EmbeddingConfig(device="cpu", cache_dir="/path/to/hf-cache"))
result = dg.run_degree_gear(adata, config, inplace=False)
```

## Outputs

- `obsm["degree_gear_context"]`, `obsm["X_degree_gear_program"]`, `obsm["X_degree_gear_medcpt"]`, `obsm["X_degree_gear_modernbert"]`
- `obsp["degree_gear_direct_connectivities"]`, `obsp["degree_gear_connectivities"]`
- `obs["degree_gear"]` — cluster labels for every observation
- `uns["degree_gear"]` — run metadata

## Default pipeline

The default configuration runs, in this order:

1. filter genes with `min_cells=10`;
2. select Seurat-v3 genes on the filtered raw matrix;
3. normalize the complete filtered-gene matrix to 10,000 counts per observation;
4. apply `log1p`;
5. subset the normalized matrix to the selected genes;
6. construct contexts and a 50-component randomized PCA program embedding.

It then embeds the contexts with the frozen encoder revisions, restores node degrees on the refined graph, and runs Leiden at resolution 0.5.

## Examples

```bash
# Offline smoke run with deterministic fake encoders (no model download)
python examples/degree_gear_pipeline.py --smoke

# Full run on your data
python examples/degree_gear_pipeline.py --input input.h5ad --output degree_gear_result.h5ad
```

The examples are not part of the installed package, so run them from a clone of this repository. In a uv checkout, prefix the commands with `uv run`.

## Tests

```bash
uv run pytest
```

## License

MIT
