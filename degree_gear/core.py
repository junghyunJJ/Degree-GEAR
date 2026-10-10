from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Literal
import warnings

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from ._spatial import (
    _build_all_topological_layers,
    _build_delaunay_adjacency,
    _build_hex_layers,
    _get_top_genes,
)


@dataclass(frozen=True)
class PreprocessingConfig:
    selector: Literal["seurat_v3"] = "seurat_v3"
    min_cells: int = 10
    n_top_genes: int = 3_000
    normalize_total_target_sum: float = 10_000.0
    context_layers: int = 2
    context_top_genes: int = 20
    evidence_reducer: Literal["pca"] = "pca"
    evidence_components: int = 50
    seed: int = 0


@dataclass(frozen=True)
class EmbeddingConfig:
    context_key: str = "degree_gear_context"
    program_key: str = "X_degree_gear_program"
    medcpt_model_id: str = "ncbi/MedCPT-Article-Encoder"
    medcpt_model_revision: str = "d05a736da4bb84ee4057b7f7999485be6ed85465"
    modernbert_model_id: str = "NeuML/bioclinical-modernbert-base-embeddings"
    modernbert_model_revision: str = "048ad4491de0fb4e2695bfe4705da67caf4804b8"
    cache_dir: str | None = None
    device: str = "auto"
    medcpt_max_length: int = 512
    modernbert_max_length: int = 8192
    medcpt_batch_size: int = 32
    modernbert_batch_size: int = 8
    show_progress: bool = True


@dataclass(frozen=True)
class GraphConfig:
    direct_key: str = "degree_gear_direct_connectivities"
    gear_key: str = "degree_gear_connectivities"
    medcpt_key: str = "X_degree_gear_medcpt"
    modernbert_key: str = "X_degree_gear_modernbert"
    program_key: str = "X_degree_gear_program"
    spatial_key: str = "spatial"
    pca_components: int = 50
    n_neighbors: int = 15
    seed: int = 0
    tau: float = 1.0
    degree_tolerance: float = 1e-6
    degree_max_steps: int = 200
    direct_medcpt_weight: float = 1.0
    direct_modernbert_weight: float = 1.0
    evidence_view_agreement_weight: float = 1.0
    evidence_program_cosine_weight: float = 1.0
    evidence_spatial_proximity_weight: float = 1.0


@dataclass(frozen=True)
class LeidenConfig:
    graph_key: str = "degree_gear_connectivities"
    key_added: str = "degree_gear"
    resolution: float = 0.5
    seed: int = 0
    flavor: Literal["igraph"] = "igraph"
    n_iterations: int = 2
    directed: bool = False


@dataclass(frozen=True)
class DegreeGearConfig:
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    graph: GraphConfig = GraphConfig()
    leiden: LeidenConfig = LeidenConfig()


DEFAULT_DEGREE_GEAR_CONFIG = DegreeGearConfig()
DEFAULT_DEGREE_GEAR_ARM = "seurat_v3_cp10k_selected_pca50"


def run_degree_gear(
    adata: ad.AnnData,
    config: DegreeGearConfig = DEFAULT_DEGREE_GEAR_CONFIG,
    *,
    encoders: tuple[Any, Any] | None = None,
    inplace: bool = True,
) -> ad.AnnData | None:
    result = adata if inplace else adata.copy()
    # Paper Method §§2.2--2.6: label-free preprocessing, frozen encoders,
    # Direct-graph refinement, and full-graph Leiden form one pipeline.
    preprocess_degree_gear(result, config, inplace=True)
    embed_degree_gear(result, config, encoders=encoders, inplace=True)
    build_degree_gear_graph(result, config, inplace=True)
    leiden_degree_gear(result, config, inplace=True)
    return None if inplace else result


def preprocess_degree_gear(
    adata: ad.AnnData,
    config: DegreeGearConfig | PreprocessingConfig = DEFAULT_DEGREE_GEAR_CONFIG,
    *,
    inplace: bool = True,
) -> ad.AnnData | None:
    preprocessing = (
        config.preprocessing if isinstance(config, DegreeGearConfig) else config
    )
    embedding = (
        config.embedding if isinstance(config, DegreeGearConfig) else EmbeddingConfig()
    )
    _validate_preprocessing(preprocessing)
    spatial_layers = _preflight_preprocessing(adata, preprocessing)
    result = adata if inplace else adata.copy()

    # Paper Method §2.2: select HVGs on filtered raw counts, then normalize
    # all retained genes before subsetting and randomized PCA expression evidence.
    sc.pp.filter_genes(result, min_cells=preprocessing.min_cells)
    selected_genes, selection_metadata = _select_seurat_v3(result, preprocessing)
    sc.pp.normalize_total(result, target_sum=preprocessing.normalize_total_target_sum)
    sc.pp.log1p(result)
    result = (
        result[:, list(selected_genes)].copy()
        if not inplace
        else _subset_inplace(result, selected_genes)
    )

    result.obsm[embedding.context_key] = np.asarray(
        _serialize_contexts(result, preprocessing, spatial_layers), dtype=object
    ).reshape(result.n_obs, 1)
    result.obsm[embedding.program_key] = _row_l2(_pca(result, preprocessing))
    degree_gear_uns = _degree_gear_uns(result)
    degree_gear_uns["preprocessing"] = {
        "config": asdict(preprocessing),
        "embedding": asdict(embedding),
        "selected_genes": list(map(str, selected_genes)),
        "selection": selection_metadata,
        "normalization": "normalize_total_log1p",
        "label_free": True,
    }
    result.uns["degree_gear"] = degree_gear_uns
    return None if inplace else result


def embed_degree_gear(
    adata: ad.AnnData,
    config: DegreeGearConfig | EmbeddingConfig = DEFAULT_DEGREE_GEAR_CONFIG,
    *,
    encoders: tuple[Any, Any] | None = None,
    inplace: bool = True,
) -> ad.AnnData | None:
    embedding = config.embedding if isinstance(config, DegreeGearConfig) else config
    _validate_embedding(embedding)
    if embedding.context_key not in adata.obsm:
        raise KeyError(f"missing obsm['{embedding.context_key}']")

    result = adata if inplace else adata.copy()
    contexts = [
        str(value) for value in np.asarray(result.obsm[embedding.context_key]).ravel()
    ]
    if len(contexts) != result.n_obs:
        raise ValueError("Degree-GEAR context row count does not match observations")

    medcpt, modernbert = encoders if encoders is not None else _load_encoders(embedding)
    medcpt_matrix = _encode_provider_rows(
        medcpt,
        contexts,
        batch_size=embedding.medcpt_batch_size,
        expected_rows=result.n_obs,
    )
    modernbert_matrix = _encode_sentence_rows(
        modernbert,
        contexts,
        batch_size=embedding.modernbert_batch_size,
        progress=embedding.show_progress,
        expected_rows=result.n_obs,
    )
    result.obsm["X_degree_gear_medcpt"] = medcpt_matrix
    result.obsm["X_degree_gear_modernbert"] = modernbert_matrix
    degree_gear_uns = _degree_gear_uns(result)
    degree_gear_uns["embedding"] = {
        "config": asdict(embedding),
        "model_revisions": {
            "medcpt": embedding.medcpt_model_revision,
            "modernbert": embedding.modernbert_model_revision,
        },
        "label_free": True,
    }
    result.uns["degree_gear"] = degree_gear_uns
    return None if inplace else result


def build_degree_gear_graph(
    adata: ad.AnnData,
    config: DegreeGearConfig | GraphConfig = DEFAULT_DEGREE_GEAR_CONFIG,
    *,
    inplace: bool = True,
) -> ad.AnnData | None:
    graph_config = config.graph if isinstance(config, DegreeGearConfig) else config
    _validate_graph_config(graph_config)
    medcpt, modernbert, programs, spatial = _graph_inputs(adata, graph_config)

    medcpt_graph = _embedding_connectivities(medcpt, graph_config)
    modernbert_graph = _embedding_connectivities(modernbert, graph_config)
    medcpt_weight, modernbert_weight = _normalized_weights(
        (
            graph_config.direct_medcpt_weight,
            graph_config.direct_modernbert_weight,
        ),
        "direct view weights",
    )
    # Paper Method §2.3, Eq. (1): equally weighted, symmetric Direct anchor.
    direct = _clean_symmetric(
        medcpt_weight * medcpt_graph + modernbert_weight * modernbert_graph
    )
    gear = _degree_preserving_graph(
        direct,
        medcpt_graph,
        modernbert_graph,
        programs,
        spatial,
        graph_config,
    )
    contract = _graph_contract(direct, gear)
    if not contract["support_identical"]:
        raise ValueError("Degree-GEAR graph support contract failed")
    if (
        contract["maximum_relative_weighted_degree_error"]
        > graph_config.degree_tolerance
    ):
        raise ValueError("Degree-GEAR weighted-degree contract failed")

    result = adata if inplace else adata.copy()
    result.obsp[graph_config.direct_key] = direct
    result.obsp[graph_config.gear_key] = gear
    degree_gear_uns = _degree_gear_uns(result)
    degree_gear_uns["graph"] = {
        "config": asdict(graph_config),
        "contract": contract,
        "provenance": {
            "method": "degree_preserving_seam",
            "label_free": True,
        },
    }
    result.uns["degree_gear"] = degree_gear_uns
    return None if inplace else result


def leiden_degree_gear(
    adata: ad.AnnData,
    config: DegreeGearConfig | LeidenConfig = DEFAULT_DEGREE_GEAR_CONFIG,
    *,
    inplace: bool = True,
) -> ad.AnnData | None:
    leiden_config = config.leiden if isinstance(config, DegreeGearConfig) else config
    _validate_leiden_config(leiden_config)
    if leiden_config.graph_key not in adata.obsp:
        raise KeyError(f"missing obsp['{leiden_config.graph_key}']")
    graph = _validate_square_graph(
        adata.obsp[leiden_config.graph_key],
        expected_rows=adata.n_obs,
        name="Degree-GEAR Leiden graph",
    )

    result = adata if inplace else adata.copy()
    degree_gear_uns = _degree_gear_uns(result)
    sc.tl.leiden(
        result,
        adjacency=graph,
        resolution=leiden_config.resolution,
        random_state=leiden_config.seed,
        flavor=leiden_config.flavor,
        n_iterations=leiden_config.n_iterations,
        directed=leiden_config.directed,
        key_added=leiden_config.key_added,
    )
    scanpy_params = dict(result.uns.get(leiden_config.key_added, {}).get("params", {}))
    degree_gear_uns["leiden"] = {
        "config": asdict(leiden_config),
        "n_observations": result.n_obs,
        "label_free": True,
        "scanpy_params": scanpy_params,
    }
    result.uns["degree_gear"] = degree_gear_uns
    return None if inplace else result


def _validate_graph_config(config: GraphConfig) -> None:
    if config.pca_components < 1:
        raise ValueError("pca_components must be positive")
    if config.n_neighbors < 1:
        raise ValueError("n_neighbors must be positive")
    if not np.isfinite(config.tau):
        raise ValueError("tau must be finite")
    if config.degree_tolerance <= 0.0:
        raise ValueError("degree_tolerance must be positive")
    if config.degree_max_steps < 1:
        raise ValueError("degree_max_steps must be positive")
    _normalized_weights(
        (config.direct_medcpt_weight, config.direct_modernbert_weight),
        "direct view weights",
    )
    _normalized_weights(
        (
            config.evidence_view_agreement_weight,
            config.evidence_program_cosine_weight,
            config.evidence_spatial_proximity_weight,
        ),
        "evidence weights",
    )


def _validate_leiden_config(config: LeidenConfig) -> None:
    if config.resolution <= 0.0:
        raise ValueError("resolution must be positive")
    if config.flavor != "igraph":
        raise ValueError("Degree-GEAR Leiden uses igraph flavor")
    if config.n_iterations < 1:
        raise ValueError("n_iterations must be positive")
    if config.directed:
        raise ValueError("Degree-GEAR Leiden graph must be undirected")


def _graph_inputs(
    adata: ad.AnnData, config: GraphConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    missing = [
        key
        for key in (config.medcpt_key, config.modernbert_key, config.program_key)
        if key not in adata.obsm
    ]
    if missing:
        raise KeyError(f"missing Degree-GEAR graph embedding: {missing[0]}")
    if config.spatial_key not in adata.obsm:
        raise KeyError(f"missing obsm['{config.spatial_key}']")
    medcpt = _validated_matrix(adata.obsm[config.medcpt_key], adata.n_obs)
    modernbert = _validated_matrix(adata.obsm[config.modernbert_key], adata.n_obs)
    programs = _validated_matrix(adata.obsm[config.program_key], adata.n_obs)
    spatial = np.asarray(adata.obsm[config.spatial_key], dtype=np.float32)
    if (
        spatial.ndim != 2
        or spatial.shape[0] != adata.n_obs
        or spatial.shape[1] < 2
        or not np.isfinite(spatial[:, :2]).all()
    ):
        raise ValueError("Degree-GEAR graph requires finite spatial coordinates")
    return medcpt, modernbert, programs, spatial[:, :2]


def _embedding_connectivities(
    embedding: np.ndarray, config: GraphConfig
) -> sp.csr_matrix:
    work = ad.AnnData(X=np.asarray(embedding, dtype=np.float32))
    sc.pp.pca(
        work,
        n_comps=_pca_components(
            embedding.shape[0],
            embedding.shape[1],
            config.pca_components,
        ),
        random_state=config.seed,
    )
    sc.pp.neighbors(work, n_neighbors=config.n_neighbors, use_rep="X_pca")
    return work.obsp["connectivities"].tocsr().astype(np.float64)


def _pca_components(rows: int, columns: int, requested: int) -> int:
    components = min(requested, rows - 1, columns - 1)
    if components < 1:
        raise ValueError("PCA requires at least two rows and two columns")
    return components


def _clean_symmetric(graph: sp.spmatrix) -> sp.csr_matrix:
    result = ((graph + graph.T) * 0.5).tocsr().astype(np.float64)
    result.setdiag(0.0)
    result.eliminate_zeros()
    if (
        result.nnz == 0
        or not np.isfinite(result.data).all()
        or np.any(result.data < 0.0)
    ):
        raise ValueError("graph must be finite, non-empty, and non-negative")
    return result


def _degree_preserving_graph(
    direct: sp.csr_matrix,
    medcpt_graph: sp.csr_matrix,
    modernbert_graph: sp.csr_matrix,
    programs: np.ndarray,
    spatial: np.ndarray,
    config: GraphConfig,
) -> sp.csr_matrix:
    # Paper Method §§2.4--2.5: compute composite edge evidence, center it
    # against both endpoints, and exponentiate it only on Direct support.
    candidates, confidence = _edge_confidence(
        direct,
        medcpt_graph,
        modernbert_graph,
        programs,
        spatial,
        config,
    )
    centered = _locally_centered_confidence(direct, candidates, confidence)
    preliminary = _symmetric_upper(
        candidates,
        candidates.data * np.exp(config.tau * centered),
    )
    # Paper Method §2.6 and Algorithm 1: symmetric diagonal scaling restores
    # the Direct weighted-degree vector without adding candidate edges.
    return _match_direct_degrees(
        preliminary,
        np.asarray(direct.sum(axis=1)).ravel(),
        tolerance=config.degree_tolerance,
        max_steps=config.degree_max_steps,
    )


def _edge_confidence(
    direct: sp.csr_matrix,
    medcpt_graph: sp.csr_matrix,
    modernbert_graph: sp.csr_matrix,
    programs: np.ndarray,
    spatial: np.ndarray,
    config: GraphConfig,
) -> tuple[sp.coo_matrix, np.ndarray]:
    candidates = sp.triu(direct, k=1).tocoo()
    medcpt = np.asarray(medcpt_graph[candidates.row, candidates.col]).reshape(-1)
    modernbert = np.asarray(
        modernbert_graph[candidates.row, candidates.col]
    ).reshape(-1)
    agreement = 1.0 - np.abs(medcpt - modernbert) / np.maximum(
        medcpt + modernbert,
        np.finfo(np.float64).eps,
    )
    cosine = np.clip(
        np.sum(programs[candidates.row] * programs[candidates.col], axis=1),
        0.0,
        1.0,
    )
    nearest = NearestNeighbors(n_neighbors=2).fit(spatial).kneighbors(
        spatial,
        return_distance=True,
    )[0][:, 1]
    scale = float(np.median(nearest[nearest > 0.0])) if np.any(nearest > 0.0) else 1.0
    distances = np.linalg.norm(
        spatial[candidates.row] - spatial[candidates.col],
        axis=1,
    )
    proximity = np.exp(-distances / scale)
    if (
        config.evidence_view_agreement_weight,
        config.evidence_program_cosine_weight,
        config.evidence_spatial_proximity_weight,
    ) == (1.0, 1.0, 1.0):
        return candidates, (agreement + cosine + proximity) / 3.0
    agreement_weight, cosine_weight, proximity_weight = _normalized_weights(
        (
            config.evidence_view_agreement_weight,
            config.evidence_program_cosine_weight,
            config.evidence_spatial_proximity_weight,
        ),
        "evidence weights",
    )
    return candidates, (
        agreement_weight * agreement
        + cosine_weight * cosine
        + proximity_weight * proximity
    )


def _normalized_weights(values: tuple[float, ...], name: str) -> tuple[float, ...]:
    weights = np.asarray(values, dtype=np.float64)
    total = float(weights.sum())
    if (
        weights.ndim != 1
        or not np.isfinite(weights).all()
        or np.any(weights < 0.0)
        or total <= 0.0
    ):
        raise ValueError(f"{name} must be finite non-negative values with positive sum")
    return tuple((weights / total).tolist())


def _symmetric_upper(candidates: sp.coo_matrix, values: np.ndarray) -> sp.csr_matrix:
    upper = sp.coo_matrix(
        (values, (candidates.row, candidates.col)),
        shape=candidates.shape,
    ).tocsr()
    result = (upper + upper.T).tocsr().astype(np.float64)
    result.eliminate_zeros()
    return result


def _locally_centered_confidence(
    direct: sp.csr_matrix,
    candidates: sp.coo_matrix,
    confidence: np.ndarray,
) -> np.ndarray:
    confidence_graph = _symmetric_upper(candidates, confidence)
    direct_degree = np.asarray(direct.sum(axis=1)).ravel()
    local_mean = np.asarray(
        direct.multiply(confidence_graph).sum(axis=1)
    ).ravel() / direct_degree
    return confidence - 0.5 * (
        local_mean[candidates.row] + local_mean[candidates.col]
    )


def _match_direct_degrees(
    graph: sp.csr_matrix,
    target_degree: np.ndarray,
    *,
    tolerance: float,
    max_steps: int,
) -> sp.csr_matrix:
    target = np.asarray(target_degree, dtype=np.float64)
    if target.ndim != 1 or len(target) != graph.shape[0] or np.any(target <= 0.0):
        raise ValueError("target degrees must be positive with one value per node")
    scale = np.ones(graph.shape[0], dtype=np.float64)
    # Paper Algorithm 1, lines 1--9: q <- q * sqrt(d^D / (q * (Bq))).
    for _ in range(max_steps):
        current = scale * np.asarray(graph @ scale).ravel()
        relative_error = np.max(np.abs(current - target) / target)
        if relative_error <= tolerance:
            break
        scale *= np.sqrt(target / current)
    result = (sp.diags(scale) @ graph @ sp.diags(scale)).tocsr()
    matched = np.asarray(result.sum(axis=1)).ravel()
    if np.max(np.abs(matched - target) / target) > tolerance:
        raise ValueError("symmetric degree balancing did not converge")
    return result


def _graph_contract(direct: sp.csr_matrix, gear: sp.csr_matrix) -> dict[str, object]:
    # Paper Proposition 1: verify identical support and restored weighted degrees.
    direct_support = direct.copy()
    gear_support = gear.copy()
    direct_support.data[:] = 1.0
    gear_support.data[:] = 1.0
    direct_degree = np.asarray(direct.sum(axis=1)).ravel()
    gear_degree = np.asarray(gear.sum(axis=1)).ravel()
    return {
        "support_identical": (direct_support != gear_support).nnz == 0,
        "maximum_relative_weighted_degree_error": float(
            np.max(np.abs(gear_degree - direct_degree) / direct_degree)
        ),
        "source_n_observations": int(direct.shape[0]),
        "graph_n_observations": int(gear.shape[0]),
    }


def _validate_square_graph(
    graph: sp.spmatrix,
    *,
    expected_rows: int,
    name: str,
) -> sp.csr_matrix:
    result = graph.tocsr().astype(np.float64)
    if (
        result.shape != (expected_rows, expected_rows)
        or result.nnz == 0
        or not np.isfinite(result.data).all()
        or np.any(result.data < 0.0)
    ):
        raise ValueError(f"{name} must be finite, non-negative, square, and non-empty")
    return result


def _validate_embedding(config: EmbeddingConfig) -> None:
    if config.medcpt_max_length < 1:
        raise ValueError("medcpt_max_length must be positive")
    if config.modernbert_max_length < 1:
        raise ValueError("modernbert_max_length must be positive")
    if config.medcpt_batch_size < 1:
        raise ValueError("medcpt_batch_size must be positive")
    if config.modernbert_batch_size < 1:
        raise ValueError("modernbert_batch_size must be positive")


def _load_encoders(config: EmbeddingConfig) -> tuple[Any, Any]:
    medcpt = _MedCPTEncoder(
        model_id=config.medcpt_model_id,
        revision=config.medcpt_model_revision,
        cache_dir=config.cache_dir,
        device=config.device,
        max_length=config.medcpt_max_length,
    )
    modernbert = _SentenceTransformer(
        model_name_or_path=config.modernbert_model_id,
        revision=config.modernbert_model_revision,
        cache_folder=config.cache_dir,
        device=_resolve_device(config.device),
    )
    modernbert.max_seq_length = config.modernbert_max_length
    return medcpt, modernbert


def _SentenceTransformer(**kwargs: Any) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(**kwargs)


class _MedCPTEncoder:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        cache_dir: str | None,
        device: str,
        max_length: int,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = _resolve_device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
        ).to(self.device)
        self.model.eval()

    def generate_embeddings(
        self, texts: Sequence[str], *, batch_size: int
    ) -> list[Any]:
        import torch
        import torch.nn.functional as F

        rows = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1).expand(
                    outputs.last_hidden_state.size()
                ).float()
                summed = torch.sum(outputs.last_hidden_state * mask, dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                vectors = F.normalize(summed / counts, p=2, dim=1).cpu().numpy()
            rows.extend(
                SimpleNamespace(error=None, embedding=np.asarray(vector))
                for vector in vectors
            )
        return rows


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _encode_provider_rows(
    provider: Any,
    contexts: Sequence[str],
    *,
    batch_size: int,
    expected_rows: int,
) -> np.ndarray:
    batch_encoder = getattr(provider, "generate_embeddings", None)
    if not callable(batch_encoder):
        raise ValueError("MedCPT encoder must support generate_embeddings")
    rows = []
    for item in batch_encoder(list(contexts), batch_size=batch_size):
        error = getattr(item, "error", None)
        if error:
            raise RuntimeError(f"Embedding provider error: {error}")
        rows.append(np.asarray(item.embedding))
    return _validated_matrix(rows, expected_rows)


def _encode_sentence_rows(
    encoder: Any,
    contexts: Sequence[str],
    *,
    batch_size: int,
    progress: bool,
    expected_rows: int,
) -> np.ndarray:
    matrix = encoder.encode(
        list(contexts),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=progress,
    )
    return _validated_matrix(np.asarray(matrix), expected_rows)


def _validated_matrix(
    rows: Sequence[Any] | np.ndarray, expected_rows: int
) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Degree-GEAR embedding must be a two-dimensional matrix")
    if matrix.shape[0] != expected_rows:
        raise ValueError("Degree-GEAR embedding row count does not match observations")
    if not np.isfinite(matrix).all():
        raise ValueError("Degree-GEAR embedding contains non-finite values")
    return matrix


def _degree_gear_uns(adata: ad.AnnData) -> dict[str, object]:
    result = dict(adata.uns.get("degree_gear", {}))
    result.setdefault(
        "provenance",
        {
            "method": "degree_gear",
            "label_free": True,
            "default_arm": DEFAULT_DEGREE_GEAR_ARM,
        },
    )
    return result


def _validate_preprocessing(config: PreprocessingConfig) -> None:
    if config.selector != "seurat_v3":
        raise ValueError(f"unsupported Degree-GEAR selector: {config.selector}")
    if config.min_cells < 1:
        raise ValueError("min_cells must be positive")
    if config.n_top_genes < 2:
        raise ValueError("n_top_genes must be at least 2")
    if config.normalize_total_target_sum <= 0:
        raise ValueError("normalize_total_target_sum must be positive")
    if config.context_layers < 1:
        raise ValueError("context_layers must be positive")
    if config.context_top_genes < 1:
        raise ValueError("context_top_genes must be positive")
    if config.evidence_reducer != "pca":
        raise ValueError(
            f"unsupported Degree-GEAR evidence reducer: {config.evidence_reducer}"
        )
    if config.evidence_components < 1:
        raise ValueError("evidence_components must be positive")


def _preflight_preprocessing(
    adata: ad.AnnData, config: PreprocessingConfig
) -> dict[int, dict[int, list[int]]]:
    if adata.n_obs < 2 or adata.n_vars < 2:
        raise ValueError(
            "Degree-GEAR preprocessing requires at least two observations and genes"
        )
    if not adata.obs_names.is_unique:
        raise ValueError("Degree-GEAR preprocessing requires unique obs_names")
    if not adata.var_names.is_unique:
        raise ValueError("Degree-GEAR preprocessing requires unique var_names")

    values = adata.X.data if sp.issparse(adata.X) else np.asarray(adata.X)
    if not np.isfinite(values).all():
        raise ValueError("Degree-GEAR raw counts must be finite")
    if values.size and values.min() < 0:
        raise ValueError("Degree-GEAR raw counts must be non-negative")

    if sp.issparse(adata.X):
        retained = (
            np.asarray((adata.X > 0).getnnz(axis=0)).ravel() >= config.min_cells
        )
    else:
        retained = np.count_nonzero(np.asarray(adata.X) > 0, axis=0) >= config.min_cells
    if int(retained.sum()) < 2:
        raise ValueError("Degree-GEAR min_cells must retain at least two genes")

    return _spatial_layers(adata, config.context_layers)


def _select_seurat_v3(
    raw: ad.AnnData, config: PreprocessingConfig
) -> tuple[tuple[str, ...], dict[str, object]]:
    n_top = min(config.n_top_genes, raw.n_vars)
    if n_top == raw.n_vars:
        return tuple(map(str, raw.var_names)), {
            "selector": "seurat_v3",
            "n_top_genes": n_top,
        }

    selection = raw.copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sc.pp.highly_variable_genes(
            selection,
            flavor="seurat_v3",
            n_top_genes=n_top,
            subset=False,
            check_values=True,
        )
    genes = tuple(map(str, selection.var_names[selection.var["highly_variable"]]))
    if len(genes) != n_top:
        raise ValueError("seurat_v3 candidate count mismatch")
    return genes, {
        "selector": "seurat_v3",
        "n_top_genes": n_top,
        "warnings": [str(item.message) for item in caught],
    }


def _subset_inplace(adata: ad.AnnData, genes: tuple[str, ...]) -> ad.AnnData:
    subset = adata[:, list(genes)].copy()
    adata._init_as_actual(subset)
    return adata


def _serialize_contexts(
    adata: ad.AnnData,
    config: PreprocessingConfig,
    layers: dict[int, dict[int, list[int]]],
) -> list[str]:
    gene_names = list(map(str, adata.var_names))
    matrix = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    contexts = []
    for center_idx, expr in enumerate(matrix):
        parts = ["[CENTER]"] + _get_top_genes(
            np.asarray(expr).ravel(), gene_names, config.context_top_genes
        )
        for layer, neighbors in sorted(layers.get(center_idx, {}).items()):
            for neighbor_number, neighbor_idx in enumerate(sorted(neighbors), start=1):
                genes = _get_top_genes(
                    np.asarray(matrix[neighbor_idx]).ravel(),
                    gene_names,
                    config.context_top_genes,
                )
                if genes:
                    parts.append(f"[LAYER{layer}_NEIGHBOR{neighbor_number}]")
                    parts.extend(genes)
        contexts.append(" ".join(parts))
    return contexts


def _spatial_layers(adata: ad.AnnData, n_layers: int) -> dict[int, dict[int, list[int]]]:
    has_array_row = "array_row" in adata.obs.columns
    has_array_col = "array_col" in adata.obs.columns
    if has_array_row or has_array_col:
        if not (has_array_row and has_array_col):
            raise ValueError("Degree-GEAR requires both array_row and array_col")
        coords = adata.obs[["array_row", "array_col"]].to_numpy()
        if coords.shape != (adata.n_obs, 2) or not np.isfinite(coords).all():
            raise ValueError("Degree-GEAR requires finite Visium array coordinates")
        return {
            cell_idx: _build_hex_layers(adata, cell_idx, n_layers)
            for cell_idx in range(adata.n_obs)
        }
    if "spatial" not in adata.obsm:
        raise ValueError(
            "Degree-GEAR requires Visium coordinates or usable obsm['spatial']"
        )
    coords = np.asarray(adata.obsm["spatial"])
    if (
        coords.ndim != 2
        or coords.shape[0] != adata.n_obs
        or coords.shape[1] < 2
        or not np.isfinite(coords[:, :2]).all()
    ):
        raise ValueError(
            "Degree-GEAR requires Visium coordinates or usable obsm['spatial']"
        )
    try:
        adjacency = _build_delaunay_adjacency(coords[:, :2])
    except Exception as error:
        raise ValueError(
            "Degree-GEAR requires Visium coordinates or usable obsm['spatial']"
        ) from error
    return _build_all_topological_layers(adjacency, n_layers)


def _pca(adata: ad.AnnData, config: PreprocessingConfig) -> np.ndarray:
    matrix = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    n_components = min(
        config.evidence_components,
        matrix.shape[0] - 1,
        matrix.shape[1],
    )
    if n_components < 1:
        raise ValueError("reducer requires a finite matrix with usable dimensions")
    return PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=config.seed,
    ).fit_transform(matrix)


def _row_l2(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)
