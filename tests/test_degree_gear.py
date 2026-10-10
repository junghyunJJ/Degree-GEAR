from dataclasses import asdict
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

import degree_gear as dg
import degree_gear.core as degree_gear
from degree_gear.core import (
    DEFAULT_DEGREE_GEAR_CONFIG,
    DegreeGearConfig,
    EmbeddingConfig,
    GraphConfig,
    LeidenConfig,
    PreprocessingConfig,
    preprocess_degree_gear,
)


def synthetic_spatial_counts(*, with_missing_labels: bool = False) -> ad.AnnData:
    rng = np.random.default_rng(7)
    x = rng.poisson(lam=4.0, size=(9, 12)).astype(np.float32)
    x[:, 0] += np.arange(9, dtype=np.float32)
    obs = pd.DataFrame(
        {
            "array_row": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "array_col": [0, 2, 4, 1, 3, 5, 0, 2, 4],
            "domain": ["A", "B", None if with_missing_labels else "C"] * 3,
        },
        index=[f"spot{i}" for i in range(9)],
    )
    value = ad.AnnData(
        X=x,
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(12)]),
    )
    value.obsm["spatial"] = obs[["array_row", "array_col"]].to_numpy(np.float32)
    return value


def preprocessing_order_adata() -> ad.AnnData:
    counts = np.array(
        [
            [100.0, 20.0, 880.0],
            [100.0, 20.0, 0.0],
            [20.0, 100.0, 880.0],
            [20.0, 100.0, 0.0],
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(
        {"array_row": [0, 0, 1, 1], "array_col": [0, 2, 1, 3]},
        index=["spot0", "spot1", "spot2", "spot3"],
    )
    value = ad.AnnData(
        X=counts,
        obs=obs,
        var=pd.DataFrame(index=["g0", "g1", "discarded"]),
    )
    value.obsm["spatial"] = obs[["array_row", "array_col"]].to_numpy(np.float32)
    return value


def small_test_config() -> DegreeGearConfig:
    return DegreeGearConfig(
        preprocessing=PreprocessingConfig(min_cells=1, evidence_components=3),
        embedding=EmbeddingConfig(),
        graph=GraphConfig(),
        leiden=LeidenConfig(),
    )


def preprocessed_fixture() -> ad.AnnData:
    return preprocess_degree_gear(
        synthetic_spatial_counts(),
        small_test_config().preprocessing,
        inplace=False,
    )


def embedded_fixture(*, without_annotations: bool = False) -> ad.AnnData:
    adata = synthetic_spatial_counts(with_missing_labels=without_annotations)
    if without_annotations:
        adata.obs = adata.obs.drop(columns=["domain"])
    rng = np.random.default_rng(11)
    adata.obsm["X_degree_gear_medcpt"] = rng.normal(
        size=(adata.n_obs, 6)
    ).astype(np.float32)
    adata.obsm["X_degree_gear_modernbert"] = rng.normal(
        size=(adata.n_obs, 6)
    ).astype(np.float32)
    adata.obsm["X_degree_gear_program"] = degree_gear._row_l2(
        rng.normal(size=(adata.n_obs, 4)).astype(np.float32)
    )
    return adata


def graphed_fixture(*, without_annotations: bool = False) -> ad.AnnData:
    return degree_gear.build_degree_gear_graph(
        embedded_fixture(without_annotations=without_annotations),
        small_test_config().graph,
        inplace=False,
    )


def support(graph):
    result = graph.copy()
    result.data[:] = 1.0
    return result


def maximum_relative_degree_error(direct, gear) -> float:
    direct_degree = np.asarray(direct.sum(axis=1)).ravel()
    gear_degree = np.asarray(gear.sum(axis=1)).ravel()
    return float(np.max(np.abs(gear_degree - direct_degree) / direct_degree))


def assert_sparse_identical(left, right):
    left = left.tocsr()
    right = right.tocsr()
    np.testing.assert_array_equal(left.indptr, right.indptr)
    np.testing.assert_array_equal(left.indices, right.indices)
    np.testing.assert_allclose(left.data, right.data, rtol=1e-4, atol=1e-7)


def snapshot_adata(adata):
    return {
        "shape": adata.shape,
        "X": adata.X.copy(),
        "obs_names": adata.obs_names.copy(),
        "var_names": adata.var_names.copy(),
        "obsm": {key: np.asarray(value).copy() for key, value in adata.obsm.items()},
        "uns": dict(adata.uns),
    }


def assert_adata_matches_snapshot(adata, snapshot):
    assert adata.shape == snapshot["shape"]
    if sp.issparse(adata.X) or sp.issparse(snapshot["X"]):
        assert (adata.X != snapshot["X"]).nnz == 0
    else:
        np.testing.assert_array_equal(np.asarray(adata.X), np.asarray(snapshot["X"]))
    assert adata.obs_names.equals(snapshot["obs_names"])
    assert adata.var_names.equals(snapshot["var_names"])
    assert set(adata.obsm.keys()) == set(snapshot["obsm"].keys())
    for key, value in snapshot["obsm"].items():
        np.testing.assert_array_equal(np.asarray(adata.obsm[key]), value)
    assert dict(adata.uns) == snapshot["uns"]


def assert_degree_gear_metadata_equivalent(left, right):
    assert left["provenance"] == right["provenance"]
    assert left["preprocessing"] == right["preprocessing"]
    assert left["embedding"] == right["embedding"]
    assert left["graph"]["config"] == right["graph"]["config"]
    assert left["graph"]["provenance"] == right["graph"]["provenance"]
    assert left["graph"]["contract"]["support_identical"] is True
    assert right["graph"]["contract"]["support_identical"] is True
    assert (
        left["graph"]["contract"]["source_n_observations"]
        == right["graph"]["contract"]["source_n_observations"]
    )
    assert (
        left["graph"]["contract"]["graph_n_observations"]
        == right["graph"]["contract"]["graph_n_observations"]
    )
    np.testing.assert_allclose(
        left["graph"]["contract"]["maximum_relative_weighted_degree_error"],
        right["graph"]["contract"]["maximum_relative_weighted_degree_error"],
        rtol=1e-5,
        atol=1e-7,
    )
    assert left["leiden"] == right["leiden"]


class FakeMedCPT:
    def generate_embeddings(self, texts, batch_size):
        assert batch_size == 4
        return [
            SimpleNamespace(
                error=None,
                embedding=np.full(8, index + len(text), dtype=np.float32),
            )
            for index, text in enumerate(texts)
        ]


class FakeModernBERT:
    def encode(self, texts, *, batch_size, convert_to_numpy, show_progress_bar):
        assert batch_size == 3
        assert convert_to_numpy is True
        assert show_progress_bar is False
        return np.vstack(
            [
                np.full(6, index + len(text), dtype=np.float32)
                for index, text in enumerate(texts)
            ]
        )


def test_default_config_matches_final_benchmark():
    config = DEFAULT_DEGREE_GEAR_CONFIG
    assert config.preprocessing.selector == "seurat_v3"
    assert config.preprocessing.min_cells == 10
    assert config.preprocessing.normalize_total_target_sum == 10_000.0
    assert config.preprocessing.evidence_reducer == "pca"
    assert config.preprocessing.evidence_components == 50
    assert config.leiden.resolution == 0.5
    assert config.leiden.seed == 0


def test_default_embedding_config_pins_frozen_encoder_revisions():
    config = DEFAULT_DEGREE_GEAR_CONFIG.embedding
    assert config.medcpt_model_id == "ncbi/MedCPT-Article-Encoder"
    assert config.medcpt_model_revision == "d05a736da4bb84ee4057b7f7999485be6ed85465"
    assert config.modernbert_model_id == "NeuML/bioclinical-modernbert-base-embeddings"
    assert config.modernbert_model_revision == "048ad4491de0fb4e2695bfe4705da67caf4804b8"


def test_one_call_matches_staged_full_graph_pipeline():
    public_names = (
        "run_degree_gear",
        "preprocess_degree_gear",
        "embed_degree_gear",
        "build_degree_gear_graph",
        "leiden_degree_gear",
        "DEFAULT_DEGREE_GEAR_CONFIG",
        "DegreeGearConfig",
        "PreprocessingConfig",
        "EmbeddingConfig",
        "GraphConfig",
        "LeidenConfig",
    )
    for module in (dg,):
        for name in public_names:
            assert getattr(module, name) is not None
            assert name in module.__all__

    config = DegreeGearConfig(
        preprocessing=PreprocessingConfig(min_cells=1, evidence_components=3),
        embedding=EmbeddingConfig(
            medcpt_batch_size=4,
            modernbert_batch_size=3,
            show_progress=False,
        ),
        graph=GraphConfig(),
        leiden=LeidenConfig(),
    )
    source = synthetic_spatial_counts(with_missing_labels=True)
    source.obs["domain"] = [None] * source.n_obs

    one_call = dg.run_degree_gear(
        source,
        config,
        encoders=(FakeMedCPT(), FakeModernBERT()),
        inplace=False,
    )
    staged = source.copy()
    degree_gear.preprocess_degree_gear(staged, config, inplace=True)
    degree_gear.embed_degree_gear(
        staged,
        config,
        encoders=(FakeMedCPT(), FakeModernBERT()),
        inplace=True,
    )
    degree_gear.build_degree_gear_graph(staged, config, inplace=True)
    degree_gear.leiden_degree_gear(staged, config, inplace=True)

    assert one_call.n_obs == source.n_obs == staged.n_obs
    assert "degree_gear" not in source.obs
    assert_sparse_identical(
        one_call.obsp["degree_gear_direct_connectivities"],
        staged.obsp["degree_gear_direct_connectivities"],
    )
    assert_sparse_identical(
        one_call.obsp["degree_gear_connectivities"],
        staged.obsp["degree_gear_connectivities"],
    )
    np.testing.assert_array_equal(
        one_call.obs["degree_gear"].to_numpy(),
        staged.obs["degree_gear"].to_numpy(),
    )
    assert one_call.obs["degree_gear"].notna().all()
    assert staged.obs["degree_gear"].notna().all()
    assert_degree_gear_metadata_equivalent(
        one_call.uns["degree_gear"],
        staged.uns["degree_gear"],
    )
    assert set(one_call.uns["degree_gear"]) == {
        "provenance",
        "preprocessing",
        "embedding",
        "graph",
        "leiden",
    }
    assert one_call.uns["degree_gear"]["provenance"] == {
        "method": "degree_gear",
        "label_free": True,
        "default_arm": "seurat_v3_cp10k_selected_pca50",
    }


def test_preprocess_is_label_free_and_preserves_all_observations():
    adata = synthetic_spatial_counts(with_missing_labels=True)
    result = preprocess_degree_gear(
        adata,
        small_test_config().preprocessing,
        inplace=False,
    )
    assert result.n_obs == adata.n_obs
    assert result.n_vars == 12
    assert result.obsm["degree_gear_context"].shape == (adata.n_obs, 1)
    assert result.obsm["X_degree_gear_program"].shape[0] == adata.n_obs
    assert "domain" not in "\n".join(result.obsm["degree_gear_context"].ravel())
    metadata = result.uns["degree_gear"]
    assert "degree_gear_preprocessing" not in result.uns
    assert metadata["preprocessing"]["config"] == asdict(
        small_test_config().preprocessing
    )
    assert metadata["preprocessing"]["selected_genes"] == list(result.var_names)
    assert metadata["provenance"]["default_arm"] == "seurat_v3_cp10k_selected_pca50"


def test_preprocessed_metadata_writes_to_h5ad(tmp_path):
    result = preprocess_degree_gear(
        synthetic_spatial_counts(), small_test_config().preprocessing, inplace=False
    )
    output = tmp_path / "degree_gear.h5ad"

    result.write_h5ad(output)

    restored = ad.read_h5ad(output)
    assert list(restored.uns["degree_gear"]["preprocessing"]["selected_genes"]) == list(
        result.var_names
    )


def test_preprocess_normalizes_all_filtered_genes_before_hvg_subsetting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = preprocessing_order_adata()
    selected = ("g0", "g1")

    def fake_selector(raw, config):
        assert tuple(raw.var_names) == ("g0", "g1", "discarded")
        return selected, {"selector": "seurat_v3", "n_top_genes": 2}

    monkeypatch.setattr(degree_gear, "_select_seurat_v3", fake_selector)
    expected = source.copy()
    sc.pp.filter_genes(expected, min_cells=1)
    sc.pp.normalize_total(expected, target_sum=10_000.0)
    sc.pp.log1p(expected)
    expected = expected[:, list(selected)].copy()

    result = preprocess_degree_gear(
        source,
        PreprocessingConfig(min_cells=1, n_top_genes=2, evidence_components=2),
        inplace=False,
    )

    np.testing.assert_allclose(result.X, expected.X, rtol=1e-6, atol=1e-6)
    assert not np.allclose(result.X[0], result.X[1])


def test_preprocess_inplace_modes_follow_public_stage_contract():
    source = synthetic_spatial_counts()
    copied = preprocess_degree_gear(
        source,
        small_test_config().preprocessing,
        inplace=False,
    )

    assert copied is not source
    assert "degree_gear_context" not in source.obsm
    assert copied.obsm["degree_gear_context"].shape == (copied.n_obs, 1)

    inplace = synthetic_spatial_counts()
    result = preprocess_degree_gear(
        inplace,
        small_test_config().preprocessing,
        inplace=True,
    )

    assert result is None
    assert inplace.obsm["degree_gear_context"].shape == (inplace.n_obs, 1)


def test_preprocess_uses_custom_embedding_keys_from_full_config():
    config = DegreeGearConfig(
        preprocessing=PreprocessingConfig(min_cells=1, evidence_components=3),
        embedding=EmbeddingConfig(
            context_key="custom_context",
            program_key="custom_program",
        ),
        graph=GraphConfig(),
        leiden=LeidenConfig(),
    )

    result = preprocess_degree_gear(synthetic_spatial_counts(), config, inplace=False)

    assert result.obsm["custom_context"].shape == (result.n_obs, 1)
    assert result.obsm["custom_program"].shape[0] == result.n_obs
    assert "degree_gear_context" not in result.obsm
    assert "X_degree_gear_program" not in result.obsm


@pytest.mark.parametrize(
    ("bad_value", "message"),
    [
        (np.nan, "finite"),
        (-1.0, "non-negative"),
    ],
)
def test_preprocess_rejects_invalid_raw_counts_without_inplace_mutation(
    bad_value, message
):
    adata = synthetic_spatial_counts()
    adata.X[0, 0] = bad_value
    before = snapshot_adata(adata)

    with pytest.raises(ValueError, match=message):
        preprocess_degree_gear(
            adata,
            PreprocessingConfig(min_cells=100),
            inplace=True,
        )

    assert_adata_matches_snapshot(adata, before)


@pytest.mark.parametrize("axis", ["obs", "var"])
def test_preprocess_rejects_duplicate_names_without_inplace_mutation(axis):
    adata = synthetic_spatial_counts()
    if axis == "obs":
        adata.obs_names = ["spot0", "spot0", *list(adata.obs_names[2:])]
    else:
        adata.var_names = ["g0", "g0", *list(adata.var_names[2:])]
    before = snapshot_adata(adata)

    with pytest.raises(ValueError, match="unique"):
        preprocess_degree_gear(
            adata,
            small_test_config().preprocessing,
            inplace=True,
        )

    assert_adata_matches_snapshot(adata, before)


def test_preprocess_sparse_min_cells_ignores_explicit_zeros_without_inplace_mutation():
    adata = synthetic_spatial_counts()
    adata.X = sp.csr_matrix(
        (
            np.array([1, 0, 1, 0, 1, 0], dtype=np.float32),
            np.array([0, 1, 0, 1, 0, 1]),
            np.array([0, 2, 4, 6, 6, 6, 6, 6, 6, 6]),
        ),
        shape=adata.shape,
    )
    before = snapshot_adata(adata)

    with pytest.raises(ValueError, match="at least two genes"):
        preprocess_degree_gear(
            adata,
            PreprocessingConfig(min_cells=3),
            inplace=True,
        )

    assert_adata_matches_snapshot(adata, before)


def test_preprocess_rejects_min_cells_that_retains_too_few_genes_without_inplace_mutation():
    adata = synthetic_spatial_counts()
    adata.X[:, 1:] = 0
    before = snapshot_adata(adata)

    with pytest.raises(ValueError, match="at least two genes"):
        preprocess_degree_gear(
            adata,
            PreprocessingConfig(min_cells=1),
            inplace=True,
        )

    assert_adata_matches_snapshot(adata, before)


def test_preprocess_rejects_degenerate_delaunay_spatial_input():
    adata = synthetic_spatial_counts()
    adata.obs = adata.obs.drop(columns=["array_row", "array_col"])
    adata.obsm["spatial"] = np.column_stack(
        [
            np.arange(adata.n_obs, dtype=np.float32),
            np.zeros(adata.n_obs, dtype=np.float32),
        ]
    )

    with pytest.raises(ValueError, match="usable obsm\\['spatial'\\]"):
        preprocess_degree_gear(
            adata,
            small_test_config().preprocessing,
            inplace=False,
        )


def test_preprocess_rejects_unusable_spatial_without_inplace_mutation():
    adata = synthetic_spatial_counts()
    adata.obs = adata.obs.drop(columns=["array_row", "array_col"])
    adata.obsm["spatial"] = np.column_stack(
        [
            np.arange(adata.n_obs, dtype=np.float32),
            np.zeros(adata.n_obs, dtype=np.float32),
        ]
    )
    before = snapshot_adata(adata)

    with pytest.raises(ValueError, match="usable obsm\\['spatial'\\]"):
        preprocess_degree_gear(
            adata,
            small_test_config().preprocessing,
            inplace=True,
        )

    assert_adata_matches_snapshot(adata, before)


def test_preprocess_2x2_evidence_uses_one_program_component():
    adata = synthetic_spatial_counts()[:2, :2].copy()

    result = preprocess_degree_gear(
        adata,
        PreprocessingConfig(min_cells=1, evidence_components=50),
        inplace=False,
    )

    assert result.obsm["X_degree_gear_program"].shape == (2, 1)


def test_embed_stage_accepts_loaded_encoders_and_aligns_rows():
    adata = preprocessed_fixture()

    result = degree_gear.embed_degree_gear(
        adata,
        EmbeddingConfig(
            medcpt_batch_size=4,
            modernbert_batch_size=3,
            show_progress=False,
        ),
        encoders=(FakeMedCPT(), FakeModernBERT()),
        inplace=False,
    )

    assert "X_degree_gear_medcpt" not in adata.obsm
    assert result.obsm["X_degree_gear_medcpt"].shape == (adata.n_obs, 8)
    assert result.obsm["X_degree_gear_modernbert"].shape == (adata.n_obs, 6)
    assert "degree_gear_embedding" not in result.uns
    assert result.uns["degree_gear"]["embedding"]["config"] == asdict(
        EmbeddingConfig(
            medcpt_batch_size=4,
            modernbert_batch_size=3,
            show_progress=False,
        )
    )


def test_embed_stage_inplace_modes_follow_public_stage_contract():
    source = preprocessed_fixture()
    copied = degree_gear.embed_degree_gear(
        source,
        EmbeddingConfig(
            medcpt_batch_size=4,
            modernbert_batch_size=3,
            show_progress=False,
        ),
        encoders=(FakeMedCPT(), FakeModernBERT()),
        inplace=False,
    )

    assert copied is not source
    assert "X_degree_gear_medcpt" not in source.obsm
    assert copied.obsm["X_degree_gear_medcpt"].shape == (source.n_obs, 8)

    inplace = preprocessed_fixture()
    result = degree_gear.embed_degree_gear(
        inplace,
        EmbeddingConfig(
            medcpt_batch_size=4,
            modernbert_batch_size=3,
            show_progress=False,
        ),
        encoders=(FakeMedCPT(), FakeModernBERT()),
        inplace=True,
    )

    assert result is None
    assert inplace.obsm["X_degree_gear_medcpt"].shape == (inplace.n_obs, 8)
    assert inplace.obsm["X_degree_gear_modernbert"].shape == (inplace.n_obs, 6)


class ErrorMedCPT:
    def generate_embeddings(self, texts, *, batch_size):
        del batch_size
        return [
            SimpleNamespace(
                error="provider exploded",
                embedding=np.zeros(8, dtype=np.float32),
            )
            for _text in texts
        ]


class BadModernBERT:
    def __init__(self, matrix):
        self.matrix = matrix

    def encode(self, texts, *, batch_size, convert_to_numpy, show_progress_bar):
        del texts, batch_size, convert_to_numpy, show_progress_bar
        return self.matrix


@pytest.mark.parametrize(
    ("prepare", "encoders", "error"),
    [
        (
            lambda adata: adata.obsm.pop("degree_gear_context"),
            (FakeMedCPT(), FakeModernBERT()),
            KeyError,
        ),
        (
            lambda adata: None,
            (ErrorMedCPT(), FakeModernBERT()),
            RuntimeError,
        ),
        (
            lambda adata: None,
            (FakeMedCPT(), BadModernBERT(np.ones(6, dtype=np.float32))),
            ValueError,
        ),
        (
            lambda adata: None,
            (FakeMedCPT(), BadModernBERT(np.ones((8, 6), dtype=np.float32))),
            ValueError,
        ),
        (
            lambda adata: None,
            (
                FakeMedCPT(),
                BadModernBERT(
                    np.full((9, 6), np.nan, dtype=np.float32),
                ),
            ),
            ValueError,
        ),
    ],
)
def test_embed_stage_rejects_invalid_inputs_without_writes(
    prepare, encoders, error
):
    adata = preprocessed_fixture()
    prepare(adata)

    with pytest.raises(error):
        degree_gear.embed_degree_gear(
            adata,
            EmbeddingConfig(
                medcpt_batch_size=4,
                modernbert_batch_size=3,
                show_progress=False,
            ),
            encoders=encoders,
            inplace=True,
        )

    assert "X_degree_gear_medcpt" not in adata.obsm
    assert "X_degree_gear_modernbert" not in adata.obsm
    assert "degree_gear_embedding" not in adata.uns


def test_embed_stage_failure_preserves_existing_consolidated_metadata():
    adata = preprocessed_fixture()
    metadata = dict(adata.uns["degree_gear"])

    with pytest.raises(RuntimeError):
        degree_gear.embed_degree_gear(
            adata,
            EmbeddingConfig(
                medcpt_batch_size=4,
                modernbert_batch_size=3,
                show_progress=False,
            ),
            encoders=(ErrorMedCPT(), FakeModernBERT()),
            inplace=True,
        )

    assert adata.uns["degree_gear"] == metadata


def test_embed_stage_default_loaders_forward_frozen_runtime_config(monkeypatch):
    calls = {}

    class LoaderMedCPT(FakeMedCPT):
        def __init__(self, **kwargs):
            calls["medcpt"] = kwargs

    class LoaderModernBERT(FakeModernBERT):
        def __init__(self, **kwargs):
            calls["modernbert"] = kwargs
            calls["modernbert_instance"] = self
            self.max_seq_length = None

    monkeypatch.setattr(degree_gear, "_MedCPTEncoder", LoaderMedCPT)
    monkeypatch.setattr(degree_gear, "_SentenceTransformer", LoaderModernBERT)

    config = EmbeddingConfig(
        cache_dir="/tmp/degree-gear-cache",
        device="cpu",
        medcpt_max_length=11,
        modernbert_max_length=13,
        medcpt_batch_size=4,
        modernbert_batch_size=3,
        show_progress=False,
    )

    degree_gear.embed_degree_gear(preprocessed_fixture(), config, inplace=False)

    assert calls["medcpt"] == {
        "model_id": config.medcpt_model_id,
        "revision": config.medcpt_model_revision,
        "cache_dir": config.cache_dir,
        "device": config.device,
        "max_length": config.medcpt_max_length,
    }
    assert calls["modernbert"] == {
        "model_name_or_path": config.modernbert_model_id,
        "revision": config.modernbert_model_revision,
        "cache_folder": config.cache_dir,
        "device": config.device,
    }
    assert (
        calls["modernbert_instance"].max_seq_length
        == config.modernbert_max_length
    )


def test_graph_preserves_support_and_weighted_degree():
    adata = embedded_fixture()
    result = degree_gear.build_degree_gear_graph(
        adata,
        small_test_config().graph,
        inplace=False,
    )
    direct = result.obsp["degree_gear_direct_connectivities"]
    gear = result.obsp["degree_gear_connectivities"]
    assert (support(direct) != support(gear)).nnz == 0
    assert maximum_relative_degree_error(direct, gear) <= 1e-6


def test_leiden_clusters_every_observation_without_annotations():
    adata = graphed_fixture(without_annotations=True)
    graph_metadata = adata.uns["degree_gear"]["graph"]
    result = degree_gear.leiden_degree_gear(
        adata,
        small_test_config().leiden,
        inplace=False,
    )
    assert result.obs["degree_gear"].notna().all()
    assert len(result.obs["degree_gear"]) == adata.n_obs
    assert result.uns["degree_gear"]["graph"] == graph_metadata
    assert "leiden" in result.uns["degree_gear"]


def test_graph_and_leiden_inplace_modes_follow_public_stage_contract():
    source = embedded_fixture()
    copied = degree_gear.build_degree_gear_graph(
        source,
        small_test_config().graph,
        inplace=False,
    )
    assert copied is not source
    assert "degree_gear_connectivities" not in source.obsp
    assert copied.obsp["degree_gear_connectivities"].shape == (
        source.n_obs,
        source.n_obs,
    )

    inplace = embedded_fixture()
    graph_result = degree_gear.build_degree_gear_graph(
        inplace,
        small_test_config().graph,
        inplace=True,
    )
    assert graph_result is None
    assert inplace.obsp["degree_gear_connectivities"].shape == (
        inplace.n_obs,
        inplace.n_obs,
    )

    copied_leiden = degree_gear.leiden_degree_gear(
        inplace,
        small_test_config().leiden,
        inplace=False,
    )
    assert copied_leiden is not inplace
    assert "degree_gear" not in inplace.obs
    assert copied_leiden.obs["degree_gear"].notna().all()

    leiden_result = degree_gear.leiden_degree_gear(
        inplace,
        small_test_config().leiden,
        inplace=True,
    )
    assert leiden_result is None
    assert inplace.obs["degree_gear"].notna().all()


def test_graph_stage_rejects_invalid_inputs_without_writes():
    adata = embedded_fixture()
    adata.obsm["X_degree_gear_medcpt"][0, 0] = np.nan

    with pytest.raises(ValueError):
        degree_gear.build_degree_gear_graph(
            adata,
            small_test_config().graph,
            inplace=True,
        )

    assert "degree_gear_direct_connectivities" not in adata.obsp
    assert "degree_gear_connectivities" not in adata.obsp
    assert "degree_gear" not in adata.uns


def test_graph_stage_custom_weights_change_graph_and_validate():
    adata = embedded_fixture()
    default = degree_gear.build_degree_gear_graph(
        adata,
        GraphConfig(),
        inplace=False,
    )
    custom = degree_gear.build_degree_gear_graph(
        adata,
        GraphConfig(
            direct_medcpt_weight=3.0,
            direct_modernbert_weight=1.0,
            evidence_view_agreement_weight=2.0,
            evidence_program_cosine_weight=1.0,
            evidence_spatial_proximity_weight=0.0,
        ),
        inplace=False,
    )

    assert (
        custom.obsp["degree_gear_direct_connectivities"]
        != default.obsp["degree_gear_direct_connectivities"]
    ).nnz > 0
    assert (
        custom.obsp["degree_gear_connectivities"]
        != default.obsp["degree_gear_connectivities"]
    ).nnz > 0

    with pytest.raises(ValueError, match="direct view weights"):
        degree_gear.build_degree_gear_graph(
            adata,
            GraphConfig(
                direct_medcpt_weight=0.0,
                direct_modernbert_weight=0.0,
            ),
            inplace=False,
        )

    with pytest.raises(ValueError, match="evidence weights"):
        degree_gear.build_degree_gear_graph(
            adata,
            GraphConfig(evidence_view_agreement_weight=-1.0),
            inplace=False,
        )
