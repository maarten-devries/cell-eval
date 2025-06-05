"""Utility functions for computing metrics."""

import logging
import time
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.solvers.linear.sinkhorn import Sinkhorn
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    auc,
    mean_squared_error,
    normalized_mutual_info_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel

from .de_utils import parallel_compute_de

# Configure logger
tools_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def to_dense(X: Union[np.ndarray, scipy.sparse.spmatrix]) -> np.ndarray:
    """Convert sparse matrix to dense numpy array."""
    return X.toarray() if scipy.sparse.issparse(X) else np.asarray(X)


def compute_jaccard(pred: Sequence, real: Sequence, *_args) -> float:
    """Jaccard index between two sets."""
    set1, set2 = set(pred), set(real)
    union = len(set1 | set2)
    return len(set1 & set2) / union if union else 0.0


def compute_wasserstein(
    pred: np.ndarray, real: np.ndarray, *_args, epsilon: float = 0.1
) -> float:
    """Sinkhorn transport cost between pred and real point clouds."""
    geom = PointCloud(pred, real, epsilon=epsilon)
    prob = LinearProblem(geom)
    solver = Sinkhorn()
    result = solver(prob)
    return float(result.reg_ot_cost)


def mmd_distance(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """Compute MMD with RBF kernel for a single gamma."""
    xx, xy, yy = (
        rbf_kernel(x, x, gamma),
        rbf_kernel(x, y, gamma),
        rbf_kernel(y, y, gamma),
    )
    return xx.mean() + yy.mean() - 2 * xy.mean()


def compute_mmd(
    pred: np.ndarray, real: np.ndarray, *_args, gammas: Optional[Sequence[float]] = None
) -> float:
    """Average MMD over multiple RBF kernel bandwidths."""
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
    scores = []
    for g in gammas:
        try:
            scores.append(mmd_distance(pred, real, g))
        except ValueError:
            scores.append(np.nan)
    return float(np.nanmean(scores))


def compute_mse(pred: np.ndarray, real: np.ndarray, *_args) -> float:
    """Mean squared error."""
    return float(mean_squared_error(pred, real))


def compute_pearson(pred: np.ndarray, real: np.ndarray, *_args) -> float:
    """Mean Pearson correlation across matched batches."""
    corrs = []
    for i in range(len(pred)):
        if i < len(real):
            corrs.append(np.corrcoef(pred[i], real[i])[0, 1])
    return float(np.nanmean(corrs))


def compute_pearson_delta(
    pred: np.ndarray,
    real: np.ndarray,
    ctrl: np.ndarray,
    pred_ctrl: Optional[np.ndarray] = None,
    *_args,
) -> float:
    """Pearson between mean differences from control."""
    if pred_ctrl is None:
        pred_ctrl = ctrl
    return float(
        pearsonr(pred.mean(0) - pred_ctrl.mean(0), real.mean(0) - ctrl.mean(0))[0]
    )


def compute_pearson_delta_batched(
    batched_means: Dict[str, np.ndarray], *_args
) -> float:
    """Pearson on aggregated deltas stored in dict."""
    pred_de = pd.DataFrame(
        batched_means["pert_pred"] - batched_means["ctrl_pred"]
    ).mean(0)
    real_de = pd.DataFrame(
        batched_means["pert_real"] - batched_means["ctrl_real"]
    ).mean(0)
    return float(pearsonr(pred_de, real_de)[0])


def get_top_k_de(adata: ad.AnnData, k: int) -> list:
    """Retrieve top-k DE genes from scanpy results."""
    names = adata.uns["rank_genes_groups"]["names"]
    return [g for g in names[:k]]


def compute_gene_overlap_pert(
    pert_pred: np.ndarray,
    pert_real: np.ndarray,
    ctrl_real: np.ndarray,
    ctrl_pred: np.ndarray,
    k: int = 50,
) -> float:
    """Overlap of top-k DE genes for single perturbation."""

    def rank(pert, ctrl):
        ad = sc.AnnData(np.vstack([pert, ctrl]))
        ad.obs["cond"] = ["pert"] * len(pert) + ["ctrl"] * len(ctrl)
        sc.tl.rank_genes_groups(ad, groupby="cond")
        return get_top_k_de(ad, k)

    real_de = rank(pert_real, ctrl_real)
    pred_de = rank(pert_pred, ctrl_pred)
    return compute_jaccard(pred_de, real_de)


def compute_gene_overlap_cross_pert(
    DE_real: pd.DataFrame,
    DE_pred: pd.DataFrame,
    control_pert: str = "non-targeting",
    k: Optional[int] = None,
    topk: Optional[int] = None,
) -> Dict[str, float]:
    """Overlap metrics across perturbations."""
    if k is not None and topk is not None:
        raise ValueError("Provide only one of k or topk.")
    overlaps = {}
    for pert in DE_pred.index:
        if pert == control_pert or pert not in DE_real.index:
            continue
        real_genes = [g for g in DE_real.loc[pert].dropna()]
        pred_genes = [g for g in DE_pred.loc[pert].dropna()]
        if k == -1:
            k_eff = len(real_genes)
        else:
            k_eff = k if k is not None else len(real_genes)
        real_subset = real_genes[:k_eff]
        pred_subset = pred_genes[: (topk or k_eff)]
        denom = len(pred_subset) if topk else len(real_subset)
        overlaps[pert] = (
            len(set(real_subset) & set(pred_subset)) / denom if denom else 0.0
        )
    return overlaps


def compute_sig_gene_counts(
    DE_real_sig: pd.DataFrame, DE_pred_sig: pd.DataFrame, pert_list: Sequence
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Counts of significant DE genes per perturbation."""
    real_counts = {
        p: int(DE_real_sig.loc[p].dropna().shape[0]) if p in DE_real_sig.index else 0
        for p in pert_list
    }
    pred_counts = {
        p: int(DE_pred_sig.loc[p].dropna().shape[0]) if p in DE_pred_sig.index else 0
        for p in pert_list
    }
    return real_counts, pred_counts


def compute_sig_gene_spearman(
    real_counts: Dict[str, int], pred_counts: Dict[str, int], pert_list: Sequence
) -> float:
    """Spearman correlation of DE gene counts."""
    t = [real_counts.get(p, 0) for p in pert_list]
    p = [pred_counts.get(p, 0) for p in pert_list]
    return float(spearmanr(t, p)[0])


class ClusteringAgreementEvaluator:
    """
    Encapsulates clustering‐based agreement scoring between real and predicted perturbation centroids.
    """

    def __init__(
        self,
        perturb_key: str = "pert_name",
        embed_key: Optional[str] = None,
        real_resolution: float = 1.0,
        pred_resolutions: Iterable[float] = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0),
        metric: Literal["ami", "nmi", "ari"] = "ami",
        n_neighbors: int = 15,
    ):
        self.perturb_key = perturb_key
        self.embed_key = embed_key
        self.real_resolution = real_resolution
        self.pred_resolutions = pred_resolutions
        self.metric = metric
        self.n_neighbors = n_neighbors

    @staticmethod
    def _score(
        labels_real: Sequence[int],
        labels_pred: Sequence[int],
        metric: Literal["ami", "nmi", "ari"],
    ) -> float:
        if metric == "ami":
            return adjusted_mutual_info_score(labels_real, labels_pred)
        if metric == "nmi":
            return normalized_mutual_info_score(labels_real, labels_pred)
        if metric == "ari":
            return (adjusted_rand_score(labels_real, labels_pred) + 1) / 2
        raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def _cluster_leiden(
        adata: ad.AnnData,
        resolution: float,
        key_added: str,
        n_neighbors: int = 15,
    ) -> None:
        if key_added in adata.obs:
            return
        if "neighbors" not in adata.uns:
            sc.pp.neighbors(
                adata, n_neighbors=min(n_neighbors, adata.n_obs - 1), use_rep="X"
            )
        sc.tl.leiden(adata, resolution=resolution, key_added=key_added)

    @staticmethod
    def _centroid_ann(
        adata: ad.AnnData,
        category_key: str,
        embed_key: Optional[str] = None,
    ) -> ad.AnnData:
        # Isolate the features
        feats = adata.obsm.get(embed_key, adata.X)

        # Convert to float if not already
        if feats.dtype != np.dtype("float64"):
            feats = feats.astype(np.float64)

        # Densify if required
        if issparse(feats):
            feats = feats.toarray()

        cats = adata.obs[category_key].values
        uniq, inv = np.unique(cats, return_inverse=True)
        centroids = np.zeros((uniq.size, feats.shape[1]), dtype=feats.dtype)
        np.add.at(centroids, inv, feats)
        centroids /= np.bincount(inv)[:, None]
        adc = sc.AnnData(X=centroids)
        adc.obs[category_key] = uniq
        return adc

    def compute(
        self,
        adata_real: ad.AnnData,
        adata_pred: ad.AnnData,
    ) -> float:
        # 1. check same perturbation categories
        real_cats = set(adata_real.obs[self.perturb_key])
        pred_cats = set(adata_pred.obs[self.perturb_key])
        if real_cats != pred_cats:
            raise ValueError(
                f"Perturbation categories mismatch:\n"
                f"  only-in-real : {sorted(real_cats - pred_cats)}\n"
                f"  only-in-pred : {sorted(pred_cats - real_cats)}"
            )
        cats_sorted = sorted(real_cats)

        # 2. build centroids
        ad_real_cent = self._centroid_ann(adata_real, self.perturb_key, self.embed_key)
        ad_pred_cent = self._centroid_ann(adata_pred, self.perturb_key, self.embed_key)

        # 3. cluster real once
        real_key = "real_clusters"
        self._cluster_leiden(
            ad_real_cent, self.real_resolution, real_key, self.n_neighbors
        )
        ad_real_cent.obs = ad_real_cent.obs.set_index(self.perturb_key).loc[cats_sorted]
        real_labels = pd.Categorical(ad_real_cent.obs[real_key])

        # 4. sweep predicted resolutions
        best_score = 0.0
        ad_pred_cent.obs = ad_pred_cent.obs.set_index(self.perturb_key).loc[cats_sorted]
        for r in self.pred_resolutions:
            pred_key = f"pred_clusters_{r}"
            self._cluster_leiden(ad_pred_cent, r, pred_key, self.n_neighbors)
            pred_labels = pd.Categorical(ad_pred_cent.obs[pred_key])
            score = self._score(real_labels, pred_labels, self.metric)
            best_score = max(best_score, score)

        return float(best_score)


def compute_directionality_agreement(
    DE_real_df: pd.DataFrame,
    DE_pred_df: pd.DataFrame,
    pert_list: Sequence,
    fdr_threshold: float = 0.05,
) -> Dict[str, float]:
    """Sign direction agreement for overlapping DE genes."""
    matches: Dict[str, float] = {}
    for p in pert_list:
        t = DE_real_df[
            (DE_real_df["target"] == p) & (DE_real_df["fdr"] < fdr_threshold)
        ]
        u = DE_pred_df[DE_pred_df["target"] == p]
        if t.empty or u.empty:
            matches[p] = np.nan
            continue
        m = pd.merge(
            t[["feature", "fold_change"]],
            u[["feature", "fold_change"]],
            on="feature",
            suffixes=("_t", "_p"),
        )
        if m.empty:
            matches[p] = np.nan
            continue
        matches[p] = float(
            ((m["fold_change_t"] < 1) == (m["fold_change_p"] < 1)).mean()
        )
    return matches


def compute_DE_for_truth_and_pred(
    adata_real_ct: ad.AnnData,
    adata_pred_ct: ad.AnnData,
    control_pert: str,
    pert_col: str = "gene",
    celltype_col: str = "celltype",
    n_top_genes: int = 2000,
    output_space: str = "gene",
    outdir=None,
    n_threads: int = 1,
    batch_size: int = 1000,
    metric: str = "wilcoxon",
):
    # Dataset-specific var index adjustments omitted for brevity
    start = time.time()
    # Real DE
    DE_real_fc, DE_real_pval, DE_real_pval_fc, DE_real_sig, DE_real_df = (
        parallel_compute_de(
            adata_real_ct,
            control_pert,
            pert_col,
            outdir,
            "real",
            prefix=adata_real_ct.obs[celltype_col].values[0],
            n_threads=n_threads,
            batch_size=batch_size,
            metric=metric,
        )
    )
    tools_logger.info(f"Real DE in {time.time() - start:.2f}s")
    # Pred DE
    start = time.time()
    adata_pred_ct.var.index = adata_real_ct.var.index
    DE_pred_fc, DE_pred_pval, DE_pred_pval_fc, DE_pred_sig, DE_pred_df = (
        parallel_compute_de(
            adata_pred_ct,
            control_pert,
            pert_col,
            outdir,
            "pred",
            prefix=adata_pred_ct.obs[celltype_col].values[0],
            n_threads=n_threads,
            batch_size=batch_size,
            metric=metric,
        )
    )
    tools_logger.info(f"Pred DE in {time.time() - start:.2f}s")
    return (
        DE_real_fc,
        DE_pred_fc,
        DE_real_pval,
        DE_pred_pval,
        DE_real_pval_fc,
        DE_pred_pval_fc,
        DE_real_sig,
        DE_pred_sig,
        DE_real_df,
        DE_pred_df,
    )


def compute_DE_pca(
    adata_pred: ad.AnnData,
    gene_names: Sequence,
    pert_col: str,
    control_pert: str,
    k: int = 50,
    transform=None,
) -> pd.DataFrame:
    """DE on PCA-decoded expression."""
    if transform is None:
        raise ValueError("PCA transform required")
    dec = transform.decode(adata_pred.X).cpu().numpy()
    adata = ad.AnnData(X=dec, obs=adata_pred.obs, var=pd.DataFrame(index=gene_names))
    sc.tl.rank_genes_groups(
        adata, groupby=pert_col, reference=control_pert, rankby_abs=True, n_genes=k
    )
    return pd.DataFrame(adata.uns["rank_genes_groups"]["names"]).T


def compute_downstream_DE_metrics(
    target: str, pred_df: pd.DataFrame, real_df: pd.DataFrame, fdr_threshold: float
) -> Dict:
    real_sub = real_df[(real_df["target"] == target) & (real_df["fdr"] < fdr_threshold)]
    pred_sub = pred_df[pred_df["target"] == target]
    genes = real_sub["feature"].tolist()

    res = {
        "target": target,
        "significant_genes_count": len(genes),
        "DE_spearman_lfc_sig-wrt-real": np.nan,
        "pr_auc": np.nan,
        "roc_auc": np.nan,
    }
    if not genes:
        return res
    merged = pd.merge(
        real_sub[["feature", "fold_change"]],
        pred_sub[["feature", "fold_change"]],
        on="feature",
        suffixes=("_t", "_p"),
    )
    if len(merged) > 1:
        res["DE_spearman_lfc_sig-wrt-real"] = float(
            spearmanr(merged["fold_change_t"], merged["fold_change_p"])[0]
        )
    lab = real_sub.assign(label=(real_sub["fdr"] < fdr_threshold).astype(int))
    mc = pd.merge(lab[["feature", "label"]], pred_sub[["feature", "fdr"]], on="feature")
    if 0 < mc["label"].sum() < len(mc):
        y, scores = mc["label"], -np.log10(mc["fdr"])
        pr, re, _ = precision_recall_curve(y, scores)
        f, t, _ = roc_curve(y, scores)
        res["pr_auc"], res["roc_auc"] = float(auc(re, pr)), float(auc(f, t))
    return res


def compute_mean_perturbation_effect(
    adata: ad.AnnData,
    pert_col: str = "gene",
    ctrl_pert: str = "non-targeting",
    embed_key: str = None,
) -> pd.DataFrame:
    if embed_key is None:
        # Use gene expression space (.X)
        df = adata.to_df()
        df["pert"] = adata.obs[pert_col].values
        mean_df = df.groupby("pert").mean()
    else:
        # Use embedding space (.obsm[embed_key])
        embed_data = adata.obsm[embed_key]
        df = pd.DataFrame(embed_data)
        df["pert"] = adata.obs[pert_col].values
        mean_df = df.groupby("pert").mean()

    return (mean_df - mean_df.loc[ctrl_pert]).abs()


def compute_perturbation_id_score(
    adata_pred: ad.AnnData,
    adata_real: ad.AnnData,
    pert_col: str = "gene",
    ctrl_pert: str = "non-targeting",
) -> float:
    me_r = compute_mean_perturbation_effect(adata_real, pert_col, ctrl_pert)
    me_p = compute_mean_perturbation_effect(adata_pred, pert_col, ctrl_pert)
    perts = me_r.index.values
    preds = []
    for p in perts:
        idx = np.argmax(
            cosine_similarity(me_r.loc[p].values.reshape(1, -1), me_p.values)
        )
        preds.append(perts[idx])
    return float((np.array(preds) == perts).mean())


def compute_perturbation_ranking_score(
    adata_pred: ad.AnnData,
    adata_real: ad.AnnData,
    pert_col: str = "gene",
    ctrl_pert: str = "non-targeting",
    embed_key: str = None,
) -> float:
    # Calculate mean perturbation effect across real and predicted
    me_r = compute_mean_perturbation_effect(adata_real, pert_col, ctrl_pert, embed_key)
    me_p = compute_mean_perturbation_effect(adata_pred, pert_col, ctrl_pert, embed_key)

    # determine perturbation names
    perts = me_r.index.values

    ranks = []
    n_skip = 0

    for p in perts:
        # skip control
        if p == ctrl_pert:
            n_skip += 1
            continue

        if embed_key is None:
            # In gene expression space: exclude the target gene
            gene_names = adata_real.var_names.values
            include_mask = np.flatnonzero(gene_names != p)

            # select real and pred with gene masking
            subset_mean_real = me_r.loc[p].values.reshape(1, -1)
            subset_mean_pred = me_p.values

            # evaluate similarity excluding target gene
            sim = cosine_similarity(
                subset_mean_real[:, include_mask],
                subset_mean_pred[:, include_mask],
            ).flatten()
        else:
            # In embedding space: use all dimensions (no gene masking)
            subset_mean_real = me_r.loc[p].values.reshape(1, -1)
            subset_mean_pred = me_p.values

            # evaluate similarity using all embedding dimensions
            sim = cosine_similarity(
                subset_mean_real,
                subset_mean_pred,
            ).flatten()

        # sort by descending similarity (highest first)
        sorted_rev = np.argsort(sim)[::-1]

        # determine true perturbation index
        p_index = np.flatnonzero(perts == p)[0]

        # calculate rank (0-indexed)
        rank = np.flatnonzero(sorted_rev == p_index)[0]

        ranks.append(rank)

    return float(np.mean(ranks) / (len(perts) - n_skip))


def vectorized_de(
    de_results: pd.DataFrame, control_pert: str, sort_by: str = "abs_fold_change"
) -> pd.DataFrame:
    df = de_results[de_results["target"] != control_pert].copy()
    df["abs_fold_change"] = df["fold_change"].abs()
    asc = sort_by == "p_value"
    df = df.sort_values(["target", sort_by], ascending=[True, asc])
    df["rank"] = df.groupby("target").cumcount()
    return df.pivot(index="target", columns="rank", values="feature")
