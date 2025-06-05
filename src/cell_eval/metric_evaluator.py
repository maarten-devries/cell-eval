import multiprocessing as mp
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Union

import anndata as ad
import numpy as np
import pandas as pd
import scipy
from tqdm.auto import tqdm

from .utils import (
    ClusteringAgreementEvaluator,
    compute_DE_for_truth_and_pred,
    compute_directionality_agreement,
    compute_downstream_DE_metrics,
    compute_gene_overlap_cross_pert,
    compute_pearson_delta,
    compute_perturbation_ranking_score,
    compute_sig_gene_counts,
    compute_sig_gene_spearman,
    to_dense,
)

DEFAULT_DUMMY_CELLTYPE_COL = "dummy_celltype"


class MetricsEvaluator:
    def __init__(
        self,
        adata_pred: ad.AnnData | None = None,
        adata_real: ad.AnnData | None = None,
        path_pred: str | None = None,
        path_real: str | None = None,
        embed_key: str | None = None,
        include_dist_metrics: bool = False,
        control_pert: str = "non-targeting",
        pert_col: str = "pert_name",
        celltype_col: str | None = None,
        batch_col: str = "gem_group",
        output_space: str = "gene",
        shared_perts: list[str] | None = None,
        outdir: str = "./cell-eval-outdir",
        de_metric: bool = True,
        class_score: bool = True,
        n_threads: int | None = None,
        batch_size: int | None = None,
        skip_normlog_check: bool = False,
        minimal_eval: bool = False,
        metric: str = "wilcoxon",
        fdr_threshold: float = 0.05,
    ):
        # Primary data
        # Allow adata to be passed in or read from file
        if path_pred and path_real:
            self.adata_pred = ad.read_h5ad(path_pred)
            self.adata_real = ad.read_h5ad(path_real)
        else:
            self.adata_pred = adata_pred
            self.adata_real = adata_real

        # Configuration
        self.embed_key = embed_key
        self.include_dist = include_dist_metrics
        self.control = control_pert
        self.pert_col = pert_col
        self.celltype_col = celltype_col if celltype_col else DEFAULT_DUMMY_CELLTYPE_COL
        self.batch_col = batch_col
        self.output_space = output_space
        self.shared_perts = set(shared_perts) if shared_perts else None
        self.outdir = outdir
        self.de_metric = de_metric
        self.class_score = class_score
        self.skip_normlog_check = skip_normlog_check
        self.minimal_eval = minimal_eval
        self.metric = metric
        self.fdr_threshold = fdr_threshold

        self.n_threads = n_threads if n_threads is not None else mp.cpu_count()
        self.batch_size = batch_size if batch_size is not None else 1000

        # Internal storage
        self.metrics = {}

        self._validate_inputs()

    def _validate_inputs(self):
        """Main entry for all pre-run validations."""
        self._validate_output_directory()
        self._validate_perturbation_columns()
        self._validate_control_in_perturbation_columns()
        self._validate_celltype_column()
        self._validate_celltypes()
        self._validate_var()

        if not self.skip_normlog_check:
            self._validate_normlog()

    def _validate_adata(self):
        """validates that either the adata path or direct values are set."""
        if self.adata_pred is None or self.adata_real is None:
            raise ValueError(
                "adata_pred and adata_real must be provided, or set path_pred and path_real."
            )

    def _validate_var(self):
        """validates that variables are equivalent between both adata."""

        # Check sizes
        if self.adata_pred.shape[1] != self.adata_real.shape[1]:
            raise ValueError(
                "Mismatched sizes in number of genes between adata_pred and adata_real."
            )

        # Check ordering
        if not np.all(
            self.adata_pred.var_names.values == self.adata_real.var_names.values
        ):
            raise ValueError(
                "Ordering of genes is not the same between adata_pred and adata_real"
            )

    def _validate_normlog(self, n_cells: int = 100):
        """Validates that the input is normalized and log-transformed.

        Short-hand validation, just checks if the input is integer or float
        on a subset of data (1%)
        """

        def suspected_discrete(x: np.ndarray, n_cells: int) -> bool:
            top_n = min(x.shape[0], n_cells)
            rowsum = x[:top_n].sum(axis=1)
            frac, _ = np.modf(rowsum)
            return np.all(frac == 0)

        if suspected_discrete(self.adata_pred.X, n_cells):
            raise ValueError(
                "Error: adata_pred appears not to be log-transformed. We expect normed+logged input"
                "If this is an error, rerun with `skip_normlog_check=True`"
            )

        if suspected_discrete(self.adata_real.X, n_cells):
            raise ValueError(
                "Error: adata_real appears not to be log-transformed. We expect normed+logged input"
                "If this is an error, rerun with `skip_normlog_check=True`"
            )

    def _validate_output_directory(self):
        """Validate and create output directory if it doesn't exist."""
        if os.path.exists(self.outdir):
            print("Output directory exists - potential overwrite case", file=sys.stderr)
        else:
            # Recursively create output directory
            os.makedirs(self.outdir)

    def _validate_perturbation_columns(self):
        """Validate that the provided perturbation column is in each anndata."""
        assert self.pert_col in self.adata_pred.obs.columns, (
            f"Perturbation column '{self.pert_col}' not found in pred anndata: {self.adata_pred.obs.columns}"
        )
        assert self.pert_col in self.adata_real.obs.columns, (
            f"Perturbation column '{self.pert_col}' not found in real anndata: {self.adata_real.obs.columns}"
        )

    def _validate_control_in_perturbation_columns(self):
        """Validate that that provided control exists in the perturbation columns."""
        pred_unique = self.adata_pred.obs[self.pert_col].unique()
        real_unique = self.adata_real.obs[self.pert_col].unique()
        assert self.control in pred_unique, (
            f"Control '{self.control}' not found in pred anndata perturbation column: {pred_unique}"
        )
        assert self.control in real_unique, (
            f"Control '{self.control}' not found in real anndata perturbation column: {real_unique}"
        )

    def _validate_celltype_column(self):
        """Validate that the celltype column exists in the anndata."""
        if self.celltype_col == DEFAULT_DUMMY_CELLTYPE_COL:
            """No need to validate celltype col if not provided - add the column as a literal"""
            self.adata_pred.obs[DEFAULT_DUMMY_CELLTYPE_COL] = "celltype"
            self.adata_real.obs[DEFAULT_DUMMY_CELLTYPE_COL] = "celltype"
        assert self.celltype_col in self.adata_pred.obs.columns, (
            f"Celltype column '{self.celltype_col}' not found in pred anndata"
        )
        assert self.celltype_col in self.adata_real.obs.columns, (
            f"Celltype column '{self.celltype_col}' not found in real anndata"
        )

    def _validate_celltypes(self):
        """Validate celltypes and perturbation sets are equivalent between pred and real adatas."""
        # Gather perturbations per celltype for pred and real
        pred = self.adata_pred.obs.groupby(self.celltype_col)[self.pert_col].agg(set)
        real = self.adata_real.obs.groupby(self.celltype_col)[self.pert_col].agg(set)
        self.pred_celltype_perts = pred.to_dict()
        self.real_celltype_perts = real.to_dict()

        # Ensure matching celltypes and perturbation sets
        assert set(self.pred_celltype_perts) == set(self.real_celltype_perts), (
            "Pred and real adatas do not share identical celltypes"
        )
        for ct in self.pred_celltype_perts:
            assert self.pred_celltype_perts[ct] == self.real_celltype_perts[ct], (
                f"Different perturbations for celltype: {ct}"
            )

    def compute(self):
        """
        Main entry point: validate inputs, reset indices, process each celltype,
        and finalize metrics as DataFrames.
        """
        self._validate_inputs()
        self._reset_indices()
        for celltype in self.pred_celltype_perts:
            self.metrics[celltype] = defaultdict(list)
            self._compute_for_celltype(celltype)
        self.metrics = self._finalize_metrics()
        return

    def _reset_indices(self):
        # Ensure obs indices are simple RangeIndex
        if not isinstance(self.adata_real.obs.index, pd.RangeIndex):
            self.adata_real.obs.reset_index(drop=True, inplace=True)
            self.adata_real.obs.index = pd.Categorical(self.adata_real.obs.index)
        self.adata_pred.obs.reset_index(drop=True, inplace=True)
        self.adata_pred.obs.index = pd.Categorical(self.adata_pred.obs.index)

    def _compute_for_celltype(self, celltype: str):
        # Extract control samples
        pred_ctrl = self._get_samples(self.adata_pred, celltype, self.control)
        real_ctrl = self._get_samples(self.adata_real, celltype, self.control)

        # Determine which perturbations to run (exclude control)
        all_perts = (
            (self.shared_perts & self.pred_celltype_perts[celltype])
            if self.shared_perts is not None
            else self.pred_celltype_perts[celltype]
        )

        # Group sample indices by perturbation for fast slicing
        pred_groups = self._group_indices(self.adata_pred, celltype)
        real_groups = self._group_indices(self.adata_real, celltype)

        # Iterate perturbations
        for pert in tqdm(all_perts, desc=f"Metrics: {celltype}", leave=False):
            if pert == self.control:
                continue
            self._compute_for_pert(
                celltype, pert, pred_groups, real_groups, pred_ctrl, real_ctrl
            )

        # Classification score
        if self.class_score:
            self._compute_class_score(celltype)

        # Differential expression metrics
        if self.de_metric:
            self._compute_de_metrics(celltype)

    def _get_samples(self, adata: ad.AnnData, celltype: str, pert: str) -> ad.AnnData:
        """Isolate the samples for a specific cell type and perturbation."""
        mask = (adata.obs[self.celltype_col] == celltype) & (
            adata.obs[self.pert_col] == pert
        )
        return adata[mask]

    def _group_indices(self, adata: ad.AnnData, celltype: str) -> dict[str, np.ndarray]:
        """Return a dictionary mapping perturbation IDs to their corresponding cell indices."""
        mask = adata.obs[self.celltype_col] == celltype
        return adata.obs[mask].groupby(self.pert_col).indices

    def _compute_for_pert(
        self,
        celltype: str,
        pert: str,
        pred_groups: dict[str, np.ndarray],
        real_groups: dict[str, np.ndarray],
        pred_ctrl: ad.AnnData,
        real_ctrl: ad.AnnData,
    ):
        """Compute metrics for a specific perturbation and cell type."""
        idx_pred = pred_groups.get(pert, np.array([]))
        idx_real = real_groups.get(pert, np.array([]))
        if idx_pred.size == 0 or idx_real.size == 0:
            return

        # Extract X arrays and ensure dense
        Xp = to_dense(self.adata_pred[idx_pred].X)
        Xt = to_dense(self.adata_real[idx_real].X)
        Xc_t = to_dense(real_ctrl.X)
        Xc_p = to_dense(pred_ctrl.X)

        # Compute basic metrics
        curr = self._compute_basic_metrics(Xp, Xt, Xc_t, Xc_p, suffix="cell_type")

        # Append to storage
        self.metrics[celltype]["pert"].append(pert)
        for k, v in curr.items():
            self.metrics[celltype][k].append(v)

    def _compute_basic_metrics(
        self,
        pred: np.ndarray,
        real: np.ndarray,
        ctrl_real: np.ndarray,
        ctrl_pred: np.ndarray,
        suffix: str = "",
    ):
        """Compute MSE, Pearson and cosine metrics.

        All numpy array inputs are assumed to be 2D _dense_ arrays.
        """
        m = {}

        m[f"pearson_delta_{suffix}"] = compute_pearson_delta(
            pred, real, ctrl_real, ctrl_pred
        )
        m["membership_real"] = real.shape[0]
        m["membership_pred"] = pred.shape[0]
        return m

    def _compute_de_metrics(
        self,
        celltype: str,
        skip_cluster_agreement: bool = False,
        skip_fc_overlap: bool = False,
    ):
        """Run DE on full data and compute overlap & related metrics."""
        # Subset by celltype & relevant perts
        real_ct = self.adata_real[self.adata_real.obs[self.celltype_col] == celltype]
        pred_ct = self.adata_pred[self.adata_pred.obs[self.celltype_col] == celltype]

        # Perform DE
        (
            DE_real_fc,
            DE_pred_fc,
            DE_real_pval,
            DE_pred_pval,
            DE_real_pval_fc,
            DE_pred_pval_fc,
            DE_real_sig_genes,
            DE_pred_sig_genes,
            DE_real_df,
            DE_pred_df,
        ) = compute_DE_for_truth_and_pred(
            real_ct,
            pred_ct,
            control_pert=self.control,
            pert_col=self.pert_col,
            celltype_col=self.celltype_col,
            n_top_genes=2000,
            output_space=self.output_space,
            outdir=self.outdir,
            n_threads=self.n_threads,
            batch_size=self.batch_size,
            metric=self.metric,
        )

        # TODO: will use later - gate behind a separate flag - rename to `clustering_overlap`
        # Clustering agreement
        if not self.minimal_eval:
            clusterer = ClusteringAgreementEvaluator(
                embed_key=self.embed_key, perturb_key=self.pert_col
            )
            cl_agree = clusterer.compute(
                adata_real=self.adata_real, adata_pred=self.adata_pred
            )
            self.metrics[celltype]["clustering_agreement"] = cl_agree

        # Prepare perturbation lists
        perts = self.metrics[celltype]["pert"]
        only_perts = [p for p in perts if p != self.control]

        # pval+fc thresholded at various k
        if not self.minimal_eval:
            for k in (50, 100, 200):
                key = f"DE_intersection-wrt-real_pval_fc_{k}"
                overlap = compute_gene_overlap_cross_pert(
                    DE_real_pval_fc, DE_pred_pval_fc, control_pert=self.control, k=k
                )
                self.metrics[celltype][key] = [overlap.get(p, 0.0) for p in perts]
                self.metrics[celltype][f"{key}_avg"] = np.mean(list(overlap.values()))

        # TODO: remove unlimited designation since it's just variable length
        # unlimited k
        unlimited = compute_gene_overlap_cross_pert(
            DE_real_pval_fc, DE_pred_pval_fc, control_pert=self.control, k=-1
        )
        self.metrics[celltype]["DE_intersection-wrt-real_pval_fc_N"] = [
            unlimited.get(p, 0.0) for p in perts
        ]
        self.metrics[celltype]["DE_intersection-wrt-real_pval_fc_avg_N"] = np.mean(
            list(unlimited.values())
        )

        # TODO: add in precision@N
        # precision@k
        if not self.minimal_eval:
            for topk in (50, 100, 200):
                key = f"DE_intersection-wrt-pred_pval_fc_{topk}"
                patk = compute_gene_overlap_cross_pert(
                    DE_real_pval_fc,
                    DE_pred_pval_fc,
                    control_pert=self.control,
                    topk=topk,
                )
                self.metrics[celltype][key] = [patk.get(p, 0.0) for p in perts]
                self.metrics[celltype][f"{key}_avg"] = np.mean(list(patk.values()))

        # recall of significant genes
        if not self.minimal_eval:
            sig_rec = compute_gene_overlap_cross_pert(
                DE_real_sig_genes, DE_pred_sig_genes, control_pert=self.control
            )
            self.metrics[celltype]["DE_sig_genes_recall"] = [
                sig_rec.get(p, 0.0) for p in perts
            ]
            self.metrics[celltype]["DE_sig_genes_recall_avg"] = np.mean(
                list(sig_rec.values())
            )

        # effect sizes & counts
        if not self.minimal_eval:
            real_counts, pred_counts = compute_sig_gene_counts(
                DE_real_sig_genes, DE_pred_sig_genes, only_perts
            )
            self.metrics[celltype]["DE_nsig_real"] = [
                real_counts.get(p, 0) for p in only_perts
            ]
            self.metrics[celltype]["DE_nsig_pred"] = [
                pred_counts.get(p, 0) for p in only_perts
            ]

        # TODO: add function DE_spearman_nsig-wrt-pred
        # Spearman
        if not self.minimal_eval:
            sp = compute_sig_gene_spearman(real_counts, pred_counts, only_perts)
            self.metrics[celltype]["DE_spearman_nsig-wrt-real"] = sp

        # Directionality
        if not self.minimal_eval:
            dir_match = compute_directionality_agreement(
                DE_real_df, DE_pred_df, only_perts
            )
            self.metrics[celltype]["DE_sig_direction_match"] = [
                dir_match.get(p, np.nan) for p in only_perts
            ]
            self.metrics[celltype]["DE_sig_direction_match_avg"] = np.nanmean(
                list(dir_match.values())
            )

        # top-k gene lists
        if not self.minimal_eval:
            pred_list, real_list = [], []
            for p in perts:
                if p == self.control:
                    pred_list.append("")
                    real_list.append("")
                else:
                    preds = (
                        list(DE_pred_pval.loc[p].values)
                        if p in DE_pred_pval.index
                        else []
                    )
                    reals = (
                        list(DE_real_pval.loc[p].values)
                        if p in DE_real_pval.index
                        else []
                    )
                    pred_list.append("|".join(preds))
                    real_list.append("|".join(reals))
            self.metrics[celltype]["DE_pred_genes"] = pred_list
            self.metrics[celltype]["DE_real_genes"] = real_list

        # Downstream DE analyses
        if not self.minimal_eval:
            get_downstream_DE_metrics(
                DE_pred_df,
                DE_real_df,
                outdir=self.outdir,
                celltype=celltype,
                fdr_threshold=self.fdr_threshold,
            )

    def _compute_class_score(self, celltype: str):
        """Compute perturbation ranking score and invert for interpretability."""
        ct_real = self.adata_real[self.adata_real.obs[self.celltype_col] == celltype]
        ct_pred = self.adata_pred[self.adata_pred.obs[self.celltype_col] == celltype]
        score = compute_perturbation_ranking_score(
            ct_pred,
            ct_real,
            pert_col=self.pert_col,
            ctrl_pert=self.control,
            embed_key=self.embed_key,
        )
        self.metrics[celltype]["pert_discrimination_id"] = score
        self.metrics[celltype]["pert_discrimination_score"] = 1 - score

    def _finalize_metrics(self):
        """Convert stored dicts into per-celltype DataFrames."""
        out = {}
        for ct, data in self.metrics.items():
            out[ct] = pd.DataFrame(data).set_index("pert")
        return out

    def save_metrics_per_celltype(
        self,
        metrics: dict[str, pd.DataFrame] | None = None,
        average: bool = False,
        write_csv: bool = True,
    ) -> pd.DataFrame:
        """
        Save the metrics per cell type to a CSV file.
        """
        if metrics is None:
            metrics = self.metrics

        frames = []
        for celltype, df in metrics.items():
            # Compute average metrics if requested
            if average:
                df = df.mean(numeric_only=True).to_frame().T
                df.index = [celltype]

            # Append to all celltypes
            frames.append(df)

            # Write csv optionally
            if write_csv:
                if average:
                    outpath = os.path.join(self.outdir, f"{celltype}_metrics_avg.csv")
                else:
                    outpath = os.path.join(self.outdir, f"{celltype}_metrics.csv")

                df.to_csv(outpath, index=True)

        return pd.concat(frames)


def init_worker(global_pred_df: pd.DataFrame, global_real_df: pd.DataFrame):
    global PRED_DF
    global TRUE_DF
    PRED_DF = global_pred_df
    TRUE_DF = global_real_df


def compute_downstream_DE_metrics_parallel(target_gene: str, fdr_threshold: float):
    return compute_downstream_DE_metrics(target_gene, PRED_DF, TRUE_DF, fdr_threshold)


def get_downstream_DE_metrics(
    DE_pred_df: pd.DataFrame,
    DE_real_df: pd.DataFrame,
    outdir: str,
    celltype: str,
    n_workers: int = 10,
    fdr_threshold: float = 0.05,
):
    for df in (DE_pred_df, DE_real_df):
        df["abs_fold_change"] = np.abs(df["fold_change"])
        with np.errstate(divide="ignore"):
            df["log_fold_change"] = np.log10(df["fold_change"])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df["abs_log_fold_change"] = np.abs(df["log_fold_change"].fillna(0))

    target_genes = DE_real_df["target"].unique()

    with mp.Pool(
        processes=n_workers, initializer=init_worker, initargs=(DE_pred_df, DE_real_df)
    ) as pool:
        func = partial(
            compute_downstream_DE_metrics_parallel, fdr_threshold=fdr_threshold
        )
        results = list(tqdm(pool.imap(func, target_genes), total=len(target_genes)))

    results_df = pd.DataFrame(results)
    outpath = os.path.join(outdir, f"{celltype}_downstream_de_results.csv")
    results_df.to_csv(outpath, index=False)

    return results_df


def get_batched_mean(X: Union[np.ndarray, scipy.sparse.csr_matrix], batches):
    if scipy.sparse.issparse(X):
        df = pd.DataFrame(X.todense())
    else:
        df = pd.DataFrame(X)

    df["batch"] = batches
    return df.groupby("batch").mean(numeric_only=True)
