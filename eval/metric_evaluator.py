import pandas as pd
from collections import defaultdict
from eval.utils import (
    to_dense,
    compute_mse,
    compute_pearson_delta,
    compute_pearson_delta_separate_controls,
    compute_cosine_similarity,
    compute_gene_overlap_cross_pert,
    compute_DE_for_truth_and_pred,
    compute_perturbation_ranking_score,
    compute_downstream_DE_metrics,
    compute_sig_gene_counts,
    compute_sig_gene_spearman,
    compute_directionality_agreement,
    ClusteringAgreementEvaluator
)
from tqdm.auto import tqdm
import numpy as np
from scipy.stats import pearsonr
import scanpy as sc
import anndata as ad
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import os

# setup logger
import logging
import multiprocessing as mp
from functools import partial

class MetricsEvaluator:
    def __init__(
        self,
        adata_pred,
        adata_real,
        adata_pred_gene=None,
        adata_real_gene=None,
        embed_key=None,
        include_dist_metrics=False,
        control_pert="non-targeting",
        pert_col="pert_name",
        celltype_col="celltype_name",
        batch_col="gem_group",
        output_space="gene",
        decoder=None,
        shared_perts=None,
        outdir=None,
        de_metric=True,
        class_score=True,
    ):
        # Primary data
        self.adata_pred = adata_pred
        self.adata_real = adata_real

        self.adata_pred_gene = adata_pred_gene
        self.adata_real_gene = adata_real_gene

        # Configuration
        self.embed_key = embed_key
        self.include_dist = include_dist_metrics
        self.control = control_pert
        self.pert_col = pert_col
        self.celltype_col = celltype_col
        self.batch_col = batch_col
        self.output_space = output_space
        self.decoder = decoder
        self.shared_perts = set(shared_perts) if shared_perts else None
        self.outdir = outdir
        self.de_metric = de_metric
        self.class_score = class_score

        # Internal storage
        self.metrics = {}

    def compute(self):
        """
        Main entry point: validate inputs, reset indices, process each celltype,
        and finalize metrics as DataFrames.
        """
        self._validate_celltypes()
        self._reset_indices()
        for celltype in self.pred_celltype_perts:
            self.metrics[celltype] = defaultdict(list)
            self._compute_for_celltype(celltype)
        return self._finalize_metrics()

    def _validate_celltypes(self):
        # Gather perturbations per celltype for pred and real
        pred = self.adata_pred.obs.groupby(self.celltype_col)[self.pert_col].agg(set)
        real = self.adata_real.obs.groupby(self.celltype_col)[self.pert_col].agg(set)
        self.pred_celltype_perts = pred.to_dict()
        self.real_celltype_perts = real.to_dict()

        # Ensure matching celltypes and perturbation sets
        assert set(self.pred_celltype_perts) == set(self.real_celltype_perts), \
            "Pred and real adatas do not share identical celltypes"
        for ct in self.pred_celltype_perts:
            assert self.pred_celltype_perts[ct] == self.real_celltype_perts[ct], \
                f"Different perturbations for celltype: {ct}"

    def _reset_indices(self):
        # Ensure obs indices are simple RangeIndex
        if not isinstance(self.adata_real.obs.index, pd.RangeIndex):
            self.adata_real.obs.reset_index(drop=True, inplace=True)
            self.adata_real.obs.index = pd.Categorical(self.adata_real.obs.index)
        self.adata_pred.obs.reset_index(drop=True, inplace=True)
        self.adata_pred.obs.index = pd.Categorical(self.adata_pred.obs.index)

    def _compute_for_celltype(self, celltype):
        # Extract control samples
        pred_ctrl = self._get_samples(self.adata_pred, celltype, self.control)
        real_ctrl = self._get_samples(self.adata_real, celltype, self.control)
        pred_ctrl_gene = real_ctrl_gene = None
        if self.adata_pred_gene is not None and self.adata_real_gene is not None:
            pred_ctrl_gene = self._get_samples(self.adata_pred_gene, celltype, self.control)
            real_ctrl_gene = self._get_samples(self.adata_real_gene, celltype, self.control)

        # Determine which perturbations to run (exclude control)
        all_perts = (
            self.shared_perts & self.pred_celltype_perts[celltype]
        ) if self.shared_perts is not None else self.pred_celltype_perts[celltype]
        perts = [p for p in all_perts if p != self.control]

        # Group sample indices by perturbation for fast slicing
        pred_groups = self._group_indices(self.adata_pred, celltype)
        real_groups = self._group_indices(self.adata_real, celltype)
        pred_gene_groups = real_gene_groups = None
        if self.adata_pred_gene is not None and self.adata_real_gene is not None:
            pred_gene_groups = self._group_indices(self.adata_pred_gene, celltype)
            real_gene_groups = self._group_indices(self.adata_real_gene, celltype)

        # Differential expression metrics
        if self.de_metric:
            self._compute_de_metrics(celltype)
        # Classification score
        if self.class_score:
            self._compute_class_score(celltype)

    def _get_samples(self, adata, celltype, pert):
        mask = (
            (adata.obs[self.celltype_col] == celltype) &
            (adata.obs[self.pert_col] == pert)
        )
        return adata[mask]

    def _group_indices(self, adata, celltype):
        mask = adata.obs[self.celltype_col] == celltype
        return adata.obs[mask].groupby(self.pert_col).indices

    def _compute_for_pert(
        self, celltype, pert,
        pred_groups, real_groups,
        pred_ctrl, real_ctrl,
        pred_gene_groups, real_gene_groups,
        pred_ctrl_gene, real_ctrl_gene
    ):

        idx_pred = pred_groups.get(pert, [])
        idx_true = real_groups.get(pert, [])
        if len(idx_pred) == 0 or len(idx_true) == 0:
            return

        # Extract X arrays and ensure dense
        Xp = to_dense(self.adata_pred[idx_pred].X)
        Xt = to_dense(self.adata_real[idx_true].X)
        Xc_t = to_dense(real_ctrl.X)
        Xc_p = to_dense(pred_ctrl.X)

        # Compute basic metrics
        curr = self._compute_basic_metrics(Xp, Xt, Xc_t, Xc_p, suffix="cell_type")

        # Gene-space metrics (counts)
        if pred_gene_groups is not None and real_gene_groups is not None:
            idx_pred_g = pred_gene_groups.get(pert, [])
            idx_true_g = real_gene_groups.get(pert, [])
            if idx_pred_g and idx_true_g:
                Xp_g = to_dense(self.adata_pred_gene[idx_pred_g].X)
                Xt_g = to_dense(self.adata_real_gene[idx_true_g].X)
                Xc_t_g = to_dense(real_ctrl_gene.X)
                Xc_p_g = to_dense(pred_ctrl_gene.X)
                gene_m = self._compute_basic_metrics(
                    Xp_g, Xt_g, Xc_t_g, Xc_p_g,
                    suffix="cell_type_counts"
                )
                curr.update(gene_m)

        # Append to storage
        self.metrics[celltype]["pert"].append(pert)
        for k, v in curr.items():
            self.metrics[celltype][k].append(v)

    def _compute_basic_metrics(self, pred, true, ctrl_true, ctrl_pred, suffix=""):
        """Compute MSE, Pearson and cosine metrics."""
        m = {}
        m[f"mse_{suffix}"] = compute_mse(pred, true, ctrl_true, ctrl_pred)
        m[f"pearson_delta_{suffix}"] = compute_pearson_delta(pred, true, ctrl_true, ctrl_pred)
        m[f"pearson_delta_sep_ctrls_{suffix}"] = compute_pearson_delta_separate_controls(pred, true, ctrl_true, ctrl_pred)
        m[f"cosine_{suffix}"] = compute_cosine_similarity(pred, true, ctrl_true, ctrl_pred)
        return m

    def _compute_de_metrics(self, celltype):
        """Run DE on full data and compute overlap & related metrics."""
        # Subset by celltype & relevant perts
        real_ct = self.adata_real[self.adata_real.obs[self.celltype_col] == celltype]
        pred_ct = self.adata_pred[self.adata_pred.obs[self.celltype_col] == celltype]
        real_gene_ct = (
            self.adata_real_gene[self.adata_real_gene.obs[self.celltype_col] == celltype]
            if self.adata_real_gene is not None else None
        )
        pred_gene_ct = (
            self.adata_pred_gene[self.adata_pred_gene.obs[self.celltype_col] == celltype]
            if self.adata_pred_gene is not None else None
        )

        # Perform DE
        (DE_true_fc, DE_pred_fc,
         DE_true_pval, DE_pred_pval,
         DE_true_pval_fc, DE_pred_pval_fc,
         DE_true_sig_genes, DE_pred_sig_genes,
         DE_true_df, DE_pred_df) = compute_DE_for_truth_and_pred(
            real_gene_ct or real_ct,
            pred_gene_ct or pred_ct,
            control_pert=self.control,
            pert_col=self.pert_col,
            n_top_genes=2000,
            output_space=self.output_space,
            outdir=self.outdir,
        )

        # unlimited k
        unlimited = compute_gene_overlap_cross_pert(DE_true_pval_fc, DE_pred_pval_fc, control_pert=self.control, k=-1)
        self.metrics[celltype]['DE_pval_fc_N'] = [unlimited.get(p, 0.0) for p in perts]
        self.metrics[celltype]['DE_pval_fc_avg_N'] = np.mean(list(unlimited.values()))

    def _compute_class_score(self, celltype):
        """Compute perturbation ranking score and invert for interpretability."""
        ct_real = self.adata_real[self.adata_real.obs[self.celltype_col] == celltype]
        ct_pred = self.adata_pred[self.adata_pred.obs[self.celltype_col] == celltype]
        score = compute_perturbation_ranking_score(
            ct_pred, ct_real, pert_col=self.pert_col, ctrl_pert=self.control
        )
        self.metrics[celltype]['perturbation_id'] = score
        self.metrics[celltype]['perturbation_score'] = 1 - score

    def _finalize_metrics(self):
        """Convert stored dicts into per-celltype DataFrames."""
        out = {}
        for ct, data in self.metrics.items():
            out[ct] = pd.DataFrame(data).set_index('pert')
        return out
    
def init_worker(global_pred_df, global_true_df):
    global PRED_DF
    global TRUE_DF
    PRED_DF = global_pred_df
    TRUE_DF = global_true_df

def compute_downstream_DE_metrics_parallel(target_gene, p_value_threshold):
    return compute_downstream_DE_metrics(target_gene, PRED_DF, TRUE_DF, p_value_threshold)

def get_downstream_DE_metrics(DE_pred_df, DE_true_df, outdir, celltype,
                              n_workers=10, p_value_threshold=0.05):

    for df in (DE_pred_df, DE_true_df):
        df['abs_fold_change'] = np.abs(df['fold_change'])
        with np.errstate(divide='ignore'):
            df['log_fold_change'] = np.log10(df['fold_change'])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df['abs_log_fold_change'] = np.abs(df['log_fold_change'].fillna(0))
    
    target_genes = DE_true_df['target'].unique()

    with mp.Pool(processes=n_workers, initializer=init_worker, initargs=(DE_pred_df, DE_true_df)) as pool:
        func = partial(compute_downstream_DE_metrics_parallel, p_value_threshold=p_value_threshold)
        results = list(tqdm(pool.imap(func, target_genes), total=len(target_genes)))

    results_df = pd.DataFrame(results)
    outpath = os.path.join(outdir, f"{celltype}_downstream_de_results.csv")
    results_df.to_csv(outpath, index=False)

    return results_df

def get_batched_mean(X, batches):
    if scipy.sparse.issparse(X):
        df = pd.DataFrame(X.todense())
    else:
        df = pd.DataFrame(X)

    df["batch"] = batches
    return df.groupby("batch").mean(numeric_only=True)

