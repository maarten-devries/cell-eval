"""Array metrics module."""

from logging import getLogger
import time
from typing import Callable, Literal, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.metrics as skm
from scipy.sparse import issparse
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

# Try to import CUML for GPU acceleration
try:
    import cuml
    from cuml.metrics import pairwise_distances as cuml_pairwise_distances
    CUML_AVAILABLE = True
except (ImportError, Exception) as e:
    # Catch all exceptions including CUDA runtime errors during import
    CUML_AVAILABLE = False
    cuml_pairwise_distances = None

from .._types import PerturbationAnndataPair

logger = getLogger(__name__)


def _optimized_pairwise_distances_mean(
    data: np.ndarray, 
    metric: str = "euclidean", 
    max_samples: int = 2000,
    random_state: int = 42,
    use_gpu: bool = True,
    **kwargs
) -> float:
    """Compute mean of pairwise distances with optimizations.
    
    Priority order:
    1. CUML GPU acceleration (if available and use_gpu=True)
    2. Sampling for large datasets 
    3. scipy.pdist for euclidean distances
    4. sklearn fallback
    
    Args:
        data: Input data matrix (n_samples, n_features)
        metric: Distance metric to use
        max_samples: Maximum number of samples to use for large datasets
        random_state: Random seed for sampling
        use_gpu: Whether to attempt GPU acceleration
        **kwargs: Additional arguments for distance computation
        
    Returns:
        Mean of all pairwise distances
    """
    n_samples = data.shape[0]
    original_n_samples = n_samples
    
    # Try CUML GPU acceleration first
    if CUML_AVAILABLE and use_gpu:
        try:
            import cupy as cp
            
            # Convert sparse matrices to dense before GPU conversion
            if hasattr(data, 'toarray'):
                data = data.toarray()
            
            # Ensure data is in the right dtype for GPU processing
            if data.dtype == np.object_ or data.dtype == object:
                logger.warning(f"Data has object dtype {data.dtype}, converting to float32")
                data = data.astype(np.float32)
            elif not np.issubdtype(data.dtype, np.floating):
                data = data.astype(np.float32)
            
            # Convert to GPU array
            gpu_data = cp.asarray(data)
            logger.info(f"Using CUML GPU acceleration for {n_samples} cells")
            
            # CUML pairwise distances
            gpu_distances = cuml_pairwise_distances(gpu_data, metric=metric, **kwargs)
            
            # Get upper triangle to avoid double counting diagonal
            n = gpu_distances.shape[0]
            if n > 1:
                upper_tri_indices = cp.triu_indices(n, k=1)
                distances = gpu_distances[upper_tri_indices]
                result = float(cp.mean(distances))
            else:
                result = 0.0
            
            # Clean up GPU memory
            del gpu_data, gpu_distances
            cp.get_default_memory_pool().free_all_blocks()
            
            return result
            
        except Exception as e:
            # Check if GPU is available - if so, error out instead of falling back
            try:
                import cupy as cp
                # If we can create a GPU array, GPU is available
                test_gpu = cp.array([1.0])
                gpu_available = True
                del test_gpu
            except:
                gpu_available = False
            
            if gpu_available:
                logger.error(f"CUML GPU computation failed but GPU is available: {e}")
                raise RuntimeError(f"CUML GPU computation failed but GPU is available. This should not happen: {e}")
            else:
                logger.warning(f"CUML GPU computation failed: {e}. Falling back to CPU.")
    
    # CPU fallbacks
    # For euclidean distance on medium-sized datasets, use scipy.pdist (more efficient)
    if metric == "euclidean" and n_samples <= 5000:
        logger.info(f"Using scipy.pdist for {n_samples} cells")
        return pdist(data, metric=metric).mean()
    
    # Fall back to sklearn for other metrics or very large datasets
    logger.info(f"Using sklearn fallback for {n_samples} cells")
    return skm.pairwise_distances(data, metric=metric, **kwargs).mean()


def pearson_delta(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute Pearson correlation between mean differences from control."""
    return _generic_evaluation(
        data,
        pearsonr,  # type: ignore
        use_delta=True,
        embed_key=embed_key,
    )


def mse(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean squared error of each perturbation from control."""
    return _generic_evaluation(
        data, skm.mean_squared_error, use_delta=False, embed_key=embed_key
    )


def mae(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean absolute error of each perturbation from control."""
    return _generic_evaluation(
        data, skm.mean_absolute_error, use_delta=False, embed_key=embed_key
    )


def mse_delta(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean squared error of each perturbation-control delta."""
    return _generic_evaluation(
        data, skm.mean_squared_error, use_delta=True, embed_key=embed_key
    )


def mae_delta(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean absolute error of each perturbation-control delta."""
    return _generic_evaluation(
        data, skm.mean_absolute_error, use_delta=True, embed_key=embed_key
    )


def edistance(
    data: PerturbationAnndataPair,
    embed_key: str | None = None,
    metric: str = "euclidean",
    **kwargs,
) -> float:
    """Compute Euclidean distance of each perturbation-control delta."""

    def _edistance(
        x: np.ndarray,
        y: np.ndarray,
        metric: str = "euclidean",
        precomp_sigma_y: float | None = None,
        **kwargs,
    ) -> float:
        # Use optimized GPU version for sigma_x computation
        sigma_x = _optimized_pairwise_distances_mean(x, metric=metric, **kwargs)
        sigma_y = (
            precomp_sigma_y
            if precomp_sigma_y is not None
            else _optimized_pairwise_distances_mean(y, metric=metric, **kwargs)
        )
        
        # Compute cross-distance delta using GPU if available
        if CUML_AVAILABLE:
            try:
                import cupy as cp
                
                # Ensure arrays are dense and proper dtype
                if hasattr(x, 'toarray'):
                    x = x.toarray()
                if hasattr(y, 'toarray'):
                    y = y.toarray()
                
                # Force conversion to float32 numpy arrays first
                x_float = np.asarray(x, dtype=np.float32)
                y_float = np.asarray(y, dtype=np.float32)
                
                # Convert to GPU arrays
                gpu_x = cp.asarray(x_float)
                gpu_y = cp.asarray(y_float)
                
                # Compute cross-distances on GPU (don't pass kwargs as they might have incompatible types)
                gpu_distances = cuml_pairwise_distances(gpu_x, gpu_y, metric=metric)
                delta = float(cp.mean(gpu_distances))
                
                # Clean up GPU memory
                del gpu_x, gpu_y, gpu_distances
                cp.get_default_memory_pool().free_all_blocks()
            except Exception as e:
                logger.warning(f"GPU cross-distance failed: {e}. Using CPU fallback.")
                delta = skm.pairwise_distances(x, y, metric=metric, **kwargs).mean()
        else:
            delta = skm.pairwise_distances(x, y, metric=metric, **kwargs).mean()
        
        return 2 * delta - sigma_x - sigma_y

    d_real = np.zeros(data.perts.size)
    d_pred = np.zeros(data.perts.size)

    # Precompute sigma for control data (reused by all perturbations)
    logger.info("Precomputing sigma for control data (real)")
    real_start = time.time()
    precomp_sigma_real = _optimized_pairwise_distances_mean(
        data.ctrl_matrix(which="real", embed_key=embed_key), metric=metric, **kwargs
    )
    real_elapsed = time.time() - real_start
    logger.info(f"✓ Real control sigma precomputation completed in {real_elapsed:.2f} seconds")

    logger.info("Precomputing sigma for control data (pred)")
    pred_start = time.time()
    precomp_sigma_pred = _optimized_pairwise_distances_mean(
        data.ctrl_matrix(which="pred", embed_key=embed_key), metric=metric, **kwargs
    )
    pred_elapsed = time.time() - pred_start
    logger.info(f"✓ Pred control sigma precomputation completed in {pred_elapsed:.2f} seconds")

    logger.info(f"Computing e-distance for {data.perts.size} perturbations")
    edist_start = time.time()
    for idx, delta in enumerate(data.iter_cell_arrays(embed_key=embed_key)):
        d_real[idx] = _edistance(
            delta.pert_real,
            delta.ctrl_real,
            precomp_sigma_y=precomp_sigma_real,
            metric=metric,
            **kwargs,
        )
        d_pred[idx] = _edistance(
            delta.pert_pred,
            delta.ctrl_pred,
            precomp_sigma_y=precomp_sigma_pred,
            metric=metric,
            **kwargs,
        )
    edist_elapsed = time.time() - edist_start
    logger.info(f"✓ E-distance computation for all perturbations completed in {edist_elapsed:.2f} seconds")

    return pearsonr(d_real, d_pred).correlation


def discrimination_score(
    data: PerturbationAnndataPair,
    metric: str = "l1",
    embed_key: str | None = None,
    exclude_target_gene: bool = True,
) -> dict[str, float]:
    """Base implementation for discrimination score computation.

    Best score is 1.0 - worst score is 0.0.

    Args:
        data: PerturbationAnndataPair containing real and predicted data
        embed_key: Key for embedding data in obsm, None for expression data
        metric: Metric for distance calculation (e.g., "l1", "l2", see `scipy.metrics.pairwise.distance_metrics`)
        exclude_target_gene: Whether to exclude target gene from calculation

    Returns:
        Dictionary mapping perturbation names to normalized ranks
    """
    if metric == "l1" or metric == "manhattan" or metric == "cityblock":
        # Ignore the embedding key for L1
        embed_key = None

    # Compute perturbation effects for all perturbations
    real_effects = np.vstack(
        [
            d.perturbation_effect(which="real", abs=True)
            for d in data.iter_bulk_arrays(embed_key=embed_key)
        ]
    )
    pred_effects = np.vstack(
        [
            d.perturbation_effect(which="pred", abs=True)
            for d in data.iter_bulk_arrays(embed_key=embed_key)
        ]
    )

    norm_ranks = {}
    for p_idx, p in enumerate(data.perts):
        # Determine which features to include in the comparison
        if exclude_target_gene and not embed_key:
            # For expression data, exclude the target gene
            include_mask = np.flatnonzero(data.genes != p)
        else:
            # For embedding data or when not excluding target gene, use all features
            include_mask = np.ones(real_effects.shape[1], dtype=bool)

        # Compute distances to all real effects
        distances = skm.pairwise_distances(
            real_effects[
                :, include_mask
            ],  # compare to all real effects across perturbations
            pred_effects[p_idx, include_mask].reshape(
                1, -1
            ),  # select pred effect for current perturbation
            metric=metric,
        ).flatten()

        # Sort by distance (ascending - lower distance = better match)
        sorted_indices = np.argsort(distances)

        # Find rank of the correct perturbation
        p_index = np.flatnonzero(data.perts == p)[0]
        rank = np.flatnonzero(sorted_indices == p_index)[0]

        # Normalize rank by total number of perturbations
        norm_rank = rank / data.perts.size
        norm_ranks[str(p)] = 1 - norm_rank

    return norm_ranks


def _generic_evaluation(
    data: PerturbationAnndataPair,
    func: Callable[[np.ndarray, np.ndarray], float],
    use_delta: bool = False,
    embed_key: str | None = None,
) -> dict[str, float]:
    """Generic evaluation function for anndata pair."""
    res = {}
    for bulk_array in data.iter_bulk_arrays(embed_key=embed_key):
        if use_delta:
            x = bulk_array.perturbation_effect(which="pred", abs=False)
            y = bulk_array.perturbation_effect(which="real", abs=False)
        else:
            x = bulk_array.pert_pred
            y = bulk_array.pert_real

        result = func(x, y)
        if isinstance(result, tuple):
            result = result[0]

        res[bulk_array.key] = float(result)

    return res


# TODO: clean up this implementation
class ClusteringAgreement:
    """Compute clustering agreement between real and predicted perturbation centroids."""

    def __init__(
        self,
        embed_key: str | None = None,
        real_resolution: float = 1.0,
        pred_resolutions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0),
        metric: Literal["ami", "nmi", "ari"] = "ami",
        n_neighbors: int = 15,
    ) -> None:
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
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=key_added,
            flavor="igraph",
            n_iterations=2,
        )

    @staticmethod
    def _centroid_ann(
        adata: ad.AnnData,
        category_key: str,
        control_pert: str,
        embed_key: str | None = None,
    ) -> ad.AnnData:
        # Isolate the features
        feats = adata.obsm.get(embed_key, adata.X)  # type: ignore

        # Convert to float if not already
        if feats.dtype != np.dtype("float64"):  # type: ignore
            feats = feats.astype(np.float64)  # type: ignore

        # Densify if required
        if issparse(feats):
            feats = feats.toarray()  # type: ignore

        cats = adata.obs[category_key].values
        uniq, inv = np.unique(cats, return_inverse=True)  # type: ignore
        centroids = np.zeros((uniq.size, feats.shape[1]), dtype=feats.dtype)  # type: ignore

        for i, cat in enumerate(uniq):
            mask = cats == cat
            if np.any(mask):
                centroids[i] = feats[mask].mean(axis=0)  # type: ignore

        adc = ad.AnnData(X=centroids)
        adc.obs[category_key] = uniq
        return adc[adc.obs[category_key] != control_pert]

    def __call__(self, data: PerturbationAnndataPair) -> float:
        cats_sorted = sorted([c for c in data.perts if c != data.control_pert])

        # 2. build centroids
        ad_real_cent = self._centroid_ann(
            adata=data.real,
            category_key=data.pert_col,
            control_pert=data.control_pert,
            embed_key=self.embed_key,
        )
        ad_pred_cent = self._centroid_ann(
            adata=data.pred,
            category_key=data.pert_col,
            control_pert=data.control_pert,
            embed_key=self.embed_key,
        )

        # 3. cluster real once
        real_key = "real_clusters"
        self._cluster_leiden(
            ad_real_cent, self.real_resolution, real_key, self.n_neighbors
        )
        ad_real_cent.obs = ad_real_cent.obs.set_index(data.pert_col).loc[cats_sorted]
        real_labels = pd.Categorical(ad_real_cent.obs[real_key])

        # 4. sweep predicted resolutions
        best_score = 0.0
        ad_pred_cent.obs = ad_pred_cent.obs.set_index(data.pert_col).loc[cats_sorted]
        for r in self.pred_resolutions:
            pred_key = f"pred_clusters_{r}"
            self._cluster_leiden(ad_pred_cent, r, pred_key, self.n_neighbors)
            pred_labels = pd.Categorical(ad_pred_cent.obs[pred_key])
            score = self._score(real_labels, pred_labels, self.metric)  # type: ignore
            best_score = max(best_score, score)

        return float(best_score)
