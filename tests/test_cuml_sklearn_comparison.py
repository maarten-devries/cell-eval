#!/usr/bin/env python
"""Test script to compare CUML and sklearn pairwise distances."""

import numpy as np
import sklearn.metrics as skm
from scipy.spatial.distance import pdist

# Try to import CUML
try:
    import cupy as cp
    from cuml.metrics import pairwise_distances as cuml_pairwise_distances
    CUML_AVAILABLE = True
    print("✓ CUML is available")
except ImportError as e:
    CUML_AVAILABLE = False
    print(f"✗ CUML not available: {e}")

def compare_pairwise_distances(n_samples=100, n_features=50, metric='euclidean', seed=42):
    """Compare pairwise distances between sklearn and CUML."""
    
    # Generate random test data
    np.random.seed(seed)
    data = np.random.randn(n_samples, n_features).astype(np.float32)
    
    print(f"\nTesting with data shape: {data.shape}, metric: {metric}")
    print("-" * 50)
    
    # 1. sklearn computation
    sklearn_distances = skm.pairwise_distances(data, metric=metric)
    sklearn_mean = sklearn_distances.mean()
    print(f"sklearn full matrix mean: {sklearn_mean:.8f}")
    
    # Get upper triangle for sklearn (excluding diagonal)
    n = sklearn_distances.shape[0]
    upper_tri_indices = np.triu_indices(n, k=1)
    sklearn_upper_mean = sklearn_distances[upper_tri_indices].mean()
    print(f"sklearn upper triangle mean: {sklearn_upper_mean:.8f}")
    
    # 2. scipy pdist computation (for reference - only does upper triangle)
    if metric == 'euclidean':
        scipy_distances = pdist(data, metric=metric)
        scipy_mean = scipy_distances.mean()
        print(f"scipy pdist mean: {scipy_mean:.8f}")
    
    # 3. CUML computation
    if CUML_AVAILABLE:
        try:
            # Convert to GPU array
            gpu_data = cp.asarray(data)
            
            # CUML pairwise distances
            gpu_distances = cuml_pairwise_distances(gpu_data, metric=metric)
            
            # Full matrix mean
            cuml_full_mean = float(cp.mean(gpu_distances))
            print(f"CUML full matrix mean: {cuml_full_mean:.8f}")
            
            # Upper triangle mean (matching the implementation in _anndata.py)
            if n > 1:
                gpu_upper_tri_indices = cp.triu_indices(n, k=1)
                distances_upper = gpu_distances[gpu_upper_tri_indices]
                cuml_upper_mean = float(cp.mean(distances_upper))
                print(f"CUML upper triangle mean: {cuml_upper_mean:.8f}")
            else:
                cuml_upper_mean = 0.0
            
            # Compare individual values (first 5x5 submatrix)
            print("\nFirst 5x5 distance matrix comparison:")
            print("sklearn:")
            print(sklearn_distances[:5, :5])
            print("\nCUML:")
            print(gpu_distances[:5, :5].get())  # .get() converts back to numpy
            
            # Compute differences
            gpu_distances_cpu = gpu_distances.get()
            max_diff = np.abs(sklearn_distances - gpu_distances_cpu).max()
            mean_diff = np.abs(sklearn_distances - gpu_distances_cpu).mean()
            
            print(f"\nMax absolute difference: {max_diff:.2e}")
            print(f"Mean absolute difference: {mean_diff:.2e}")
            
            # Check if results are close
            tolerance = 1e-5
            all_close = np.allclose(sklearn_distances, gpu_distances_cpu, rtol=tolerance, atol=tolerance)
            upper_means_close = np.abs(sklearn_upper_mean - cuml_upper_mean) < tolerance
            
            print(f"\nAll values close (tolerance={tolerance}): {all_close}")
            print(f"Upper triangle means close: {upper_means_close}")
            
            # Clean up GPU memory
            del gpu_data, gpu_distances
            cp.get_default_memory_pool().free_all_blocks()
            
            return all_close and upper_means_close
            
        except Exception as e:
            print(f"Error during CUML computation: {e}")
            return False
    else:
        print("Skipping CUML comparison (not available)")
        return None

def test_different_metrics():
    """Test various distance metrics."""
    metrics_to_test = ['euclidean', 'manhattan', 'cosine']
    
    print("=" * 60)
    print("Testing different metrics")
    print("=" * 60)
    
    results = {}
    for metric in metrics_to_test:
        try:
            result = compare_pairwise_distances(metric=metric)
            results[metric] = result
        except Exception as e:
            print(f"Error testing {metric}: {e}")
            results[metric] = False
    
    return results

def test_different_sizes():
    """Test with different data sizes."""
    sizes_to_test = [(50, 20), (100, 50), (500, 100), (1000, 200)]
    
    print("\n" + "=" * 60)
    print("Testing different data sizes")
    print("=" * 60)
    
    results = {}
    for n_samples, n_features in sizes_to_test:
        try:
            result = compare_pairwise_distances(n_samples=n_samples, n_features=n_features)
            results[(n_samples, n_features)] = result
        except Exception as e:
            print(f"Error testing size {n_samples}x{n_features}: {e}")
            results[(n_samples, n_features)] = False
    
    return results

if __name__ == "__main__":
    print("CUML vs sklearn pairwise distances comparison")
    print("=" * 60)
    
    # Run tests
    metric_results = test_different_metrics()
    size_results = test_different_sizes()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if CUML_AVAILABLE:
        print("\nMetric test results:")
        for metric, passed in metric_results.items():
            status = "✓ PASSED" if passed else "✗ FAILED" if passed is False else "- SKIPPED"
            print(f"  {metric}: {status}")
        
        print("\nSize test results:")
        for size, passed in size_results.items():
            status = "✓ PASSED" if passed else "✗ FAILED" if passed is False else "- SKIPPED"
            print(f"  {size[0]}x{size[1]}: {status}")
        
        all_passed = all(v for v in list(metric_results.values()) + list(size_results.values()) if v is not None)
        print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    else:
        print("CUML not available - cannot run comparison tests")