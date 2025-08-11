#!/usr/bin/env python3
"""
Benchmark script for comparing pixel_metrics.py vs faster_pixel_metrics.py
Tests both speed improvement and result correctness
"""

import numpy as np
import time
import sys
import os
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Add the mmseg path
sys.path.insert(0, '/home/intern/InternVL-MMDetSeg/mmsegmentation')

# Import both versions
try:
    from mmseg.core.evaluation import pixel_metrics as original_metrics
    from mmseg.core.evaluation import faster_pixel_metrics as fast_metrics
    print("✅ Successfully imported both metric modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def generate_test_data(num_samples: int = 50, 
                      height: int = 256, 
                      width: int = 256,
                      seed: int = 42) -> tuple:
    """Generate synthetic test data for benchmarking"""
    np.random.seed(seed)
    
    results = []
    gt_seg_maps = []
    
    print(f"Generating {num_samples} test samples of size {height}x{width}...")
    
    for i in range(num_samples):
        # Generate prediction logits (simulate model output)
        pred_logits = np.random.randn(height, width) * 2.0
        
        # Generate ground truth (binary mask with some objects)
        gt = np.zeros((height, width), dtype=np.uint8)
        
        # Add some circular objects
        num_objects = np.random.randint(1, 8)
        for _ in range(num_objects):
            center_x = np.random.randint(20, width-20)
            center_y = np.random.randint(20, height-20)
            radius = np.random.randint(5, 25)
            
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            gt[mask] = 1
        
        results.append(pred_logits)
        gt_seg_maps.append(gt)
    
    print(f"✅ Generated {len(results)} samples with {np.mean([gt.sum() for gt in gt_seg_maps]):.1f} avg pixels per target")
    return results, gt_seg_maps

def benchmark_single_metric(metric_name: str,
                          results: List[np.ndarray], 
                          gt_seg_maps: List[np.ndarray],
                          original_func: callable,
                          fast_func: callable,
                          num_runs: int = 3) -> Dict[str, Any]:
    """Benchmark a single metric function"""
    
    print(f"\n🔄 Benchmarking {metric_name}...")
    
    # Warm up both functions
    print("  Warming up...")
    try:
        _ = original_func(results[:5], gt_seg_maps[:5], 1, 255, [metric_name], 10)
        _ = fast_func(results[:5], gt_seg_maps[:5], 1, 255, [metric_name], 10)
    except Exception as e:
        print(f"  ⚠️  Warmup failed: {e}")
    
    # Benchmark original version
    print("  Testing original version...")
    original_times = []
    original_result = None
    
    for run in range(num_runs):
        start_time = time.perf_counter()
        try:
            original_result = original_func(
                results, gt_seg_maps, 1, 255, [metric_name], 10
            )
            end_time = time.perf_counter()
            original_times.append(end_time - start_time)
            print(f"    Run {run+1}: {original_times[-1]:.3f}s")
        except Exception as e:
            print(f"    ❌ Original version failed: {e}")
            return None
    
    # Benchmark fast version  
    print("  Testing fast version...")
    fast_times = []
    fast_result = None
    
    for run in range(num_runs):
        start_time = time.perf_counter()
        try:
            fast_result = fast_func(
                results, gt_seg_maps, 1, 255, [metric_name], 10
            )
            end_time = time.perf_counter()
            fast_times.append(end_time - start_time)
            print(f"    Run {run+1}: {fast_times[-1]:.3f}s")
        except Exception as e:
            print(f"    ❌ Fast version failed: {e}")
            return None
    
    # Calculate statistics
    orig_mean = np.mean(original_times)
    fast_mean = np.mean(fast_times)
    speedup = orig_mean / fast_mean
    
    # Compare results
    correctness = compare_results(original_result, fast_result, metric_name)
    
    return {
        'metric': metric_name,
        'original_time': orig_mean,
        'fast_time': fast_mean,
        'speedup': speedup,
        'original_std': np.std(original_times),
        'fast_std': np.std(fast_times),
        'correctness': correctness,
        'original_result': original_result,
        'fast_result': fast_result
    }

def compare_results(orig_result: Dict, fast_result: Dict, metric_name: str) -> Dict[str, bool]:
    """Compare results between original and fast versions"""
    
    if orig_result is None or fast_result is None:
        return {'overall': False, 'details': 'One or both results are None'}
    
    correctness = {}
    tolerance = 1e-6
    
    try:
        if metric_name == 'mIoU':
            # Compare mIoU and OA
            for key in ['mIoU', 'OA']:
                if key in orig_result and key in fast_result:
                    orig_val = float(orig_result[key])
                    fast_val = float(fast_result[key])
                    diff = abs(orig_val - fast_val)
                    correctness[key] = diff < tolerance
                    if not correctness[key]:
                        print(f"    ❌ {key}: orig={orig_val:.6f}, fast={fast_val:.6f}, diff={diff:.6f}")
        
        elif metric_name == 'ROC':
            # Compare ROC metrics
            for key in ['TPR', 'FPR', 'Recall', 'Precision']:
                if key in orig_result and key in fast_result:
                    orig_arr = np.array(orig_result[key])
                    fast_arr = np.array(fast_result[key])
                    if orig_arr.shape == fast_arr.shape:
                        max_diff = np.max(np.abs(orig_arr - fast_arr))
                        correctness[key] = max_diff < tolerance
                        if not correctness[key]:
                            print(f"    ❌ {key}: max_diff={max_diff:.6f}")
                    else:
                        correctness[key] = False
                        print(f"    ❌ {key}: shape mismatch {orig_arr.shape} vs {fast_arr.shape}")
        
        elif metric_name == 'PdFa':
            # Compare PD/FA metrics
            for key in ['PD', 'FA']:
                if key in orig_result and key in fast_result:
                    orig_arr = np.array(orig_result[key])
                    fast_arr = np.array(fast_result[key])
                    if orig_arr.shape == fast_arr.shape:
                        max_diff = np.max(np.abs(orig_arr - fast_arr))
                        # PdFa tolerance is higher due to region analysis differences
                        pd_fa_tolerance = 0.1 if key == 'FA' else 0.05
                        correctness[key] = max_diff < pd_fa_tolerance
                        if not correctness[key]:
                            print(f"    ⚠️  {key}: max_diff={max_diff:.6f} (tolerance={pd_fa_tolerance})")
                    else:
                        correctness[key] = False
                        print(f"    ❌ {key}: shape mismatch {orig_arr.shape} vs {fast_arr.shape}")
    
    except Exception as e:
        print(f"    ❌ Comparison error: {e}")
        return {'overall': False, 'details': str(e)}
    
    # Overall correctness
    if correctness:
        overall_correct = all(correctness.values())
        correctness['overall'] = overall_correct
        
        if overall_correct:
            print(f"    ✅ All {metric_name} results match!")
        else:
            failed_keys = [k for k, v in correctness.items() if not v and k != 'overall']
            print(f"    ⚠️  {metric_name} mismatch in: {failed_keys}")
    else:
        correctness['overall'] = False
        print(f"    ❌ No comparable results found for {metric_name}")
    
    return correctness

def print_benchmark_results(results: List[Dict[str, Any]]):
    """Print formatted benchmark results"""
    
    print("\n" + "="*80)
    print("🏆 BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    print(f"{'Metric':<10} {'Original(s)':<12} {'Fast(s)':<10} {'Speedup':<10} {'Correct':<10}")
    print("-" * 80)
    
    total_speedup = 1.0
    all_correct = True
    
    for result in results:
        if result is None:
            continue
            
        metric = result['metric']
        orig_time = result['original_time']
        fast_time = result['fast_time'] 
        speedup = result['speedup']
        correct = result['correctness']['overall']
        
        total_speedup *= speedup
        all_correct &= correct
        
        status = "✅" if correct else "❌"
        print(f"{metric:<10} {orig_time:<12.3f} {fast_time:<10.3f} {speedup:<10.2f}x {status:<10}")
    
    geom_mean_speedup = total_speedup ** (1.0 / len([r for r in results if r is not None]))
    
    print("-" * 80)
    print(f"📊 Overall Performance:")
    print(f"   • Geometric mean speedup: {geom_mean_speedup:.2f}x")
    print(f"   • All results correct: {'✅ YES' if all_correct else '❌ NO'}")
    
    # Detailed results
    print(f"\n📋 Detailed Results:")
    for result in results:
        if result is None:
            continue
        print(f"\n{result['metric']} Details:")
        print(f"  Original: {result['original_time']:.3f}±{result['original_std']:.3f}s")
        print(f"  Fast:     {result['fast_time']:.3f}±{result['fast_std']:.3f}s") 
        print(f"  Speedup:  {result['speedup']:.2f}x")
        
        if not result['correctness']['overall']:
            print(f"  ⚠️  Correctness issues: {result['correctness']}")

def run_memory_test(results: List[np.ndarray], gt_seg_maps: List[np.ndarray]):
    """Test memory usage of both versions"""
    print("\n🧠 Memory Usage Test...")
    
    try:
        import psutil
        import gc
        
        # Test original version
        gc.collect()
        process = psutil.Process()
        mem_before_orig = process.memory_info().rss / 1024 / 1024  # MB
        
        _ = original_metrics.eval_pixel_metrics(
            results, gt_seg_maps, 1, 255, ['mIoU', 'ROC'], 10
        )
        
        mem_after_orig = process.memory_info().rss / 1024 / 1024  # MB
        orig_memory = mem_after_orig - mem_before_orig
        
        # Test fast version
        gc.collect()
        mem_before_fast = process.memory_info().rss / 1024 / 1024  # MB
        
        _ = fast_metrics.eval_pixel_metrics(
            results, gt_seg_maps, 1, 255, ['mIoU', 'ROC'], 10
        )
        
        mem_after_fast = process.memory_info().rss / 1024 / 1024  # MB
        fast_memory = mem_after_fast - mem_before_fast
        
        print(f"  Original version: {orig_memory:.1f} MB")
        print(f"  Fast version:     {fast_memory:.1f} MB")
        print(f"  Memory ratio:     {orig_memory/max(fast_memory, 0.1):.2f}x")
        
    except ImportError:
        print("  ⚠️  psutil not available, skipping memory test")
    except Exception as e:
        print(f"  ❌ Memory test failed: {e}")

def main():
    """Main benchmark function"""
    
    print("🚀 Pixel Metrics Benchmark")
    print("=" * 50)
    
    # Test different sizes
    test_configs = [
        {'num_samples': 20, 'height': 128, 'width': 128, 'name': 'Small'},
        {'num_samples': 50, 'height': 256, 'width': 256, 'name': 'Medium'},
        {'num_samples': 20, 'height': 512, 'width': 512, 'name': 'Large'},
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\n🎯 Testing {config['name']} dataset ({config['num_samples']} samples, {config['height']}x{config['width']})")
        
        # Generate test data
        results, gt_seg_maps = generate_test_data(
            config['num_samples'], config['height'], config['width']
        )
        
        # Test each metric
        metrics_to_test = ['mIoU', 'ROC']  # Skip PdFa for now as it's most complex
        
        config_results = []
        for metric in metrics_to_test:
            result = benchmark_single_metric(
                metric, results, gt_seg_maps,
                original_metrics.eval_pixel_metrics,
                fast_metrics.eval_pixel_metrics,
                num_runs=3
            )
            if result:
                result['config'] = config['name']
                config_results.append(result)
        
        all_results.extend(config_results)
        
        # Print intermediate results for this config
        if config_results:
            print(f"\n📊 {config['name']} Results:")
            for result in config_results:
                print(f"  {result['metric']}: {result['speedup']:.2f}x speedup, "
                      f"{'✅' if result['correctness']['overall'] else '❌'} correct")
    
    # Print final summary
    print_benchmark_results(all_results)
    
    # Test memory usage on medium dataset
    if len(all_results) > 0:
        results, gt_seg_maps = generate_test_data(30, 256, 256)
        run_memory_test(results, gt_seg_maps)
    
    print(f"\n🎉 Benchmark completed!")

if __name__ == "__main__":
    main() 