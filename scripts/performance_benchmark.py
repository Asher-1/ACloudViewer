#!/usr/bin/env python3
"""
Performance Benchmarking Script for CloudViewer
This script measures various performance metrics before and after optimizations.
"""

import os
import sys
import time
import psutil
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Any

try:
    import cloudViewer as cv3d
    import numpy as np
except ImportError:
    print("CloudViewer not found. Please build and install first.")
    sys.exit(1)


class PerformanceBenchmark:
    """Performance benchmarking suite for CloudViewer operations."""
    
    def __init__(self):
        self.results = {}
        self.memory_usage = []
        self.cpu_usage = []
        
    def measure_time(self, func, *args, **kwargs) -> Tuple[float, Any]:
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return end_time - start_time, result
    
    def measure_memory(self, func, *args, **kwargs) -> Tuple[float, Any]:
        """Measure memory usage of a function."""
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        return mem_after - mem_before, result
    
    def benchmark_point_cloud_operations(self, num_points: int = 100000) -> Dict[str, float]:
        """Benchmark basic point cloud operations."""
        print(f"Benchmarking point cloud operations with {num_points} points...")
        
        # Generate random point cloud
        points = np.random.rand(num_points, 3).astype(np.float32)
        pcd = cv3d.geometry.PointCloud()
        pcd.points = cv3d.utility.Vector3dVector(points)
        
        benchmarks = {}
        
        # KD-Tree construction
        time_taken, _ = self.measure_time(pcd.build_kdtree)
        benchmarks['kdtree_construction'] = time_taken
        
        # Normal estimation
        time_taken, _ = self.measure_time(
            pcd.estimate_normals,
            search_param=cv3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        benchmarks['normal_estimation'] = time_taken
        
        # Voxel downsampling
        time_taken, downsampled = self.measure_time(pcd.voxel_down_sample, 0.05)
        benchmarks['voxel_downsampling'] = time_taken
        
        # Statistical outlier removal
        time_taken, _ = self.measure_time(
            pcd.remove_statistical_outlier, nb_neighbors=20, std_ratio=2.0
        )
        benchmarks['outlier_removal'] = time_taken
        
        return benchmarks
    
    def benchmark_mesh_operations(self) -> Dict[str, float]:
        """Benchmark mesh processing operations."""
        print("Benchmarking mesh operations...")
        
        # Create test mesh
        mesh = cv3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
        
        benchmarks = {}
        
        # Mesh smoothing
        time_taken, _ = self.measure_time(
            mesh.filter_smooth_laplacian, number_of_iterations=10
        )
        benchmarks['mesh_smoothing'] = time_taken
        
        # Mesh simplification
        time_taken, _ = self.measure_time(
            mesh.simplify_quadric_decimation, target_number_of_triangles=1000
        )
        benchmarks['mesh_simplification'] = time_taken
        
        # Normal computation
        time_taken, _ = self.measure_time(mesh.compute_vertex_normals)
        benchmarks['mesh_normals'] = time_taken
        
        return benchmarks
    
    def benchmark_io_operations(self, file_size: str = "medium") -> Dict[str, float]:
        """Benchmark I/O operations."""
        print(f"Benchmarking I/O operations ({file_size} files)...")
        
        benchmarks = {}
        
        # Generate test data based on size
        if file_size == "small":
            num_points = 10000
        elif file_size == "medium":
            num_points = 100000
        else:  # large
            num_points = 1000000
        
        points = np.random.rand(num_points, 3).astype(np.float32)
        pcd = cv3d.geometry.PointCloud()
        pcd.points = cv3d.utility.Vector3dVector(points)
        
        # PLY write
        temp_file = f"/tmp/test_{file_size}.ply"
        time_taken, _ = self.measure_time(cv3d.io.write_point_cloud, temp_file, pcd)
        benchmarks['ply_write'] = time_taken
        
        # PLY read
        time_taken, _ = self.measure_time(cv3d.io.read_point_cloud, temp_file)
        benchmarks['ply_read'] = time_taken
        
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return benchmarks
    
    def benchmark_registration(self) -> Dict[str, float]:
        """Benchmark point cloud registration operations."""
        print("Benchmarking registration operations...")
        
        # Create source and target point clouds
        source = cv3d.geometry.PointCloud()
        target = cv3d.geometry.PointCloud()
        
        source_points = np.random.rand(10000, 3).astype(np.float32)
        target_points = source_points + np.random.normal(0, 0.01, source_points.shape)
        
        source.points = cv3d.utility.Vector3dVector(source_points)
        target.points = cv3d.utility.Vector3dVector(target_points)
        
        # Estimate normals
        source.estimate_normals()
        target.estimate_normals()
        
        benchmarks = {}
        
        # FPFH feature computation
        time_taken, source_fpfh = self.measure_time(
            cv3d.pipelines.registration.compute_fpfh_feature,
            source,
            cv3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100)
        )
        benchmarks['fpfh_computation'] = time_taken
        
        target_fpfh = cv3d.pipelines.registration.compute_fpfh_feature(
            target,
            cv3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100)
        )
        
        # RANSAC registration
        time_taken, _ = self.measure_time(
            cv3d.pipelines.registration.registration_ransac_based_on_feature_matching,
            source, target, source_fpfh, target_fpfh, True, 0.075,
            cv3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [], cv3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        benchmarks['ransac_registration'] = time_taken
        
        return benchmarks
    
    def run_comprehensive_benchmark(self, iterations: int = 3) -> Dict[str, Dict[str, float]]:
        """Run comprehensive performance benchmark."""
        print(f"Running comprehensive benchmark ({iterations} iterations)...")
        
        all_results = {}
        
        # Run multiple iterations and average results
        for i in range(iterations):
            print(f"\nIteration {i + 1}/{iterations}")
            
            iteration_results = {}
            
            # Point cloud operations
            iteration_results['point_cloud'] = self.benchmark_point_cloud_operations()
            
            # Mesh operations
            iteration_results['mesh'] = self.benchmark_mesh_operations()
            
            # I/O operations
            iteration_results['io'] = self.benchmark_io_operations()
            
            # Registration
            iteration_results['registration'] = self.benchmark_registration()
            
            # Store results
            for category, benchmarks in iteration_results.items():
                if category not in all_results:
                    all_results[category] = {name: [] for name in benchmarks.keys()}
                
                for name, time_taken in benchmarks.items():
                    all_results[category][name].append(time_taken)
        
        # Calculate averages and standard deviations
        final_results = {}
        for category, benchmarks in all_results.items():
            final_results[category] = {}
            for name, times in benchmarks.items():
                avg_time = statistics.mean(times)
                std_time = statistics.stdev(times) if len(times) > 1 else 0
                final_results[category][name] = {
                    'avg': avg_time,
                    'std': std_time,
                    'min': min(times),
                    'max': max(times)
                }
        
        return final_results
    
    def print_results(self, results: Dict[str, Dict[str, Dict[str, float]]]):
        """Print benchmark results in a formatted way."""
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*80)
        
        for category, benchmarks in results.items():
            print(f"\n{category.upper()} OPERATIONS:")
            print("-" * 40)
            
            for name, stats in benchmarks.items():
                print(f"{name:<25}: {stats['avg']:.4f}s ± {stats['std']:.4f}s "
                      f"(min: {stats['min']:.4f}s, max: {stats['max']:.4f}s)")
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save benchmark results to a file."""
        import json
        
        # Add system information
        system_info = {
            'platform': sys.platform,
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'python_version': sys.version,
            'cloudviewer_version': getattr(cv3d, '__version__', 'unknown'),
            'numpy_version': np.__version__,
        }
        
        output = {
            'system_info': system_info,
            'benchmark_results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description='CloudViewer Performance Benchmark')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of iterations to run each benchmark')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output file for results')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick benchmark with fewer operations')
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark()
    
    if args.quick:
        print("Running quick benchmark...")
        results = benchmark.benchmark_point_cloud_operations(num_points=10000)
        benchmark.print_results({'point_cloud': {k: {'avg': v, 'std': 0, 'min': v, 'max': v} 
                                               for k, v in results.items()}})
    else:
        results = benchmark.run_comprehensive_benchmark(args.iterations)
        benchmark.print_results(results)
        benchmark.save_results(results, args.output)
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()