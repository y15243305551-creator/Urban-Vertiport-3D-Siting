import numpy as np
import networkx as nx
import heapq
import matplotlib.pyplot as plt
from tqdm import tqdm
from bitarray import bitarray
from collections import defaultdict
import time
import multiprocessing as mp
import pickle
import os
import shutil
import math
from scipy.spatial import KDTree

from .cache_manager import CacheManager
from .visualization import RealtimeSelectionVisualizer
from .utils import calculate_coverage, improved_normalization
from .reachability_calculation import ReachabilityCalculator
from .utils import calculate_theoretical_max_coverage

class VertiportOptimizer:
    """低空起降点优化算法类"""
    
    def __init__(self, target_coverage=0.95, cover_weight=1.5, connectivity_weight=0.5):
        self.target_coverage = target_coverage
        self.cover_weight = cover_weight
        self.connectivity_weight = connectivity_weight
        self.cost_multiplier = 5000
        self.fixed_cost = 150000
        
        # 初始化缓存管理器
        self.cache_manager = CacheManager(cache_dir="cache")
        
        # 可达性计算器将在运行时设置
        self.reachability_calc = None
    
    def set_reachability_calculator(self, reachability_calc):
        """设置可达性计算器"""
        self.reachability_calc = reachability_calc
    
    def integrated_efficiency_selection_with_aggregated_data(self, aggregated_data, grid_original_info,
                                                           cover_radii, relay_radii, city_shapefile,
                                                           batch_size=5000, enable_realtime_visualization=True):
        """
        基于综合效率的迭代选点算法 - 明确使用集计数据
        """
        print("=== 基于综合效率迭代选点（使用集计数据）===")
        
        # 从集计数据中提取变量
        station_coords = aggregated_data['station_coords']
        demand_coords = aggregated_data['demand_coords']
        building_coords = aggregated_data['building_coords']
        building_heights = aggregated_data['building_heights']
        building_elevations = aggregated_data['building_elevations']
        building_areas = aggregated_data['building_areas']
        station_heights = aggregated_data['station_heights']
        station_elevations = aggregated_data['station_elevations']
        demand_elevations = aggregated_data['demand_elevations']
        
        # 初始化实时可视化
        visualizer = None
        if enable_realtime_visualization:
            visualizer = RealtimeSelectionVisualizer(station_coords, demand_coords, cover_radii, city_shapefile)
            print("实时动态可视化已启用（使用集计数据近似显示）")
        
        # 性能监控初始化
        start_time = time.time()
        iteration_data = {
            'iterations': [], 'coverage_rates': [], 'selected_count': [], 'time_stamps': [],
            'coverage_weights': [], 'connectivity_weights': [], 'coverage_efficiencies': [],
            'connectivity_efficiencies': [], 'normalized_coverage_eff': [], 'normalized_connectivity_eff': [],
            'selected_stations_history': [], 'uncovered_counts': []
        }
        
        # 初始化数据
        station_coords = np.asarray(station_coords)
        demand_coords = np.asarray(demand_coords)
        cover_radii = np.asarray(cover_radii)
        relay_radii = np.asarray(relay_radii)
        total_demand_points = len(demand_coords)
        total_candidates = len(station_coords)
        
        selected_stations = []
        
        # 使用位数组跟踪未覆盖的需求点
        uncovered = bitarray(len(demand_coords))
        uncovered.setall(True)
        
        # 定义reverse_index_data和connectivity_matrix变量
        reverse_index_data = None
        connectivity_matrix = None
        
        try:
            # 计算数据哈希 - 使用集计数据
            data_hash = self.cache_manager.get_data_hash(
                station_coords, demand_coords, building_coords,
                building_heights, building_elevations, building_areas
            )
            
            # 步骤1: 构建反向索引 - 使用集计数据
            reverse_index_data = self.build_reverse_index(
                data_hash, station_coords, demand_coords,
                building_coords, building_heights, building_elevations, building_areas,
                station_heights, station_elevations, demand_elevations,
                batch_size
            )
            
            # 检查反向索引数据
            if not reverse_index_data:
                print("❌ 错误：反向索引数据为空，无法继续")
                return [], iteration_data, None, None
            
            # 步骤2: 计算连通性矩阵 - 使用集计数据
            connectivity_matrix = self.compute_connectivity_matrix(
                data_hash=data_hash,
                station_coords=station_coords,
                station_heights=station_heights,
                station_elevations=station_elevations,
                building_coords=building_coords,
                building_heights=building_heights,
                building_elevations=building_elevations,
                building_areas=building_areas,
                relay_radii=relay_radii,
                demand_coords=demand_coords,
                demand_elevations=demand_elevations,
                batch_size=2000,
            )
            
            # 步骤3: 迭代选点 - 使用集计数据
            print("开始基于综合效率迭代选点（使用集计数据）")
            iteration = 0
            stagnation_count = 0
            previous_coverage = 0
            max_stagnation_iterations = 100
            
            while True:
                iteration += 1
                current_coverage = 1 - uncovered.count() / len(uncovered)
                uncovered_count = uncovered.count()
                
                # 记录迭代数据
                iteration_data['iterations'].append(iteration)
                iteration_data['coverage_rates'].append(current_coverage)
                iteration_data['selected_count'].append(len(selected_stations))
                iteration_data['time_stamps'].append(time.time() - start_time)
                iteration_data['selected_stations_history'].append(selected_stations.copy())
                iteration_data['uncovered_counts'].append(uncovered_count)
                
                # 更新实时可视化
                if visualizer:
                    visualizer.update_plot(iteration, selected_stations, current_coverage)
                
                # 计算动态权重
                progress = min(current_coverage / self.target_coverage, 1.0)
                coverage_weight = 1.5 - 0.5 * progress
                connectivity_weight = 0.5 + 0.5 * progress
                coverage_weight = max(coverage_weight, 0.1)
                connectivity_weight = max(connectivity_weight, 0.1)
                
                iteration_data['coverage_weights'].append(coverage_weight)
                iteration_data['connectivity_weights'].append(connectivity_weight)
                
                # 停滞检测
                if abs(current_coverage - previous_coverage) < 0.0001:
                    stagnation_count += 1
                    print(f"覆盖率停滞 ({current_coverage:.4%})，停滞计数: {stagnation_count}/{max_stagnation_iterations}")
                else:
                    stagnation_count = 0
                
                print(f"\n迭代 {iteration}, 当前覆盖率: {current_coverage:.2%}, 选中站点: {len(selected_stations)}")
                print(f"未覆盖需求点: {uncovered_count}, 进度: {progress:.1%}")
                print(f"权重 - 覆盖: {coverage_weight:.2f}, 连通: {connectivity_weight:.2f}")
                
                previous_coverage = current_coverage
                
                # 终止条件检查
                if current_coverage >= self.target_coverage:
                    print(f"达到目标覆盖率 {current_coverage:.2%}，终止选点")
                    break
                if stagnation_count >= max_stagnation_iterations:
                    print(f"连续 {max_stagnation_iterations} 次迭代覆盖率无变化，提前终止")
                    break
                if uncovered_count == 0:
                    print("所有需求点已被覆盖，终止选点")
                    break
                
                # 计算覆盖效率 - 直接从反向索引数据获取
                station_coverage = defaultdict(int)
                for d_idx, stations in reverse_index_data.items():
                    if uncovered[d_idx]:
                        for s in stations:
                            station_coverage[s] += 1
                
                # 选择最佳站点
                best_station = None
                best_integrated_efficiency = -1
                best_coverage_efficiency = 0
                best_connectivity_efficiency = 0
                best_normalized_coverage = 0
                best_normalized_connectivity = 0
                
                # 第一遍：找到最大效率值用于归一化
                max_coverage_efficiency = 0
                max_connectivity_efficiency = 0
                valid_candidates = []
                
                for station_idx in range(len(station_coords)):
                    if station_idx in selected_stations:
                        continue
                    
                    coverage_count = station_coverage[station_idx]
                    connectivity_count = np.sum(connectivity_matrix[station_idx, :])
                    cost = station_heights[station_idx] * self.cost_multiplier + self.fixed_cost
                    
                    if cost <= 0 or coverage_count == 0:
                        continue
                    
                    coverage_efficiency, connectivity_efficiency = improved_normalization(
                        station_idx, station_coverage, connectivity_count,
                        uncovered_count, total_candidates, cost, total_demand_points
                    )
                    
                    max_coverage_efficiency = max(max_coverage_efficiency, coverage_efficiency)
                    max_connectivity_efficiency = max(max_connectivity_efficiency, connectivity_efficiency)
                    valid_candidates.append(station_idx)
                
                if not valid_candidates:
                    print("没有有效的候选站点，提前终止")
                    break
                
                # 避免除零错误
                if max_coverage_efficiency == 0:
                    max_coverage_efficiency = 1
                if max_connectivity_efficiency == 0:
                    max_connectivity_efficiency = 1
                
                # 第二遍：选择最佳站点
                for station_idx in valid_candidates:
                    coverage_count = station_coverage[station_idx]
                    connectivity_count = np.sum(connectivity_matrix[station_idx, :])
                    cost = station_heights[station_idx] * self.cost_multiplier + self.fixed_cost
                    
                    coverage_efficiency, connectivity_efficiency = improved_normalization(
                        station_idx, station_coverage, connectivity_count,
                        uncovered_count, total_candidates, cost, total_demand_points
                    )
                    
                    normalized_coverage = coverage_efficiency / max_coverage_efficiency
                    normalized_connectivity = connectivity_efficiency / max_connectivity_efficiency
                    
                    integrated_efficiency = (
                        coverage_weight * normalized_coverage + 
                        connectivity_weight * normalized_connectivity
                    )
                    
                    if integrated_efficiency > best_integrated_efficiency:
                        best_integrated_efficiency = integrated_efficiency
                        best_station = station_idx
                        best_coverage_efficiency = coverage_efficiency
                        best_connectivity_efficiency = connectivity_efficiency
                        best_normalized_coverage = normalized_coverage
                        best_normalized_connectivity = normalized_connectivity
                
                # 记录效率数据
                iteration_data['coverage_efficiencies'].append(best_coverage_efficiency)
                iteration_data['connectivity_efficiencies'].append(best_connectivity_efficiency)
                iteration_data['normalized_coverage_eff'].append(best_normalized_coverage)
                iteration_data['normalized_connectivity_eff'].append(best_normalized_connectivity)
                
                if best_station is None:
                    print("无更多可选的候选站点")
                    break
                
                # 添加最佳站点
                selected_stations.append(best_station)
                
                # 高亮显示新选中的站点
                if visualizer:
                    visualizer.highlight_new_station(best_station, station_coverage[best_station])
                
                # 更新覆盖状态 - 直接从反向索引数据更新
                update_count = 0
                for d_idx, stations in reverse_index_data.items():
                    if best_station in stations and uncovered[d_idx]:
                        uncovered[d_idx] = False
                        update_count += 1
                
                # 显示选中站点的详细信息
                connectivity_count = np.sum(connectivity_matrix[best_station, :])
                cost = station_heights[best_station] * self.cost_multiplier + self.fixed_cost
                
                print(f"✅ 选中站点 {best_station}")
                print(f"  - 综合效率: {best_integrated_efficiency:.6f}")
                print(f"  - 覆盖效率: {best_coverage_efficiency:.6f} (归一化后: {best_normalized_coverage:.6f})")
                print(f"  - 连通效率: {best_connectivity_efficiency:.6f} (归一化后: {best_normalized_connectivity:.6f})")
                print(f"  - 覆盖 {station_coverage[best_station]} 个新需求点")
                print(f"  - 更新了 {update_count} 个需求点的覆盖状态")
                print(f"  - 与 {connectivity_count} 个候选站点连通")
                print(f"  - 建设成本: {cost:,.0f} 元")

        finally:
            # 清理临时文件
            temp_cache_dir = "temp_demand_cache"
            if os.path.exists(temp_cache_dir):
                shutil.rmtree(temp_cache_dir, ignore_errors=True)
                print("✅ 临时缓存文件已清理")
            
            # 关闭可视化
            if visualizer:
                visualizer.close()
        
        total_time = time.time() - start_time
        print(f"迭代选点完成，总耗时: {total_time:.2f}秒")
        print(f"最终迭代次数: {iteration}, 最终覆盖率: {current_coverage:.2%}")
        
        return selected_stations, iteration_data, reverse_index_data, connectivity_matrix

    def build_reverse_index(self, data_hash, station_coords, demand_coords,
                           building_coords, building_heights, building_elevations, building_areas,
                           station_heights, station_elevations, demand_elevations,
                           batch_size=5000):
        """
        构建反向索引 - 修复版本
        """
        print("构建反向索引（使用多进程并行计算）...")
        
        # 尝试加载缓存
        cached_reverse_index, _ = self.cache_manager.load_reverse_index_cache(data_hash)
        if cached_reverse_index is not None:
            print("✅ 使用缓存的反向索引数据")
            return cached_reverse_index
        
        # 创建临时缓存目录
        temp_cache_dir = os.path.abspath("temp_demand_cache")
        if os.path.exists(temp_cache_dir):
            shutil.rmtree(temp_cache_dir, ignore_errors=True)
        os.makedirs(temp_cache_dir, exist_ok=True)
        
        # 存储所有反向索引数据
        all_reverse_index = {}
        batch_files_info = []
        
        # 创建进程池
        cpu_count = max(1, mp.cpu_count() - 1)
        print(f"使用 {cpu_count} 个进程进行反向索引计算")
        
        # 外层进度条：批次处理
        batch_range = range(0, len(demand_coords), batch_size)
        batch_pbar = tqdm(batch_range, desc="构建反向索引", unit="批", ncols=100)
        
        for start in batch_pbar:
            end = min(start + batch_size, len(demand_coords))
            coords_batch = demand_coords[start:end]
            
            # 准备任务参数
            tasks = []
            for i in range(len(coords_batch)):
                d_idx = start + i
                demand_point = coords_batch[i]
                tasks.append((d_idx, demand_point, station_coords, 
                             building_coords, building_heights, building_elevations, building_areas,
                             demand_coords, demand_elevations, station_heights, station_elevations))
            
            batch_data = {}
            
            # 使用进程池并行处理
            with mp.Pool(processes=cpu_count) as pool:
                results_pbar = tqdm(total=len(tasks), desc=f"批次 {start//batch_size + 1}", 
                                  leave=False, unit="点", ncols=80)
                
                for result in pool.imap_unordered(self.compute_demand_coverage, tasks):
                    d_idx, valid_stations = result
                    if valid_stations:
                        batch_data[d_idx] = valid_stations
                        # 同时更新总的反向索引
                        all_reverse_index[d_idx] = valid_stations
                    results_pbar.update(1)
                    results_pbar.set_description(f"批次 {start//batch_size + 1} (已处理 {results_pbar.n}/{len(tasks)})")
                
                results_pbar.close()
            
            # 保存批次数据到文件（用于当前会话）
            batch_filename = f"batch_{start}_{end}.pkl"
            batch_filepath = os.path.join(temp_cache_dir, batch_filename)
            
            os.makedirs(os.path.dirname(batch_filepath), exist_ok=True)
            
            with open(batch_filepath, "wb") as f:
                pickle.dump(batch_data, f)
            
            batch_files_info.append({
                'filename': batch_filename,
                'start': start,
                'end': end,
                'filepath': batch_filepath
            })
            
            batch_pbar.set_description(f"构建反向索引 (已完成 {len(batch_files_info)} 批)")
        
        batch_pbar.close()
        
        # 保存完整的反向索引到缓存
        self.cache_manager.save_reverse_index_cache(data_hash, all_reverse_index, batch_files_info)
        
        print(f"✅ 反向索引构建完成，共 {len(all_reverse_index)} 个需求点的覆盖信息")
        
        return all_reverse_index

    def compute_demand_coverage(self, args):
        """
        多进程工作函数：计算单个需求点的覆盖站点
        """
        # 解包所有需要的参数
        (demand_idx, demand_point, station_coords, building_coords, 
         building_heights, building_elevations, building_areas, 
         demand_coords, demand_elevations, station_heights, station_elevations) = args
        
        valid_stations = []
        
        # 使用可达性计算器检查每个站点
        for station_idx in range(len(station_coords)):
            is_reachable, _, _, _ = self.reachability_calc.calculate_reachability(
                station_idx, demand_point, 'cover',
                building_coords, building_heights, building_elevations, building_areas,
                station_coords, station_heights, station_elevations, demand_coords, demand_elevations
            )
            if is_reachable:
                valid_stations.append(station_idx)
        
        return (demand_idx, valid_stations)

    def compute_connectivity_matrix(self, data_hash, station_coords, station_heights, station_elevations,
                                  building_coords, building_heights, building_elevations, building_areas,
                                  relay_radii, demand_coords, demand_elevations, batch_size=2000):
        """
        分批 + 多进程 加速可达性连通矩阵计算
        """
        # 1. 尝试读取缓存
        cached = self.cache_manager.load_connectivity_matrix_cache(data_hash)
        if cached is not None:
            return cached

        print("开始构建连通矩阵（多进程 + 可达函数 + 分批）...")

        n = len(station_coords)
        connectivity = np.zeros((n, n), dtype=bool)

        # KDTree 预筛选
        station_tree = KDTree(station_coords)
        max_radius = np.max(relay_radii)

        cpu_count = max(1, mp.cpu_count() - 1)
        print(f"使用 {cpu_count} 个进程")

        # 外层进度条：分批处理
        batch_range = range(0, n, batch_size)
        batch_pbar = tqdm(batch_range, desc="构建连通矩阵", unit="批", ncols=100)

        # --- 分批 ---
        for start in batch_pbar:
            end = min(start + batch_size, n)

            tasks = []

            # 内层进度条：当前批次内任务构建
            task_pbar = tqdm(range(start, end), desc=f"批次 {start//batch_size + 1}", 
                            leave=False, unit="站", ncols=80)

            # 构造任务列表
            for i in task_pbar:
                # 预筛选临近点（避免全 N²）
                candidates = station_tree.query_ball_point(station_coords[i], r=max_radius)

                for j in candidates:
                    if j <= i:   # 保持对称矩阵只算上三角
                        continue

                    tasks.append((
                        i, j,
                        station_coords, station_heights, station_elevations,
                        building_coords, building_heights, building_elevations, building_areas,
                        demand_coords, demand_elevations
                    ))

                # 更新内层进度条描述
                task_pbar.set_description(f"批次 {start//batch_size + 1} (站 {i-start+1}/{end-start})")

            task_pbar.close()

            if not tasks:
                continue

            # 使用进程池处理任务
            with mp.Pool(processes=cpu_count) as pool:
                # 处理任务进度条
                process_pbar = tqdm(total=len(tasks), desc=f"处理批次 {start//batch_size + 1}", 
                                   leave=False, unit="任务", ncols=80)

                # 使用imap_unordered获取结果
                results = []
                for result in pool.imap_unordered(self.connectivity_worker, tasks):
                    results.append(result)
                    process_pbar.update(1)
                    process_pbar.set_description(f"处理批次 {start//batch_size + 1} ({process_pbar.n}/{len(tasks)})")

                process_pbar.close()

            # 写入矩阵
            for (i, j, flag) in results:
                if flag:
                    connectivity[i, j] = True
                    connectivity[j, i] = True

            # 更新外层进度条描述
            completed_stations = min(end, n)
            batch_pbar.set_description(f"构建连通矩阵 (已完成 {completed_stations}/{n} 站点)")

        batch_pbar.close()

        print("正在保存连通矩阵缓存...")
        self.cache_manager.save_connectivity_matrix_cache(data_hash, connectivity)

        print("连通矩阵构建完成")
        return connectivity

    def connectivity_worker(self, args):
        """
        单个任务：判断 station_i ↔ station_j 是否双向可达
        """
        (i, j, station_coords, station_heights, station_elevations,
         building_coords, building_heights, building_elevations, building_areas,
         demand_coords, demand_elevations) = args

        # 检查 i -> j 的可达性
        reachable_ij, _, _, _ = self.reachability_calc.calculate_reachability(
            i, station_coords[j], 'relay',
            building_coords, building_heights, building_elevations, building_areas,
            station_coords, station_heights, station_elevations, demand_coords, demand_elevations
        )

        if not reachable_ij:
            return (i, j, False)

        # 检查 j -> i 的可达性
        reachable_ji, _, _, _ = self.reachability_calc.calculate_reachability(
            j, station_coords[i], 'relay',
            building_coords, building_heights, building_elevations, building_areas,
            station_coords, station_heights, station_elevations, demand_coords, demand_elevations
        )

        return (i, j, reachable_ji)

    def solve(self, aggregated_data, grid_original_info, city_shapefile, physical_params, drone_params, model_params):
        """
        完整的优化算法流程
        """
        print("开始执行低空起降点选址优化算法...")
        overall_start_time = time.time()

        # 性能数据收集
        performance_data = {}

        # 初始化可达性计算器
        self.reachability_calc = ReachabilityCalculator(physical_params, drone_params, model_params)
        
        # 计算高建筑KDTree
        building_total_heights = aggregated_data['building_heights'] - aggregated_data['building_elevations']
        high_building_mask = building_total_heights > drone_params.get('cruise_altitude', 200)
        high_building_indices = np.where(high_building_mask)[0]
        
        if len(high_building_indices) > 0:
            high_building_coords = aggregated_data['building_coords'][high_building_indices]
            high_building_kdtree = KDTree(high_building_coords)
            self.reachability_calc.set_high_building_kdtree(high_building_kdtree, high_building_indices)
            print(f"高建筑KDTree构建完成，共{len(high_building_coords)}个高度超过{drone_params.get('cruise_altitude', 200)}米的建筑")
        
        # 计算服务半径
        cover_radii = self.reachability_calc.compute_cover_radii(
            aggregated_data['station_coords'], 
            aggregated_data['station_heights'], 
            aggregated_data['station_elevations']
        )
        relay_radii = self.reachability_calc.compute_relay_radii(
            aggregated_data['station_coords'], 
            aggregated_data['station_heights'], 
            aggregated_data['station_elevations']
        )

        # 步骤1-2: 基于综合效率迭代选点
        print("=== 算法计算阶段（使用集计数据）===")
        selected_stations, iteration_data, reverse_index_data, connectivity_matrix = self.integrated_efficiency_selection_with_aggregated_data(
            aggregated_data, grid_original_info,
            cover_radii, relay_radii, city_shapefile,
            enable_realtime_visualization=True
        )

        performance_data['iteration'] = iteration_data
        print(f"初始选点完成，选中 {len(selected_stations)} 个集计起降点")

        # 步骤3: 全局连通性修复
        selected_stations, repair_data = self.mst_connectivity_repair_with_aggregated_data(
            selected_stations, aggregated_data, grid_original_info
        )

        performance_data['repair'] = repair_data
        print(f"连通性修复后，共有 {len(selected_stations)} 个起降点")

        # 步骤4: 冗余剪枝
        selected_stations, prune_data = self.redundancy_pruning_with_aggregated_data(
            selected_stations, aggregated_data, grid_original_info
        )

        performance_data['prune'] = prune_data
        print(f"冗余剪枝后，最终选中 {len(selected_stations)} 个起降点")

        # 总时间统计
        total_time = time.time() - overall_start_time
        performance_data['total_time'] = total_time
        print(f"算法总运行时间: {total_time:.2f}秒")

        # 返回结果
        results = {
            'selected_stations': selected_stations,
            'performance_data': performance_data,
            'reverse_index_data': reverse_index_data,
            'connectivity_matrix': connectivity_matrix,
            'cover_radii': cover_radii,
            'relay_radii': relay_radii
        }

        return results

    def mst_connectivity_repair_with_aggregated_data(self, selected_stations, aggregated_data, grid_original_info):
        """
        基于Prim算法最小生成树的连通性修复（使用集计数据）
        """
        print("=== 步骤3: 基于最小生成树的连通性修复（使用集计数据） ===")
        start_time = time.time()
        
        if len(selected_stations) <= 1:
            print("起降点数量不足，无需连通性修复")
            return selected_stations, {'repair_time': 0, 'added_stations': 0}
        
        # 从集计数据中提取变量
        station_coords = aggregated_data['station_coords']
        station_heights = aggregated_data['station_heights']
        station_elevations = aggregated_data['station_elevations']
        building_coords = aggregated_data['building_coords']
        building_heights = aggregated_data['building_heights']
        building_elevations = aggregated_data['building_elevations']
        building_areas = aggregated_data['building_areas']
        demand_coords = aggregated_data['demand_coords']
        demand_elevations = aggregated_data['demand_elevations']
        
        # 辅助函数：计算连通效率
        def get_connectivity_efficiency(station_idx):
            cost = station_heights[station_idx] * self.cost_multiplier + self.fixed_cost
            return 1.0 / cost if cost > 0 else 0.0
        
        # 辅助函数：获取双向连通的边（使用可达函数）
        def get_bidirectional_edges_with_reachability(nodes):
            edges = []
            reachability_cache = {}
            
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    node_i = nodes[i]
                    node_j = nodes[j]
                    
                    # 使用可达函数检查双向连通条件
                    cache_key1 = (node_i, node_j, 'relay')
                    cache_key2 = (node_j, node_i, 'relay')
                    
                    if cache_key1 in reachability_cache:
                        reachable1 = reachability_cache[cache_key1]
                    else:
                        reachable1, _, _, _ = self.reachability_calc.calculate_reachability(
                            node_i, station_coords[node_j], 'relay',
                            building_coords, building_heights, building_elevations, building_areas,
                            station_coords, station_heights, station_elevations, 
                            demand_coords, demand_elevations
                        )
                        reachability_cache[cache_key1] = reachable1
                    
                    if cache_key2 in reachability_cache:
                        reachable2 = reachability_cache[cache_key2]
                    else:
                        reachable2, _, _, _ = self.reachability_calc.calculate_reachability(
                            node_j, station_coords[node_i], 'relay',
                            building_coords, building_heights, building_elevations, building_areas,
                            station_coords, station_heights, station_elevations, 
                            demand_coords, demand_elevations
                        )
                        reachability_cache[cache_key2] = reachable2
                    
                    if reachable1 and reachable2:
                        distance = np.linalg.norm(station_coords[node_i] - station_coords[node_j])
                        edges.append((i, j, distance))
            return edges
        
        # Prim算法构建最小生成树
        def prim_algorithm(nodes, edges):
            n = len(nodes)
            if n == 0:
                return [], set()
                
            # 构建邻接表
            adj = {i: [] for i in range(n)}
            for i, j, weight in edges:
                adj[i].append((j, weight))
                adj[j].append((i, weight))
            
            mst_edges = []
            visited = set()
            candidate_edges = []  # 最小堆：(weight, i, j)
            
            # 从节点0开始
            visited.add(0)
            for neighbor, weight in adj[0]:
                heapq.heappush(candidate_edges, (weight, 0, neighbor))
            
            while candidate_edges and len(visited) < n:
                weight, i, j = heapq.heappop(candidate_edges)
                if j not in visited:
                    mst_edges.append((i, j, weight))
                    visited.add(j)
                    for neighbor, new_weight in adj[j]:
                        if neighbor not in visited:
                            heapq.heappush(candidate_edges, (new_weight, j, neighbor))
            
            return mst_edges, visited
        
        # 补点策略（使用可达函数）
        def repair_isolated_node_with_reachability(isolated_node, connected_component, all_stations):
            """
            为孤立节点寻找连接方案（使用可达函数）
            返回: 新增的站点列表
            """
            print(f"修复孤立节点 {isolated_node}")
            additional_stations = []
            
            # 1. 找到距离孤立节点最近的已连接节点
            min_distance = float('inf')
            closest_node = None
            
            for node in connected_component:
                distance = np.linalg.norm(station_coords[isolated_node] - station_coords[node])
                if distance < min_distance:
                    min_distance = distance
                    closest_node = node
            
            if closest_node is None:
                print(f"无法找到孤立节点 {isolated_node} 的最近连接点")
                return additional_stations
            
            print(f"孤立节点 {isolated_node} 的最近连接点: {closest_node}, 距离: {min_distance:.2f}m")
            
            # 2. 收集两起降点各自的接力任务可达范围内的可互通的未选中起降点
            def get_connectable_candidates_with_reachability(station, exclude_stations):
                candidates = []
                for candidate in range(len(station_coords)):
                    if candidate in exclude_stations:
                        continue
                    
                    # 使用可达函数检查双向连通
                    reachable1, _, _, _ = self.reachability_calc.calculate_reachability(
                        station, station_coords[candidate], 'relay',
                        building_coords, building_heights, building_elevations, building_areas,
                        station_coords, station_heights, station_elevations, 
                        demand_coords, demand_elevations
                    )
                    reachable2, _, _, _ = self.reachability_calc.calculate_reachability(
                        candidate, station_coords[station], 'relay',
                        building_coords, building_heights, building_elevations, building_areas,
                        station_coords, station_heights, station_elevations, 
                        demand_coords, demand_elevations
                    )
                    
                    if reachable1 and reachable2:
                        candidates.append(candidate)
                return candidates
            
            candidates_isolated = get_connectable_candidates_with_reachability(isolated_node, all_stations)
            candidates_closest = get_connectable_candidates_with_reachability(closest_node, all_stations)
            
            print(f"孤立节点候选点: {len(candidates_isolated)} 个")
            print(f"最近节点候选点: {len(candidates_closest)} 个")
            
            # 3. 寻找能同时覆盖两点的单个候选点（按连通效率排序）
            common_candidates = []
            for candidate in candidates_isolated:
                if candidate in candidates_closest:
                    # 验证候选点能同时连接孤立节点和最近节点
                    reachable1, _, _, _ = self.reachability_calc.calculate_reachability(
                        candidate, station_coords[isolated_node], 'relay',
                        building_coords, building_heights, building_elevations, building_areas,
                        station_coords, station_heights, station_elevations, 
                        demand_coords, demand_elevations
                    )
                    reachable2, _, _, _ = self.reachability_calc.calculate_reachability(
                        candidate, station_coords[closest_node], 'relay',
                        building_coords, building_heights, building_elevations, building_areas,
                        station_coords, station_heights, station_elevations, 
                        demand_coords, demand_elevations
                    )
                    
                    if reachable1 and reachable2:
                        efficiency = get_connectivity_efficiency(candidate)
                        common_candidates.append((efficiency, candidate))
            
            common_candidates.sort(reverse=True)
            
            # 3.1 如果能找到单个候选点
            if common_candidates:
                best_efficiency, best_candidate = common_candidates[0]
                additional_stations.append(best_candidate)
                print(f"✅ 找到单个候选点 {best_candidate}，连通效率: {best_efficiency:.6f}")
                return additional_stations
            
            # 3.2 如果不能，寻找候选点组
            print("未找到单个候选点，尝试寻找候选点组")
            
            candidate_pairs = []
            for cand_i in candidates_isolated:
                for cand_j in candidates_closest:
                    if cand_i != cand_j:
                        # 检查两个候选点之间是否能连通
                        reachable, _, _, _ = self.reachability_calc.calculate_reachability(
                            cand_i, station_coords[cand_j], 'relay',
                            building_coords, building_heights, building_elevations, building_areas,
                            station_coords, station_heights, station_elevations, 
                            demand_coords, demand_elevations
                        )
                        
                        if reachable:
                            efficiency_i = get_connectivity_efficiency(cand_i)
                            efficiency_j = get_connectivity_efficiency(cand_j)
                            total_efficiency = efficiency_i + efficiency_j
                            candidate_pairs.append((total_efficiency, cand_i, cand_j))
            
            candidate_pairs.sort(reverse=True)
            
            # 3.2.1 如果能找到候选点组
            if candidate_pairs:
                best_efficiency, best_cand_i, best_cand_j = candidate_pairs[0]
                additional_stations.extend([best_cand_i, best_cand_j])
                print(f"✅ 找到候选点组 ({best_cand_i}, {best_cand_j})，总连通效率: {best_efficiency:.6f}")
                return additional_stations
            
            print(f"❌ 无法为孤立节点 {isolated_node} 找到连接方案")
            return additional_stations
        
        # 主修复逻辑
        additional_stations = []
        max_repair_iterations = 10
        special_marked_stations = set()
        
        for iteration in range(max_repair_iterations):
            print(f"\n--- 连通性修复迭代 {iteration + 1} ---")
            
            # 当前所有站点（原始选中 + 新增）
            current_stations = selected_stations + additional_stations
            print(f"当前站点总数: {len(current_stations)}")
            
            # 步骤1: 构建带权子图（只包含双向连通的边）
            edges = get_bidirectional_edges_with_reachability(current_stations)
            print(f"构建带权子图，包含 {len(edges)} 条双向连通边")
            
            # 步骤2: Prim算法构建最小生成树
            mst_edges, visited_nodes = prim_algorithm(current_stations, edges)
            print(f"最小生成树包含 {len(visited_nodes)}/{len(current_stations)} 个节点")
            
            # 步骤3: 检查连通性
            if len(visited_nodes) == len(current_stations):
                print("✅ 网络已全连通！")
                break
            else:
                # 找到孤立节点
                all_nodes = set(range(len(current_stations)))
                isolated_indices = all_nodes - visited_nodes
                isolated_nodes = [current_stations[i] for i in isolated_indices]
                
                print(f"发现 {len(isolated_nodes)} 个孤立节点: {isolated_nodes}")
                
                # 构建已连接分量的节点集合
                connected_nodes = [current_stations[i] for i in visited_nodes]
                
                # 为每个孤立节点寻找连接方案
                new_additions = []
                for isolated_node in isolated_nodes:
                    if isolated_node in special_marked_stations:
                        print(f"跳过特殊标记的孤立节点 {isolated_node}")
                        continue
                    
                    repair_result = repair_isolated_node_with_reachability(isolated_node, connected_nodes, current_stations)
                    new_additions.extend(repair_result)
                
                if new_additions:
                    # 去重
                    new_additions = list(set(new_additions) - set(additional_stations))
                    additional_stations.extend(new_additions)
                    print(f"本轮新增 {len(new_additions)} 个站点")
                else:
                    print("本轮未能找到新的连接方案，修复完成")
                    break
        
        # 最终结果
        final_stations = selected_stations + additional_stations
        
        # 最终连通性验证
        final_edges = get_bidirectional_edges_with_reachability(final_stations)
        final_mst_edges, final_visited = prim_algorithm(final_stations, final_edges)
        
        if len(final_visited) == len(final_stations):
            print("🎉 连通性修复成功！网络已全连通")
        else:
            remaining_isolated = len(final_stations) - len(final_visited)
            print(f"⚠️  连通性修复后仍有 {remaining_isolated} 个孤立节点")
        
        repair_time = time.time() - start_time
        added_count = len(additional_stations)
        
        print(f"\n连通性修复完成:")
        print(f"- 耗时: {repair_time:.2f}秒")
        print(f"- 新增站点: {added_count}个")
        print(f"- 最终站点总数: {len(final_stations)}个")
        
        return final_stations, {
            'repair_time': repair_time, 
            'added_stations': added_count,
            'final_station_count': len(final_stations)
        }

    def redundancy_pruning_with_aggregated_data(self, selected_stations, aggregated_data, grid_original_info,
                                              coverage_tolerance=0.000):
        """
        冗余剪枝算法 - 使用集计数据
        修改：只要剪枝后覆盖率依然大于目标覆盖率且全连通，就可以剪枝
        """
        print("=== 步骤4: 冗余剪枝（使用集计数据） ===")
        start_time = time.time()

        if len(selected_stations) <= 1:
            return selected_stations, {'prune_time': 0, 'removed_stations': 0}

        # 从集计数据中提取变量
        station_coords = aggregated_data['station_coords']
        demand_coords = aggregated_data['demand_coords']
        building_coords = aggregated_data['building_coords']
        building_heights = aggregated_data['building_heights']
        building_elevations = aggregated_data['building_elevations']
        building_areas = aggregated_data['building_areas']
        station_heights = aggregated_data['station_heights']
        station_elevations = aggregated_data['station_elevations']
        demand_elevations = aggregated_data['demand_elevations']

        # ---------------------------
        # Step 1: 计算初始覆盖率（使用可达函数）
        # ---------------------------
        print("计算初始覆盖率（使用可达函数）...")
        
        # 计算初始覆盖率
        covered_count = 0
        reachability_cache = {}
        
        for i, demand_point in enumerate(tqdm(demand_coords, desc="计算初始覆盖率", unit="点")):
            is_covered = False
            for station_idx in selected_stations:
                cache_key = (station_idx, i, 'cover')
                
                if cache_key in reachability_cache:
                    is_reachable = reachability_cache[cache_key]
                else:
                    is_reachable, _, _, _ = self.reachability_calc.calculate_reachability(
                        station_idx, demand_point, 'cover',
                        building_coords, building_heights, building_elevations, building_areas,
                        station_coords, station_heights, station_elevations,
                        demand_coords, demand_elevations
                    )
                    reachability_cache[cache_key] = is_reachable
                
                if is_reachable:
                    is_covered = True
                    break
            
            if is_covered:
                covered_count += 1
        
        initial_coverage = covered_count / len(demand_coords)
        print(f"剪枝前覆盖率: {initial_coverage:.4%}")
        
        # 构建选中站点的连通图（使用可达函数）
        def get_connectivity_graph(nodes):
            G = nx.Graph()
            for i, station_idx in enumerate(nodes):
                G.add_node(i)  # 使用索引作为节点ID
            
            # 添加边
            reachability_cache = {}
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    s1, s2 = nodes[i], nodes[j]
                    
                    cache_key1 = (s1, s2, 'relay')
                    cache_key2 = (s2, s1, 'relay')
                    
                    if cache_key1 in reachability_cache:
                        reachable1 = reachability_cache[cache_key1]
                    else:
                        reachable1, _, _, _ = self.reachability_calc.calculate_reachability(
                            s1, station_coords[s2], 'relay',
                            building_coords, building_heights, building_elevations, building_areas,
                            station_coords, station_heights, station_elevations,
                            demand_coords, demand_elevations                                      
                        )
                        reachability_cache[cache_key1] = reachable1
                    
                    if cache_key2 in reachability_cache:
                        reachable2 = reachability_cache[cache_key2]
                    else:
                        reachable2, _, _, _ = self.reachability_calc.calculate_reachability(
                            s2, station_coords[s1], 'relay',
                            building_coords, building_heights, building_elevations, building_areas,
                            station_coords, station_heights, station_elevations,
                            demand_coords, demand_elevations  
                        )
                        reachability_cache[cache_key2] = reachable2
                    
                    if reachable1 and reachable2:
                        G.add_edge(i, j)
            return G

        G = get_connectivity_graph(selected_stations)

        # ---------------------------
        # Step 2: 构建精确的覆盖映射（使用可达函数）
        # ---------------------------
        print("构建精确的覆盖映射（使用可达函数）...")
        
        # 记录每个需求点被哪些站点覆盖
        demand_covered_by = defaultdict(list)
        reachability_cache = {}
        
        for d_idx, demand_point in enumerate(tqdm(demand_coords, desc="构建覆盖映射", unit="点")):
            for station_idx in selected_stations:
                cache_key = (station_idx, d_idx, 'cover')
                
                if cache_key in reachability_cache:
                    is_reachable = reachability_cache[cache_key]
                else:
                    is_reachable, _, _, _ = self.reachability_calc.calculate_reachability(
                        station_idx, demand_point, 'cover',
                        building_coords, building_heights, building_elevations, building_areas,
                        station_coords, station_heights, station_elevations,
                        demand_coords, demand_elevations
                    )
                    reachability_cache[cache_key] = is_reachable
                
                if is_reachable:
                    demand_covered_by[d_idx].append(station_idx)
        
        # 计算每个站点的独占覆盖需求点
        station_unique_coverage = {}
        for station_idx in selected_stations:
            unique_count = 0
            for d_idx, covering_stations in demand_covered_by.items():
                if len(covering_stations) == 1 and station_idx in covering_stations:
                    unique_count += 1
            station_unique_coverage[station_idx] = unique_count

        # ---------------------------
        # Step 3: 按成本降序尝试移除
        # ---------------------------
        print("开始冗余剪枝...")
        
        # 按成本排序（从高到低）
        station_costs = []
        for station_idx in selected_stations:
            cost = station_heights[station_idx] * self.cost_multiplier + self.fixed_cost
            station_costs.append((cost, station_idx))
        
        station_costs.sort(reverse=True)

        pruned_stations = set(selected_stations)
        removed_stations = []
        
        progress_bar = tqdm(station_costs, desc="剪枝进度", unit="点", ncols=100)
        
        for cost, station_idx in progress_bar:
            if station_idx not in pruned_stations:
                continue
                
            if len(pruned_stations) <= 1:
                break

            # 模拟移除：检查连通性和覆盖率
            temp_stations = list(pruned_stations - {station_idx})
            
            # 检查连通性：移除后网络是否仍然连通
            temp_indices = [selected_stations.index(s) for s in temp_stations]
            temp_graph = G.subgraph(temp_indices)
            is_connected = nx.is_connected(temp_graph) if len(temp_indices) > 1 else True
            
            if not is_connected:
                continue
                
            # 检查覆盖率：移除后覆盖率是否满足要求（使用可达函数）
            temp_covered_count = 0
            temp_reachability_cache = {}
            
            for i, demand_point in enumerate(demand_coords):
                is_covered = False
                for temp_station in temp_stations:
                    cache_key = (temp_station, i, 'cover')
                    
                    if cache_key in temp_reachability_cache:
                        is_reachable = temp_reachability_cache[cache_key]
                    else:
                        is_reachable, _, _, _ = self.reachability_calc.calculate_reachability(
                            temp_station, demand_point, 'cover',
                            building_coords, building_heights, building_elevations, building_areas,
                            station_coords, station_heights, station_elevations,
                            demand_coords, demand_elevations
                        )
                        temp_reachability_cache[cache_key] = is_reachable
                    
                    if is_reachable:
                        is_covered = True
                        break
                
                if is_covered:
                    temp_covered_count += 1
            
            temp_coverage = temp_covered_count / len(demand_coords)
            
            # 修改条件：只要剪枝后覆盖率依然大于目标覆盖率且全连通，就可以剪枝
            if temp_coverage >= self.target_coverage - coverage_tolerance and is_connected:
                # 可以安全移除
                pruned_stations.remove(station_idx)
                removed_stations.append(station_idx)
                
                # 更新进度条描述
                progress_bar.set_description(f"剪枝进度 (移除{len(removed_stations)}个)")
                
                print(f"✅ 移除站点 {station_idx} (成本: {cost:,.0f}元)")
                print(f"   移除后覆盖率: {temp_coverage:.4%} (目标覆盖率: {self.target_coverage:.4%})")
                print(f"   网络连通性: {'保持连通' if is_connected else '断开'}")

        # ---------------------------
        # Step 4: 最终验证
        # ---------------------------
        final_stations = list(pruned_stations)
        
        # 计算最终覆盖率（使用可达函数）
        final_covered_count = 0
        final_reachability_cache = {}
        
        for i, demand_point in enumerate(demand_coords):
            is_covered = False
            for station_idx in final_stations:
                cache_key = (station_idx, i, 'cover')
                
                if cache_key in final_reachability_cache:
                    is_reachable = final_reachability_cache[cache_key]
                else:
                    is_reachable, _, _, _ = self.reachability_calc.calculate_reachability(
                        station_idx, demand_point, 'cover',
                        building_coords, building_heights, building_elevations, building_areas,
                        station_coords, station_heights, station_elevations,
                        demand_coords, demand_elevations
                    )
                    final_reachability_cache[cache_key] = is_reachable
                
                if is_reachable:
                    is_covered = True
                    break
            
            if is_covered:
                final_covered_count += 1
        
        final_coverage = final_covered_count / len(demand_coords)
        prune_time = time.time() - start_time
        
        # 验证最终连通性
        final_graph = get_connectivity_graph(final_stations)
        is_final_connected = nx.is_connected(final_graph) if len(final_stations) > 1 else True
        
        print(f"\n剪枝完成: {len(selected_stations)} → {len(final_stations)} 个站点")
        print(f"移除了 {len(removed_stations)} 个冗余站点")
        print(f"最终覆盖率: {final_coverage:.4%} (目标: {self.target_coverage:.4%})")
        print(f"网络连通性: {'保持连通' if is_final_connected else '断开'}")
        print(f"冗余剪枝耗时: {prune_time:.2f}秒")
        
        # 验证覆盖率没有显著降低
        coverage_loss = initial_coverage - final_coverage
        if final_coverage >= self.target_coverage - coverage_tolerance and is_final_connected:
            print(f"✅ 剪枝成功: 覆盖率满足要求且网络保持连通")
        else:
            print(f"❌ 剪枝失败: 覆盖率或连通性不满足要求")
        
        return final_stations, {
            'prune_time': prune_time, 
            'removed_stations': len(removed_stations),
            'coverage_loss': coverage_loss,
            'initial_coverage': initial_coverage,
            'final_coverage': final_coverage,
            'is_connected': is_final_connected
        }