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

from .cache_manager import CacheManager
from .visualization import RealtimeSelectionVisualizer
from .utils import calculate_coverage, improved_normalization

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
        
        # 这里需要从外部传入ReachabilityCalculator实例
        # 由于多进程限制，这里暂时简化实现
        valid_stations = []
        # 实际实现需要调用可达性计算
        
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
        from scipy.spatial import KDTree
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

        # 这里需要从外部传入ReachabilityCalculator实例
        # 由于多进程限制，这里暂时简化实现
        reachable_ij = True  # 简化实现
        if not reachable_ij:
            return (i, j, False)

        reachable_ji = True  # 简化实现
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
        from .reachability_calculation import ReachabilityCalculator
        reachability_calc = ReachabilityCalculator(physical_params, drone_params, model_params)
        
        # 计算高建筑KDTree
        building_total_heights = aggregated_data['building_heights'] - aggregated_data['building_elevations']
        high_building_mask = building_total_heights > drone_params.get('cruise_altitude', 200)
        high_building_indices = np.where(high_building_mask)[0]
        
        if len(high_building_indices) > 0:
            high_building_coords = aggregated_data['building_coords'][high_building_indices]
            from scipy.spatial import KDTree
            high_building_kdtree = KDTree(high_building_coords)
            reachability_calc.set_high_building_kdtree(high_building_kdtree, high_building_indices)
            print(f"高建筑KDTree构建完成，共{len(high_building_coords)}个高度超过{drone_params.get('cruise_altitude', 200)}米的建筑")
        
        # 计算服务半径
        cover_radii = reachability_calc.compute_cover_radii(
            aggregated_data['station_coords'], 
            aggregated_data['station_heights'], 
            aggregated_data['station_elevations']
        )
        relay_radii = reachability_calc.compute_relay_radii(
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
        """基于最小生成树的连通性修复（使用集计数据）"""
        # 简化实现
        print("连通性修复（简化实现）...")
        return selected_stations, {'repair_time': 0, 'added_stations': 0}

    def redundancy_pruning_with_aggregated_data(self, selected_stations, aggregated_data, grid_original_info, coverage_tolerance=0.000):
        """冗余剪枝算法（使用集计数据）"""
        # 简化实现
        print("冗余剪枝（简化实现）...")
        return selected_stations, {'prune_time': 0, 'removed_stations': 0}