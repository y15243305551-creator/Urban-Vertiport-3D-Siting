import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import networkx as nx
import heapq
import math
import pandas as pd
from scipy.spatial import KDTree
from collections import defaultdict
from pyproj import Proj
from tqdm import tqdm
import pickle
import os
from bitarray import bitarray
import shutil 
import time
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import geopandas as gpd
from scipy.interpolate import griddata
import vispy
vispy.use(app="pyqt5")
from vispy import scene, app
from vispy.visuals import transforms
from vispy.color import Colormap
import psutil
import matplotlib.animation as animation
import multiprocessing as mp

# 字体设置（只在主进程中执行）
if __name__ == '__main__':
    try:
        # 尝试设置中文字体
        plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 测试中文显示
        test_fig, test_ax = plt.subplots(figsize=(1, 1))
        test_ax.text(0.5, 0.5, '测试', fontsize=12, ha='center', va='center')
        plt.close(test_fig)
        print("中文字体设置成功")
        
        # 设置全局使用中文
        USE_CHINESE = True
    except Exception as e:
        print(f"中文字体设置失败: {e}")
        print("将使用英文标签")
        USE_CHINESE = False
else:
    # 子进程中的设置
    USE_CHINESE = False

# ========================== 1. 参数计算 ==========================
# 物理参数
pi = math.pi  # 圆周率
g = 9.81  # 重力加速度，单位：m/s^2
v_w = -10  # 风速，单位：m/s
b = 200  # 无人机平飞高度，单位：m
rho = 1.225  # 空气密度 单位：kg/m^3

# 无人机参数
m_d = 32  # 无人机质量，单位：kg
m_p_max = 10  # 无人机最大载荷，单位：kg
v_u = 4  # 无人机起飞速度，单位：m/s
v_d = 3  # 无人机降落速度，单位：m/s
v_h = 14  # 无人机水平飞行速度，单位：m/s
A = 1.7  # 无人机横截面积，单位：m^2
Ah = 1  # 无人机平飞时大约截面积，单位：m^2
r = 0.47  # 螺旋桨半径，单位：m
Ap = pi * r**2  # 螺旋桨面积，单位：m^2
N = 8  # 螺旋桨数量
E_total = 7668320   # 总电池容量，单位：J

# 比值与系数
eta = 0.85  # 无人机电机功率传输效率
Cd = 0.7  # 无人机阻力系数
Ct = 0.04 # 无人机推力系数
electricity_consumption_rate = 0.8  # 执行任务时消耗的电量比例

# 模型参数
cover_K = 1000
relay_K = 1000
cost_multiplier=5000   # 成本比例系数
fixed_cost = 150000  # 固定成本
target_coverage_percentage = 0.95   # 覆盖率
elevation_min = -41   # 最低高程高度
MAX_OBSTACLE_COUNT = 10  # 最大避障次数限制
MAX_OBSTACLE_THRESHOLD = 10
grid_size = 300  # 定义grid_size

# 数据文件路径
District = '全部区'
city_shapefile = f"geojson/{District}.geojson"
station_file = f"stations/{District}stations.xlsx"
demand_file = f"demand_points/{District}demand_points.xlsx"
elevation_file = "等高线.geojson"
building_file = "building.xlsx"

# 坐标转换函数
def gcj02_to_utm(lons, lats):
    proj = Proj(
        proj='utm',
        zone=54,
        ellips='krass',
        towgs84='0,0,0,0,0,0,0',
        preserve_units=True
    )
    if isinstance(lons, (list, tuple, np.ndarray)):
        # 批量转换
        return np.array([proj(lon, lat) for lon, lat in zip(lons, lats)])
    else:
        # 单个点转换
        return proj(lons, lats)

# 多进程工作函数
def compute_demand_coverage(args):
    """
    多进程工作函数：计算单个需求点的覆盖站点（完整修复）
    """
    # 解包所有需要的参数
    (demand_idx, demand_point, station_coords, building_coords, 
     building_heights, building_elevations, building_areas, 
     demand_coords, demand_elevations, station_heights, station_elevations) = args
    
    valid_stations = []
    for station_idx in range(len(station_coords)):
        is_reachable, _, _, _ = calculate_reachability(
            station_idx, demand_point, 'cover',
            building_coords, building_heights, building_elevations, building_areas,
            station_coords, station_heights, station_elevations, demand_coords, demand_elevations
        )
        if is_reachable:
            valid_stations.append(station_idx)
    
    return (demand_idx, valid_stations)

def connectivity_worker(args):
    """
    单个任务：判断 station_i ↔ station_j 是否双向可达（完整修复）
    """
    (i, j, station_coords, station_heights, station_elevations,
     building_coords, building_heights, building_elevations, building_areas,
     demand_coords, demand_elevations) = args

    reachable_ij, _, _, _ = calculate_reachability(
        i, station_coords[j], 'relay',
        building_coords, building_heights, building_elevations, building_areas,
        station_coords, station_heights, station_elevations, demand_coords, demand_elevations
    )

    if not reachable_ij:
        return (i, j, False)

    reachable_ji, _, _, _ = calculate_reachability(
        j, station_coords[i], 'relay',
        building_coords, building_heights, building_elevations, building_areas,
        station_coords, station_heights, station_elevations, demand_coords, demand_elevations
    )

    return (i, j, reachable_ji)

# 数据读取和初始化（放在主程序块中）
if __name__ == '__main__':
    print(f"------读取起降点和需求点数据------")

    # 读取数据
    station_df = pd.read_excel(station_file)
    demand_df = pd.read_excel(demand_file)
    building_df = pd.read_excel(building_file)

    # 提取起降点数据 - 使用向量化操作但保留进度条
    station_heights = np.array([
        row['Height'] for _, row in tqdm(station_df.iterrows(), total=station_df.shape[0], 
                                       desc="读取起降点建筑高度", unit="行", ncols=100)
    ])  
    station_elevations = np.array([
        row['DEM高度'] for _, row in tqdm(station_df.iterrows(), total=station_df.shape[0], 
                                        desc="读取起降点高程高度", unit="行", ncols=100)
    ]) 
    station_fids = np.array([
        row['fid'] for _, row in tqdm(station_df.iterrows(), total=station_df.shape[0], 
                                    desc="读取起降点fid", unit="行", ncols=100)
    ], dtype=int)  

    # 提取建筑数据（用于3D可视化）
    building_areas = np.array([
        row['roof_area'] for _, row in tqdm(building_df.iterrows(), total=building_df.shape[0], 
                                            desc="读取屋顶面积", unit="行", ncols=100)
    ])

    if District == '全部区':
        building_heights = np.array([
            row['Height'] for _, row in tqdm(building_df.iterrows(), total=building_df.shape[0], 
                                           desc="读取建筑高度", unit="行", ncols=100)
        ])  
        building_elevations = np.array([
            row['DEM高度'] for _, row in tqdm(building_df.iterrows(), total=building_df.shape[0], 
                                            desc="读取高程高度", unit="行", ncols=100)
        ]) 
        
        # 使用向量化坐标转换但保留进度条
        print("读取建筑坐标...")
        building_lons = building_df['经度'].values
        building_lats = building_df['纬度'].values
        building_coords = np.array([
            gcj02_to_utm(lon, lat) 
            for lon, lat in tqdm(zip(building_lons, building_lats), 
                               total=len(building_lons), desc="转换建筑坐标", unit="行", ncols=100)
        ])
        
        building_fids = np.array([
            row['fid'] for _, row in tqdm(building_df.iterrows(), total=building_df.shape[0], 
                                        desc="读取建筑fid", unit="行", ncols=100)
        ], dtype=int)
    else:
        building_heights = np.array([])
        building_elevations = np.array([])
        building_coords = np.array([])
        building_fids = np.array([])

    demand_elevations = np.array([
        row['DEM高度'] if 'DEM高度' in row else 0
        for _, row in tqdm(demand_df.iterrows(), total=demand_df.shape[0],
                          desc="读取需求点高程", unit="行", ncols=100)
    ])

    # 提取坐标并转换 - 使用向量化操作但保留进度条
    print("读取起降点坐标...")
    station_lons = station_df['经度'].values
    station_lats = station_df['纬度'].values
    station_coords = np.array([
        gcj02_to_utm(lon, lat) 
        for lon, lat in tqdm(zip(station_lons, station_lats), 
                           total=len(station_lons), desc="转换起降点坐标", unit="行", ncols=100)
    ])

    print("读取需求点坐标...")
    demand_lons = demand_df['经度'].values
    demand_lats = demand_df['纬度'].values
    demand_coords = np.array([
        gcj02_to_utm(lon, lat) 
        for lon, lat in tqdm(zip(demand_lons, demand_lats), 
                           total=len(demand_lons), desc="转换需求点坐标", unit="行", ncols=100)
    ])

    # ========================== 数据集计功能 ==========================

    def aggregate_points_by_grid_simplified(grid_size, min_building_height_threshold=None):
        """
        简化的网格化集计功能 - 直接选择代表性起降点
        """
        if min_building_height_threshold is None:
            min_building_height_threshold = b
        
        print(f"开始网格化集计，网格大小: {grid_size}m，超高建筑阈值: {min_building_height_threshold}m")
        print(f"起降点选择策略: 成本最低")
        
        # 计算所有数据的边界
        all_coords = []
        if len(building_coords) > 0:
            all_coords.append(building_coords)
        if len(station_coords) > 0:
            all_coords.append(station_coords)
        if len(demand_coords) > 0:
            all_coords.append(demand_coords)
        
        if not all_coords:
            print("没有数据可集计")
            return None, None, None
        
        all_coords = np.vstack(all_coords)
        min_x, min_y = np.min(all_coords, axis=0)
        max_x, max_y = np.max(all_coords, axis=0)
        
        # 创建网格
        x_bins = np.arange(min_x, max_x + grid_size, grid_size)
        y_bins = np.arange(min_y, max_y + grid_size, grid_size)
        
        print(f"网格范围: X[{min_x:.0f}, {max_x:.0f}], Y[{min_y:.0f}, {max_y:.0f}]")
        print(f"网格数量: {len(x_bins)-1} x {len(y_bins)-1} = {(len(x_bins)-1)*(len(y_bins)-1)}")
        
        # 集计建筑数据
        aggregated_buildings = aggregate_buildings_by_grid(
            building_coords, building_heights, building_elevations, building_areas, building_fids,
            x_bins, y_bins, min_building_height_threshold
        )
        
        # 集计起降点数据（直接选择代表性起降点）
        aggregated_stations, selected_mapping, grid_original_info = aggregate_stations_by_grid_with_selection(
            station_coords, station_heights, station_elevations, station_fids,
            x_bins, y_bins, station_df, strategy="lowest_cost"
        )
        
        # 集计需求点数据
        aggregated_demands = aggregate_demands_by_grid(
            demand_coords, demand_elevations,
            x_bins, y_bins
        )
        
        aggregated_data = aggregated_buildings + aggregated_stations + aggregated_demands
        
        # 返回集计数据和选中的原始站点映射
        return aggregated_data, selected_mapping, grid_original_info

    def aggregate_buildings_by_grid(building_coords, building_heights, building_elevations, 
                                   building_areas, building_fids, x_bins, y_bins, height_threshold):
        """集计建筑数据 - 分别存储高度和高程"""
        if len(building_coords) == 0:
            return [np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]
        
        # 计算建筑总高（建筑物高度-高程）用于判断超高建筑
        building_total_heights = building_heights - building_elevations
        
        # 为每个建筑分配网格索引
        x_indices = np.digitize(building_coords[:, 0], x_bins) - 1
        y_indices = np.digitize(building_coords[:, 1], y_bins) - 1
        
        # 创建网格字典
        grid_dict = {}
        high_buildings = []  # 单独存储超高建筑
        
        for i in range(len(building_coords)):
            grid_key = (x_indices[i], y_indices[i])
            
            # 检查是否为超高建筑（使用总高判断）
            if building_total_heights[i] > height_threshold:
                high_buildings.append({
                    'coord': building_coords[i],
                    'height': building_heights[i],          # 原始建筑物高度
                    'elevation': building_elevations[i],    # 高程
                    'area': building_areas[i],
                    'fid': building_fids[i]
                })
            else:
                if grid_key not in grid_dict:
                    grid_dict[grid_key] = {
                        'coords': [],
                        'heights': [],           # 原始建筑物高度
                        'elevations': [],        # 高程
                        'areas': [],
                        'fids': []
                    }
                
                grid_dict[grid_key]['coords'].append(building_coords[i])
                grid_dict[grid_key]['heights'].append(building_heights[i])
                grid_dict[grid_key]['elevations'].append(building_elevations[i])
                grid_dict[grid_key]['areas'].append(building_areas[i])
                grid_dict[grid_key]['fids'].append(building_fids[i])
        
        # 处理每个网格的普通建筑
        aggregated_coords = []
        aggregated_heights = []        # 存储原始建筑物高度
        aggregated_elevations = []     # 存储高程
        aggregated_areas = []
        aggregated_fids = []
        
        for grid_key, data in grid_dict.items():
            if len(data['coords']) > 0:
                # 计算密度重心（按面积加权）
                coords_array = np.array(data['coords'])
                areas_array = np.array(data['areas'])
                
                # 面积加权平均坐标
                weighted_coords = np.average(coords_array, axis=0, weights=areas_array)
                
                # 平均原始建筑物高度
                avg_height = np.mean(data['heights'])
                
                # 平均高程
                avg_elevation = np.mean(data['elevations'])
                
                # 总面积
                total_area = np.sum(areas_array)
                
                # 生成新的fid（使用网格坐标）
                new_fid = f"grid_{grid_key[0]}_{grid_key[1]}"
                
                aggregated_coords.append(weighted_coords)
                aggregated_heights.append(avg_height)              # 存储平均原始高度
                aggregated_elevations.append(avg_elevation)        # 存储平均高程
                aggregated_areas.append(total_area)
                aggregated_fids.append(new_fid)
        
        # 添加超高建筑（不合并）
        for building in high_buildings:
            aggregated_coords.append(building['coord'])
            aggregated_heights.append(building['height'])          # 原始高度
            aggregated_elevations.append(building['elevation'])    # 高程
            aggregated_areas.append(building['area'])
            aggregated_fids.append(building['fid'])
        
        print(f"建筑集计完成: {len(building_coords)} → {len(aggregated_coords)}")
        print(f"  - 普通建筑: {len(aggregated_coords) - len(high_buildings)} 个集计点")
        print(f"  - 超高建筑: {len(high_buildings)} 个单独保留")
        
        # 返回时，分别存储高度和高程
        return [
            np.array(aggregated_coords),
            np.array(aggregated_heights),      # 原始建筑物高度
            np.array(aggregated_elevations),   # 高程
            np.array(aggregated_areas),
            np.array(aggregated_fids)
        ]

    def aggregate_stations_by_grid_with_selection(station_coords, station_heights, station_elevations, 
                                                station_fids, x_bins, y_bins, original_station_df, 
                                                strategy="lowest_cost"):
        """集计起降点数据，并保存网格内所有原始起降点的fid信息"""
        if len(station_coords) == 0:
            return [np.array([]), np.array([]), np.array([]), np.array([])], {}, {}
        
        # 为每个起降点分配网格索引
        x_indices = np.digitize(station_coords[:, 0], x_bins) - 1
        y_indices = np.digitize(station_coords[:, 1], y_bins) - 1
        
        # 创建网格字典和映射字典
        grid_dict = {}
        grid_to_selected_mapping = {}  # 网格键 -> 选中的原始站点索引
        grid_to_all_originals = {}     # 网格键 -> 网格内所有原始站点索引和fid
        
        for i in range(len(station_coords)):
            grid_key = (x_indices[i], y_indices[i])
            
            if grid_key not in grid_dict:
                grid_dict[grid_key] = {
                    'coords': [],
                    'heights': [],
                    'elevations': [],
                    'fids': [],
                    'original_indices': [],  # 保存原始索引
                    'all_original_data': []  # 保存网格内所有原始站点的完整信息
                }
            
            grid_dict[grid_key]['coords'].append(station_coords[i])
            grid_dict[grid_key]['heights'].append(station_heights[i])
            grid_dict[grid_key]['elevations'].append(station_elevations[i])
            grid_dict[grid_key]['fids'].append(station_fids[i])
            grid_dict[grid_key]['original_indices'].append(i)
            
            # 保存每个原始站点的完整信息
            grid_dict[grid_key]['all_original_data'].append({
                'original_index': i,
                'fid': station_fids[i],
                'coord': station_coords[i],
                'height': station_heights[i],
                'elevation': station_elevations[i]
            })
        
        # 处理每个网格，直接选择代表性起降点
        aggregated_coords = []
        aggregated_heights = []
        aggregated_elevations = []
        aggregated_fids = []
        
        # 新添加：保存网格内所有原始站点信息的字典
        grid_original_info = {}
        
        for grid_key, data in grid_dict.items():
            if len(data['coords']) > 0:
                # 使用成本最低策略直接选择代表性起降点
                selected_original_idx = select_representative_station_in_grid_direct(
                    data, strategy
                )
                
                # 使用选中的原始起降点数据
                selected_coord = station_coords[selected_original_idx]
                selected_height = station_heights[selected_original_idx]
                selected_elevation = station_elevations[selected_original_idx]
                selected_fid = station_fids[selected_original_idx]
                
                aggregated_coords.append(selected_coord)
                aggregated_heights.append(selected_height)
                aggregated_elevations.append(selected_elevation)
                aggregated_fids.append(selected_fid)
                
                # 记录映射关系
                grid_to_selected_mapping[grid_key] = selected_original_idx
                
                # 保存网格内所有原始站点的信息
                grid_original_info[grid_key] = {
                    'selected_original_index': selected_original_idx,
                    'selected_fid': selected_fid,
                    'all_originals': data['all_original_data'],  # 所有原始站点信息
                    'grid_coord': selected_coord  # 网格代表的坐标
                }
        
        print(f"起降点集计完成: {len(station_coords)} → {len(aggregated_coords)}")
        
        return [
            np.array(aggregated_coords),
            np.array(aggregated_heights),
            np.array(aggregated_elevations),
            np.array(aggregated_fids)
        ], grid_to_selected_mapping, grid_original_info

    def select_representative_station_in_grid_direct(grid_data, strategy="lowest_cost"):
        """直接在网格数据中选择代表性起降点（返回原始索引）"""
        if len(grid_data['original_indices']) == 0:
            return None
        
        if len(grid_data['original_indices']) == 1:
            return grid_data['original_indices'][0]
        
        # 使用成本最低策略
        if strategy == "lowest_cost":
            # 计算每个站点的成本
            costs = []
            for height in grid_data['heights']:
                cost = height * cost_multiplier + fixed_cost
                costs.append(cost)
            
            # 选择成本最低的站点
            min_cost_idx = np.argmin(costs)
            return grid_data['original_indices'][min_cost_idx]
        
        else:
            # 其他策略的实现（如果需要）
            # 这里我们只实现成本最低策略
            costs = []
            for height in grid_data['heights']:
                cost = height * cost_multiplier + fixed_cost
                costs.append(cost)
            
            min_cost_idx = np.argmin(costs)
            return grid_data['original_indices'][min_cost_idx]

    def aggregate_demands_by_grid(demand_coords, demand_elevations, x_bins, y_bins):
        """集计需求点数据"""
        if len(demand_coords) == 0:
            return [np.array([]), np.array([])]
        
        # 为每个需求点分配网格索引
        x_indices = np.digitize(demand_coords[:, 0], x_bins) - 1
        y_indices = np.digitize(demand_coords[:, 1], y_bins) - 1
        
        # 创建网格字典
        grid_dict = {}
        
        for i in range(len(demand_coords)):
            grid_key = (x_indices[i], y_indices[i])
            
            if grid_key not in grid_dict:
                grid_dict[grid_key] = {
                    'coords': [],
                    'elevations': []
                }
            
            grid_dict[grid_key]['coords'].append(demand_coords[i])
            grid_dict[grid_key]['elevations'].append(demand_elevations[i])
        
        # 处理每个网格
        aggregated_coords = []
        aggregated_elevations = []
        
        for grid_key, data in grid_dict.items():
            if len(data['coords']) > 0:
                # 计算重心（简单平均）
                avg_coord = np.mean(data['coords'], axis=0)
                avg_elevation = np.mean(data['elevations'])
                
                aggregated_coords.append(avg_coord)
                aggregated_elevations.append(avg_elevation)
        
        print(f"需求点集计完成: {len(demand_coords)} → {len(aggregated_coords)}")
        
        return [
            np.array(aggregated_coords),
            np.array(aggregated_elevations)
        ]

    def get_original_stations_from_mapping_enhanced(selected_grid_stations, grid_original_info, aggregated_station_fids):
        """
        增强版映射函数：使用网格内所有原始站点信息来正确映射
        """
        original_selected_stations = []
        original_selected_fids = []
        
        print("开始增强版站点映射...")
        
        for grid_station_idx in selected_grid_stations:
            # 获取集计站点的fid
            grid_fid = aggregated_station_fids[grid_station_idx]
            
            print(f"处理集计站点 {grid_station_idx}, FID: {grid_fid}")
            
            # 查找这个集计站点对应的网格信息
            found = False
            for grid_key, grid_info in grid_original_info.items():
                if grid_info['selected_fid'] == grid_fid:
                    # 找到对应的网格，获取选中的原始站点索引
                    original_idx = grid_info['selected_original_index']
                    original_selected_stations.append(original_idx)
                    original_selected_fids.append(grid_fid)
                    
                    # 打印调试信息
                    print(f"✅ 找到映射: 集计索引 {grid_station_idx} -> 原始索引 {original_idx}")
                    print(f"   网格坐标: {grid_key}")
                    print(f"   网格内原始站点数量: {len(grid_info['all_originals'])}")
                    
                    # 显示网格内所有站点的成本信息
                    all_costs = []
                    for original_data in grid_info['all_originals']:
                        cost = original_data['height'] * cost_multiplier + fixed_cost
                        all_costs.append((original_data['original_index'], cost))
                    
                    # 按成本排序
                    all_costs.sort(key=lambda x: x[1])
                    print(f"   网格内站点成本排序: {all_costs}")
                    print(f"   选中站点成本: {all_costs[0][1] if all_costs else 'N/A'}")
                    
                    found = True
                    break
            
            if not found:
                print(f"❌ 警告: 未找到集计站点 {grid_station_idx} (FID: {grid_fid}) 对应的网格信息")
                # 作为备选，使用集计索引本身
                original_selected_stations.append(grid_station_idx)
                original_selected_fids.append(grid_fid)
        
        print(f"映射完成: {len(selected_grid_stations)} 个网格点 → {len(original_selected_stations)} 个原始起降点")
        
        return original_selected_stations, original_selected_fids

    # ========================== 执行数据集计 ==========================

    # 保存原始数据用于可视化和成本计算
    original_station_coords = station_coords.copy()
    original_demand_coords = demand_coords.copy()
    original_building_coords = building_coords.copy()
    original_station_heights = station_heights.copy()
    original_station_elevations = station_elevations.copy()
    original_station_fids = station_fids.copy()
    original_demand_elevations = demand_elevations.copy()
    original_building_heights = building_heights.copy()
    original_building_elevations = building_elevations.copy()
    original_building_areas = building_areas.copy()
    original_building_fids = building_fids.copy()

    print("开始数据集计...")
    aggregated_data, selected_mapping, grid_original_info = aggregate_points_by_grid_simplified(grid_size=grid_size)

    # 更新全局变量为集计后的数据（用于计算）
    (building_coords, building_heights, building_elevations, building_areas, building_fids,
    station_coords, station_heights, station_elevations, station_fids,
    demand_coords, demand_elevations) = aggregated_data

    # ✅ 立即保存真正的集计数据引用
    aggregated_data_ref = {
        'station_coords': station_coords.copy(),        # 真正的集计数据
        'demand_coords': demand_coords.copy(),          # 真正的集计数据  
        'building_coords': building_coords.copy(),      # 真正的集计数据
        'building_heights': building_heights.copy(),    # 真正的集计数据
        'building_elevations': building_elevations.copy(), # 真正的集计数据
        'building_areas': building_areas.copy(),        # 真正的集计数据
        'station_heights': station_heights.copy(),      # 真正的集计数据
        'station_elevations': station_elevations.copy(), # 真正的集计数据
        'demand_elevations': demand_elevations.copy()   # 真正的集计数据
    }

    # 保存集计后的fid用于映射
    aggregated_station_fids = station_fids.copy()

    # ========================== 建立两套数据系统 ==========================
    print("建立两套数据系统...")

    # 1. 原始数据 (Original Data) - 用于可视化和成本计算
    original_data = {
        'station_coords': original_station_coords,
        'demand_coords': original_demand_coords, 
        'building_coords': original_building_coords,
        'station_heights': original_station_heights,
        'station_elevations': original_station_elevations,
        'station_fids': original_station_fids,
        'demand_elevations': original_demand_elevations,
        'building_heights': original_building_heights,
        'building_elevations': original_building_elevations,
        'building_areas': original_building_areas,
        'building_fids': original_building_fids
    }

    # 2. 集计数据 (Aggregated Data) - 用于算法计算
    aggregated_data = {
        'station_coords': station_coords,  # 集计后的站点坐标
        'demand_coords': demand_coords,    # 集计后的需求点坐标
        'building_coords': building_coords, # 集计后的建筑坐标
        'station_heights': station_heights,
        'station_elevations': station_elevations, 
        'station_fids': station_fids,
        'demand_elevations': demand_elevations,
        'building_heights': building_heights,
        'building_elevations': building_elevations,
        'building_areas': building_areas,
        'building_fids': building_fids
    }

    print("两套数据系统建立完成:")
    print(f"原始数据 - 站点: {len(original_data['station_coords'])}, 需求点: {len(original_data['demand_coords'])}, 建筑: {len(original_data['building_coords'])}")
    print(f"集计数据 - 站点: {len(aggregated_data['station_coords'])}, 需求点: {len(aggregated_data['demand_coords'])}, 建筑: {len(aggregated_data['building_coords'])}")

    print("数据集计完成！")
    print(f"集计后数据量:")
    print(f"  建筑: {len(building_coords)}")
    print(f"  起降点: {len(station_coords)}") 
    print(f"  需求点: {len(demand_coords)}")
    print(f"  网格信息数量: {len(grid_original_info)}")

    # 重新构建高建筑KDTree（使用集计后的数据）
    print("重新构建高建筑KDTree...")
    building_total_heights = building_heights - building_elevations
    high_building_mask = building_total_heights > b
    high_building_indices = np.where(high_building_mask)[0]

    if len(high_building_indices) > 0:
        high_building_coords = building_coords[high_building_indices]
        high_building_kdtree = KDTree(high_building_coords)
        print(f"高建筑KDTree构建完成，共{len(high_building_coords)}个高度超过{b}米的建筑")
    else:
        high_building_kdtree = None
        print("没有高度超过平飞高度的建筑")

    print(f"------数据读取和集计完毕------")

# 1.1 加载城市边界数据
def load_city_boundary(city_shapefile):
    city_boundary = gpd.read_file(city_shapefile)
    proj = Proj(
        proj='utm',
        zone=54,
        ellips='krass',
        towgs84='0,0,0,0,0,0,0',
        preserve_units=True
    )
    city_boundary = city_boundary.to_crs(proj.srs)
    return city_boundary

# 1.2 加载等高线数据
def load_elevation_data(elevation_file):
    elevation_data = gpd.read_file(elevation_file)
    proj = Proj(
        proj='utm',
        zone=54,
        ellips='krass',
        towgs84='0,0,0,0,0,0,0',
        preserve_units=True
    )
    elevation_data = elevation_data.to_crs(proj.srs)
    return elevation_data

# 1.3 生成高程热力图
def generate_elevation_heatmap(elevation_data):
    """根据等高线数据生成高程热力图数据"""
    points = []
    elevations = []
    for idx, row in elevation_data.iterrows():
        geom = row.geometry
        elev = row['ELEV']  # 假设高程字段名为'ELEV'
        if geom.geom_type == 'LineString':
            for coord in geom.coords:
                points.append(coord)
                elevations.append(elev)
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                for coord in line.coords:
                    points.append(coord)
                    elevations.append(elev)
    
    # 转换为numpy数组
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    z = np.array(elevations)
    
    # 生成网格
    xi = np.linspace(x.min(), x.max(), 2000)
    yi = np.linspace(y.min(), y.max(), 2000)
    xi, yi = np.meshgrid(xi, yi)
    
    # 使用线性插值
    zi = griddata((x, y), z, (xi, yi), method='linear')
    
    return xi, yi, zi

# ========================== 可达函数相关计算 ==========================

def calculate_power(omega):
    """计算功率（使用已经缩放好的功率公式）"""
    return 2.33e-3 * omega**3 + 1.528e-2 * omega**2 + 1.027e-2 * omega + 64.63

def calculate_omega_u(l_i):
    """计算起飞角速度（修正公式）"""
    numerator = ((2 * (m_d + m_p_max) * g + rho * A * Cd * v_u**2)**2 + (rho * A * Cd * v_w**2)**2)**(1/2)
    denominator = N * rho * Ap * Ct * r**2
    return math.sqrt(numerator / denominator)

def calculate_omega_d(l_i):
    """计算降落角速度（修正公式）"""
    numerator = ((2 * (m_d + m_p_max) * g - rho * A * Cd * v_d**2)**2 + (rho * A * Cd * v_w**2)**2)**(1/2)
    denominator = N * rho * Ap * Ct * r**2
    return math.sqrt(numerator / denominator)

def calculate_omega_h():
    """计算平飞角速度（修正公式）"""
    numerator = (4 * (m_d + m_p_max)**2 * g**2 + rho**2 * Ah**2 * Cd**2 * (v_h - v_w)**4)**(1/4)
    denominator = math.sqrt(N * rho * Ap * Ct * r**2)
    return numerator / denominator

def calculate_vertical_energy(l_i_start, l_i_end, task_type='cover'):
    """计算垂直起降能耗 - 区分起点和终点高度"""
    omega_u = calculate_omega_u(max(l_i_start, l_i_end))
    omega_d = calculate_omega_d(max(l_i_start, l_i_end))
    
    P_u = calculate_power(omega_u)
    P_d = calculate_power(omega_d)
    
    if task_type == 'cover':
        # 普通任务：起点起飞 + 终点降落 + 终点起飞 + 起点降落
        return (P_u * l_i_start / v_u + P_d * l_i_end / v_d + 
                P_u * l_i_end / v_u + P_d * l_i_start / v_d)
    else:  # relay
        # 接力任务：起点起飞 + 终点降落
        return P_u * l_i_start / v_u + P_d * l_i_end / v_d

def calculate_horizontal_energy(distance, task_type='cover'):
    """计算水平飞行能耗"""
    omega_h = calculate_omega_h()
    P_h = calculate_power(omega_h)
    
    if task_type == 'cover':
        # 普通任务：往返距离
        return P_h * 2 * distance / v_h
    else:  # relay
        # 接力任务：单程距离
        return P_h * distance / v_h

def is_point_on_segment(p, a, b):
    """判断点p是否在线段ab上"""
    # 使用向量叉积和点积判断
    cross = np.cross(b - a, p - a)
    if abs(cross) > 1e-10:  # 不在直线上
        return False
    
    # 在线段上
    dot1 = np.dot(p - a, b - a)
    dot2 = np.dot(p - b, a - b)
    return dot1 >= 0 and dot2 >= 0

def find_obstacles_on_path(start_coord, end_coord, building_coords, building_heights, building_elevations, building_areas):
    """优化版：使用KDTree精确查找路径上的障碍物"""
    obstacles = []
    path_vector = end_coord - start_coord
    path_length = np.linalg.norm(path_vector)
    
    if path_length == 0:
        return obstacles
    
    # 快速检查：如果路径很短，直接返回（没有足够空间形成障碍）
    if path_length < 500:  # 按照您的要求改为500米
        return obstacles
    
    # 使用全局高建筑KDTree
    if 'high_building_kdtree' not in globals() or high_building_kdtree is None:
        return obstacles
    
    # 路径方向单位向量
    path_dir = path_vector / path_length
    
    # 查询路径附近的潜在障碍物
    # 计算路径的边界框，扩大搜索范围确保覆盖所有可能障碍物
    path_center = (start_coord + end_coord) / 2
    search_radius = path_length / 2 + 50  # 路径长度一半+50米缓冲
    
    # 使用KDTree快速找到路径附近的高建筑
    nearby_indices = high_building_kdtree.query_ball_point(path_center, search_radius)
    
    # 对这些潜在障碍物进行精确几何计算
    for kd_index in nearby_indices:
        # 获取原始建筑索引
        original_idx = high_building_indices[kd_index]
        
        building_coord = building_coords[original_idx]
        # 计算总高（建筑物高度-高程）
        building_total_height = building_heights[original_idx] - building_elevations[original_idx]
        
        # 计算建筑到路径的垂直距离
        building_vector = building_coord - start_coord
        path_vector_3d = np.append(path_vector, 0)
        building_vector_3d = np.append(building_vector, 0)
        cross_product_3d = np.cross(path_vector_3d, building_vector_3d)
        distance_to_path = np.abs(cross_product_3d[2]) / path_length
        
        # 计算建筑在路径上的投影点
        t = np.dot(building_vector, path_dir)
        projection_point = start_coord + t * path_dir
        
        # 检查投影点是否在线段上
        if 0 <= t <= path_length and distance_to_path < 20:  # 20米范围内认为是路径上的障碍物
            obstacles.append({
                'coord': building_coord,
                'height': building_total_height,  # 总高（建筑物高度-高程）
                'area': building_areas[original_idx],
                'distance_to_path': distance_to_path,
                'projection_t': t
            })
    
    # 按路径上的位置排序
    obstacles.sort(key=lambda x: x['projection_t'])
    return obstacles

def calculate_detour_energy(obstacles, start_coord, end_coord, task_type='cover'):
    """计算绕行能耗 - 根据建筑物面积计算绕行距离"""
    if not obstacles:
        return 0
    
    base_distance = np.linalg.norm(end_coord - start_coord)
    
    # 计算绕行路径
    detour_path = [start_coord]
    current_pos = start_coord
    
    for obstacle in obstacles:
        # 根据建筑物面积计算边长（假设为正方形）
        side_length = math.sqrt(obstacle['area'])
        
        # 绕行策略：从障碍物旁边绕行，绕行距离为边长
        obstacle_coord = obstacle['coord']
        
        # 计算绕行方向（垂直于路径方向）
        path_vector = end_coord - start_coord
        path_dir = path_vector / np.linalg.norm(path_vector)
        perpendicular_dir = np.array([-path_dir[1], path_dir[0]])  # 垂直方向
        
        # 绕行点（在障碍物旁边）
        detour_point1 = obstacle_coord + perpendicular_dir * side_length
        detour_point2 = obstacle_coord - perpendicular_dir * side_length
        
        # 选择绕行距离较短的路径
        dist1 = (np.linalg.norm(detour_point1 - current_pos) + 
                np.linalg.norm(end_coord - detour_point1))
        dist2 = (np.linalg.norm(detour_point2 - current_pos) + 
                np.linalg.norm(end_coord - detour_point2))
        
        if dist1 < dist2:
            detour_path.append(detour_point1)
            current_pos = detour_point1
        else:
            detour_path.append(detour_point2)
            current_pos = detour_point2
    
    detour_path.append(end_coord)
    
    # 计算绕行总距离
    detour_distance = 0
    for i in range(len(detour_path) - 1):
        detour_distance += np.linalg.norm(detour_path[i+1] - detour_path[i])
    
    extra_distance = detour_distance - base_distance
    
    # 计算绕行额外能耗
    omega_h = calculate_omega_h()
    P_h = calculate_power(omega_h)
    energy_per_meter = P_h / v_h
    
    if task_type == 'cover':
        # 普通任务：往返都要绕行
        return 2 * extra_distance * energy_per_meter
    else:
        # 接力任务：单程绕行
        return extra_distance * energy_per_meter

def calculate_climb_energy(obstacles, start_coord, end_coord, task_type='cover'):
    """计算爬升能耗 - 对每个障碍物单独计算"""
    if not obstacles:
        return 0
    
    climb_energy = 0
    omega_u = calculate_omega_u(0)  # 使用0作为基础高度，实际高度在循环中计算
    omega_d = calculate_omega_d(0)
    P_u = calculate_power(omega_u)
    P_d = calculate_power(omega_d)
    
    for obstacle in obstacles:
        # 需要爬升到障碍物上方10米
        required_height = obstacle['height'] + 10
        
        # 计算额外爬升高度
        extra_climb_height = required_height - b
        
        if extra_climb_height > 0:
            # 计算单个障碍物的爬升能耗
            single_climb_energy = P_u * extra_climb_height / v_u + P_d * extra_climb_height / v_d
            
            if task_type == 'cover':
                # 普通任务：往返都要爬升
                climb_energy += 2 * single_climb_energy
            else:
                # 接力任务：单程爬升
                climb_energy += single_climb_energy
    
    return climb_energy

def calculate_obstacle_energy(obstacles, start_coord, end_coord, task_type='cover'):
    """计算避障能耗 - 选择绕行或爬升中能耗较小的策略"""
    if not obstacles:
        return 0, 0, 'none'
    
    detour_energy = calculate_detour_energy(obstacles, start_coord, end_coord, task_type)
    climb_energy = calculate_climb_energy(obstacles, start_coord, end_coord, task_type)
    
    # 选择能耗较小的策略
    if detour_energy < climb_energy:
        return detour_energy, len(obstacles), 'detour'
    else:
        return climb_energy, len(obstacles), 'climb'

def calculate_reachability(station_idx, target_coord, task_type,
                           building_coords, building_heights, building_elevations, building_areas,
                           station_coords, station_heights, station_elevations, 
                           demand_coords, demand_elevations,
                           show_progress=False, progress_desc=None):
    """
    可达函数计算 - 完整修复全局变量依赖问题
    """
    t0 = time.time()
    station_coord = station_coords[station_idx]
    station_height = station_heights[station_idx]
    station_elevation = station_elevations[station_idx]

    # 识别目标类型 - 现在使用传入的参数而不是全局变量
    demand_match = np.where((demand_coords == target_coord).all(axis=1))[0]
    if len(demand_match) > 0:
        d_idx = demand_match[0]
        target_height = 0
        # 安全地获取高程，检查索引是否在范围内
        if d_idx < len(demand_elevations):
            target_elevation = demand_elevations[d_idx]
        else:
            target_elevation = 0
        target_type = "demand"
    else:
        target_idx = None
        for i, coord in enumerate(station_coords):
            if np.array_equal(coord, target_coord):
                target_idx = i
                break
        if target_idx is not None:
            target_height = station_heights[target_idx]
            target_elevation = station_elevations[target_idx]
            target_type = "station"
        else:
            target_height = 0
            target_elevation = 0
            target_type = "unknown"

    # ========================= 计算阶段 =========================
    start_total_height = station_height + station_elevation
    end_total_height = target_height + target_elevation
    l_i_start = max(b - start_total_height, 0)
    l_i_end = max(b - end_total_height, 0)

    # 首先检测路径上的所有障碍物
    t1 = time.time()
    all_obstacles = find_obstacles_on_path(station_coord, target_coord,
                                          building_coords, building_heights,
                                          building_elevations, building_areas)
    total_obstacle_count = len(all_obstacles)
    t2 = time.time()

    # 障碍物数量阈值检查
    if total_obstacle_count > MAX_OBSTACLE_THRESHOLD:
        return False, float('inf'), total_obstacle_count, 'exceed_threshold'

    # 如果障碍物数量在阈值内，继续计算能耗
    # 只取前MAX_OBSTACLE_COUNT个障碍物用于能耗计算
    obstacles_for_calculation = all_obstacles[:MAX_OBSTACLE_COUNT]
    obstacle_count_for_calculation = len(obstacles_for_calculation)

    # 垂直能耗
    t3 = time.time()
    vertical_energy = calculate_vertical_energy(l_i_start, l_i_end, task_type)
    t4 = time.time()

    # 水平能耗
    distance = np.linalg.norm(target_coord - station_coord)
    horizontal_energy = calculate_horizontal_energy(distance, task_type)
    t5 = time.time()

    # 避障能耗（只计算前MAX_OBSTACLE_COUNT个障碍物）
    obstacle_energy, _, strategy = calculate_obstacle_energy(
        obstacles_for_calculation, station_coord, target_coord, task_type)
    t6 = time.time()

    total_energy = vertical_energy + horizontal_energy + obstacle_energy
    reachable = (total_energy <= E_total * electricity_consumption_rate) and (obstacle_count_for_calculation <= MAX_OBSTACLE_COUNT)

    return reachable, total_energy, total_obstacle_count, strategy

def calculate_max_service_radius(station_idx, task_type):
    """计算最大服务半径 - 使用论文中的公式"""
    station_height = station_heights[station_idx]
    station_elevation = station_elevations[station_idx]
    total_height = station_height - station_elevation
    l_i = max(b - total_height, 10)
    
    # 角速度计算
    ω_u = calculate_omega_u(l_i)
    ω_d = calculate_omega_d(l_i)
    ω_h = calculate_omega_h()
    
    # 功率计算
    P_u = calculate_power(ω_u)
    P_d = calculate_power(ω_d)
    P_h = calculate_power(ω_h)
    
    if task_type == 'cover':
        # 普通任务覆盖半径 r_i
        r_i = v_h * (E_total * electricity_consumption_rate - 
                     (P_u * l_i / v_u + P_u * (b - elevation_min) / v_u + 
                      P_d * (b - elevation_min) / v_d + P_d * l_i / v_d)) / (2 * P_h)
        return max(r_i, 0)
    else:  # relay
        # 接力任务覆盖半径 R_i
        R_i = v_h * (E_total * electricity_consumption_rate - 
                     (P_u * l_i / v_u + P_d * (b - elevation_min) / v_d)) / P_h
        return max(R_i, 0)

# 计算覆盖半径和接力半径（用于预筛选）
if __name__ == '__main__':
    print("计算最大服务半径用于预筛选...")
    cover_radii = np.array([calculate_max_service_radius(i, 'cover') for i in tqdm(range(len(station_coords)), desc="计算覆盖半径")])
    relay_radii = np.array([calculate_max_service_radius(i, 'relay') for i in tqdm(range(len(station_coords)), desc="计算接力半径")])

    print("======= 数据处理完成，准备开始算法计算 =======")

# ========================== 2. 算法主体 ==========================
CACHE_DIR = "cache"  # 缓存目录
#  缓存管理类 
class CacheManager:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, name, data_hash):
        return os.path.join(self.cache_dir, f"{name}_{data_hash}.pkl")
    
    def _compute_data_hash(self):
        """原有的数据哈希计算方法"""
        import hashlib
        import json
        
        # 所有影响可达性计算的参数
        reachability_params = {
            # 物理参数
            'g': g,
            'v_w': v_w,
            'b': b,
            'rho': rho,
            
            # 无人机参数
            'm_d': m_d,
            'm_p_max': m_p_max,
            'v_u': v_u,
            'v_d': v_d,
            'v_h': v_h,
            'A': A,
            'Ah': Ah,
            'r': r,
            'Ap': Ap,
            'N': N,
            'E_total': E_total,
            
            # 系数
            'eta': eta,
            'Cd': Cd,
            'Ct': Ct,
            'electricity_consumption_rate': electricity_consumption_rate,
            
            # 模型参数
            'cover_K': cover_K,
            'relay_K': relay_K,
            'MAX_OBSTACLE_COUNT': MAX_OBSTACLE_COUNT,
            'elevation_min': elevation_min,
        }
        
        # 数据统计信息
        data_stats = {
            'district': District,
            'station_count': len(station_coords),
            'demand_count': len(demand_coords),
            'building_count': len(building_coords),
            
            # 站点坐标统计
            'station_coords_mean': station_coords.mean().tolist() if len(station_coords) > 0 else [],
            'station_coords_std': station_coords.std().tolist() if len(station_coords) > 0 else [],
            
            # 需求点坐标统计
            'demand_coords_mean': demand_coords.mean().tolist() if len(demand_coords) > 0 else [],
            'demand_coords_std': demand_coords.std().tolist() if len(demand_coords) > 0 else [],
            
            # 建筑坐标统计
            'building_coords_mean': building_coords.mean().tolist() if len(building_coords) > 0 else [],
            'building_coords_std': building_coords.std().tolist() if len(building_coords) > 0 else [],
            
            # 高度统计
            'station_heights_mean': float(station_heights.mean()) if len(station_heights) > 0 else 0,
            'station_heights_std': float(station_heights.std()) if len(station_heights) > 0 else 0,
            'station_elevations_mean': float(station_elevations.mean()) if len(station_elevations) > 0 else 0,
            'station_elevations_std': float(station_elevations.std()) if len(station_elevations) > 0 else 0,
            'building_heights_mean': float(building_heights.mean()) if len(building_heights) > 0 else 0,
            'building_heights_std': float(building_heights.std()) if len(building_heights) > 0 else 0,
            'building_elevations_mean': float(building_elevations.mean()) if len(building_elevations) > 0 else 0,
            'building_elevations_std': float(building_elevations.std()) if len(building_elevations) > 0 else 0,
            
            # 面积统计
            'building_areas_mean': float(building_areas.mean()) if len(building_areas) > 0 else 0,
            'building_areas_std': float(building_areas.std()) if len(building_areas) > 0 else 0,
        }
        
        # 将参数和数据统计合并为一个字符串
        hash_data = {
            'params': reachability_params,
            'stats': data_stats
        }
        
        # 使用JSON序列化确保一致性
        hash_str = json.dumps(hash_data, sort_keys=True, separators=(',', ':'))
        return hashlib.md5(hash_str.encode()).hexdigest()[:12]
    
    def get_data_hash(self, *arrays):
        """
        计算数据哈希值 - 用于缓存标识
        """
        import hashlib
        import json
        
        hash_data = {}
        
        for i, arr in enumerate(arrays):
            if hasattr(arr, 'shape'):
                # numpy数组
                hash_data[f'array_{i}'] = {
                    'shape': arr.shape,
                    'dtype': str(arr.dtype),
                    'min': float(np.min(arr)) if arr.size > 0 else 0,
                    'max': float(np.max(arr)) if arr.size > 0 else 0,
                    'mean': float(np.mean(arr)) if arr.size > 0 else 0,
                    'sum': float(np.sum(arr)) if arr.size > 0 else 0
                }
            else:
                # 其他类型数据
                hash_data[f'array_{i}'] = {
                    'type': type(arr).__name__,
                    'len': len(arr) if hasattr(arr, '__len__') else 0,
                    'repr': str(arr)[:100]  # 只取前100个字符
                }
        
        # 添加影响计算的关键参数
        hash_data['params'] = {
            'MAX_OBSTACLE_COUNT': MAX_OBSTACLE_COUNT,
            'MAX_OBSTACLE_THRESHOLD': MAX_OBSTACLE_THRESHOLD,
            'electricity_consumption_rate': electricity_consumption_rate,
            'b': b,
            'E_total': E_total
        }
        
        # 使用JSON序列化确保一致性
        hash_str = json.dumps(hash_data, sort_keys=True, separators=(',', ':'))
        return hashlib.md5(hash_str.encode()).hexdigest()[:12]
    
    def load_reverse_index_cache(self, data_hash):
        """加载反向索引缓存"""
        cache_file = f"{self.cache_dir}/reverse_index_{data_hash}.pkl"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    if cache_data.get('data_hash') == data_hash:
                        print("✅ 加载反向索引缓存成功")
                        return cache_data['reverse_index'], cache_data['batch_files_info']
                    else:
                        print("⚠️ 缓存数据哈希不匹配，重新计算")
            except Exception as e:
                print(f"❌ 加载反向索引缓存失败: {e}")
        return None, None
    
    def save_reverse_index_cache(self, data_hash, reverse_index, batch_files_info):
        """保存反向索引缓存"""
        cache_file = f"{self.cache_dir}/reverse_index_{data_hash}.pkl"
        try:
            cache_data = {
                'data_hash': data_hash,
                'reverse_index': reverse_index,
                'batch_files_info': batch_files_info
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print("✅ 反向索引缓存保存成功")
        except Exception as e:
            print(f"❌ 保存反向索引缓存失败: {e}")
    
    def load_connectivity_matrix_cache(self, data_hash):
        """加载连通性矩阵缓存"""
        cache_file = f"{self.cache_dir}/connectivity_{data_hash}.pkl"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    if cache_data.get('data_hash') == data_hash:
                        print("✅ 加载连通性矩阵缓存成功")
                        return cache_data['connectivity_matrix']
                    else:
                        print("⚠️ 缓存数据哈希不匹配，重新计算")
            except Exception as e:
                print(f"❌ 加载连通性矩阵缓存失败: {e}")
        return None
    
    def save_connectivity_matrix_cache(self, data_hash, connectivity_matrix):
        """保存连通性矩阵缓存"""
        cache_file = f"{self.cache_dir}/connectivity_{data_hash}.pkl"
        try:
            cache_data = {
                'data_hash': data_hash,
                'connectivity_matrix': connectivity_matrix
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print("✅ 连通性矩阵缓存保存成功")
        except Exception as e:
            print(f"❌ 保存连通性矩阵缓存失败: {e}")
    
    def clear_old_caches(self, keep_recent=3):
        """清理旧的缓存，只保留最近的几个"""
        try:
            cache_files = []
            for fname in os.listdir(self.cache_dir):
                if fname.startswith(f"{self.district}_cov"):
                    cache_path = os.path.join(self.cache_dir, fname)
                    if os.path.isdir(cache_path):
                        mtime = os.path.getmtime(cache_path)
                        cache_files.append((mtime, cache_path))
            
            # 按修改时间排序，保留最新的
            cache_files.sort(reverse=True)
            for mtime, cache_path in cache_files[keep_recent:]:
                print(f"清理旧缓存: {os.path.basename(cache_path)}")
                shutil.rmtree(cache_path, ignore_errors=True)
                
        except Exception as e:
            print(f"清理缓存时出错: {e}")

cache = CacheManager(cache_dir="cache")
class RealtimeSelectionVisualizer:
    """
    实时动态选点过程可视化类 - 支持自动关闭和MP4保存
    """
    def __init__(self, station_coords, demand_coords, cover_radii, city_shapefile, save_animation=True):
        self.station_coords = station_coords
        self.demand_coords = demand_coords
        self.cover_radii = cover_radii
        self.city_shapefile = city_shapefile
        self.save_animation = save_animation
        
        # 初始化图形 - 只创建一个图形窗口
        plt.ion()  # 开启交互模式
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # 设置字体
        matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
        matplotlib.rcParams['axes.unicode_minus'] = False
        
        # 加载城市边界
        self.city_boundary = load_city_boundary(city_shapefile)
        
        # 记录选中的站点
        self.selected_stations = []
        self.current_iteration = 0
        
        # 动画保存设置
        if self.save_animation:
            self.animation_frames = []  # 用于保存动画的帧
            self.animation_filename = f"selection_process_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            print(f"将保存视频到: {self.animation_filename}")
        
        # 初始化可视化元素
        self.init_plot()
    
    def init_plot(self):
        """初始化绘图元素"""
        # 清空图形
        self.ax.clear()
        
        # 绘制城市边界
        self.city_boundary.plot(ax=self.ax, color='lightgray', edgecolor='black', alpha=0.5, linewidth=2)
        
        # 绘制所有候选站点（浅色）
        self.candidate_stations = self.ax.scatter(self.station_coords[:, 0], self.station_coords[:, 1], 
                       c='green', s=10, alpha=0.5, label='候选站点')
        
        # 绘制所有需求点（初始为红色，表示未覆盖）
        self.demand_points = self.ax.scatter(self.demand_coords[:, 0], self.demand_coords[:, 1], 
                                            c='blue', s=5, alpha=0.7, label='未覆盖需求点')
        
        # 初始化选中站点散点图（空）
        self.selected_points = self.ax.scatter([], [], c='blue', s=100, marker='*', label='选中站点')
        
        # 初始化覆盖范围圆
        self.cover_circles = []
        
        # 添加标题和标签
        self.ax.set_title('实时选点过程 - 迭代 0', fontsize=16)
        self.ax.set_xlabel('UTM X坐标', fontsize=12)
        self.ax.set_ylabel('UTM Y坐标', fontsize=12)
        
        # 添加图例
        self.ax.legend(loc='upper right')
        
        # 添加覆盖率文本
        self.coverage_text = self.ax.text(0.02, 0.98, '覆盖率: 0.00%\n选中站点: 0', 
                                         transform=self.ax.transAxes, fontsize=12,
                                         verticalalignment='top',
                                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)  # 短暂暂停以显示初始状态
        
        # 保存初始帧
        if self.save_animation:
            self.save_current_frame()
    
    def save_current_frame(self):
        """保存当前帧 - 使用更可靠的方法"""
        if self.save_animation:
            try:
                # 直接保存当前图形为临时文件，然后读取
                temp_filename = f"temp_frame_{len(self.animation_frames)}.png"
                self.fig.savefig(temp_filename, dpi=100, bbox_inches='tight', facecolor='white')
                
                # 读取临时文件
                from PIL import Image
                img = Image.open(temp_filename)
                # 转换为RGB
                img = img.convert('RGB')
                # 转换为numpy数组
                img_array = np.array(img)
                
                self.animation_frames.append(img_array)
                
                # 删除临时文件
                import os
                os.remove(temp_filename)
                
            except Exception as e:
                print(f"保存帧时出错: {e}")
    
    def update_plot(self, iteration, selected_stations, coverage_rate):
        """更新绘图 - 在同一图形上更新"""
        self.current_iteration = iteration
        self.selected_stations = selected_stations.copy()
        
        # 清空覆盖圆
        for circle in self.cover_circles:
            circle.remove()
        self.cover_circles = []
        
        # 计算当前覆盖的需求点
        covered_demand = self.calculate_covered_demand(selected_stations)
        
        # 更新需求点颜色 - 在同一图形上更新
        demand_colors = ['green' if covered else 'red' for covered in covered_demand]
        self.demand_points.set_color(demand_colors)
        
        # 更新选中站点 - 在同一图形上更新
        if len(selected_stations) > 0:
            self.selected_points.set_offsets(self.station_coords[selected_stations])
            
            # 添加覆盖范围圆 - 在同一图形上更新
            for station_idx in selected_stations:
                circle = plt.Circle(self.station_coords[station_idx], self.cover_radii[station_idx], 
                                   color='blue', fill=False, linestyle='--', linewidth=1, alpha=0.5)
                self.ax.add_patch(circle)
                self.cover_circles.append(circle)
        
        # 更新标题和覆盖率文本 - 在同一图形上更新
        self.ax.set_title(f'实时选点过程 - 迭代 {iteration}', fontsize=16)
        self.coverage_text.set_text(f'覆盖率: {coverage_rate:.2%}\n选中站点: {len(selected_stations)}')
        
        # 刷新图形 - 在同一图形上更新
        plt.draw()
        plt.pause(0.5)  # 暂停0.5秒以便观察
        
        # 保存当前帧
        if self.save_animation:
            self.save_current_frame()
    
    def calculate_covered_demand(self, selected_stations):
        """计算哪些需求点被覆盖"""
        if not selected_stations:
            return np.zeros(len(self.demand_coords), dtype=bool)
        
        covered = np.zeros(len(self.demand_coords), dtype=bool)
        station_tree = KDTree(self.station_coords[selected_stations])
        cover_radii_subset = self.cover_radii[selected_stations]
        max_radius = np.max(cover_radii_subset)
        
        for i, demand_point in enumerate(self.demand_coords):
            indices = station_tree.query_ball_point(demand_point, max_radius)
            for j in indices:
                if np.linalg.norm(demand_point - self.station_coords[selected_stations[j]]) <= cover_radii_subset[j]:
                    covered[i] = True
                    break
        
        return covered
    
    def highlight_new_station(self, new_station_idx, coverage_count):
        """高亮显示新选中的站点 - 在同一图形上更新"""
        # 临时高亮新选中的站点
        temp_highlight = self.ax.scatter(
            self.station_coords[new_station_idx, 0], 
            self.station_coords[new_station_idx, 1], 
            c='red', s=150, marker='*', alpha=0.8)
        
        # 添加临时文本说明
        highlight_text = self.ax.text(
            self.station_coords[new_station_idx, 0] + 100, 
            self.station_coords[new_station_idx, 1] + 100,
            f'新站点\n覆盖{coverage_count}个需求点',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # 刷新图形
        plt.draw()
        plt.pause(1.0)  # 暂停1秒以便观察
        
        # 保存高亮帧
        if self.save_animation:
            self.save_current_frame()
        
        # 移除临时高亮
        temp_highlight.remove()
        highlight_text.remove()
        
        # 刷新图形
        plt.draw()
        plt.pause(0.1)
        
        # 保存恢复后的帧
        if self.save_animation:
            self.save_current_frame()
    
    def save_animation_file(self):
        """保存动画文件为MP4"""
        if not self.save_animation or not self.animation_frames:
            return
            
        print(f"正在保存视频到: {self.animation_filename}")
        
        try:
            # 使用OpenCV保存MP4
            import cv2
            
            # 获取第一帧的尺寸
            height, width, layers = self.animation_frames[0].shape
            
            # 创建视频编写器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(self.animation_filename, fourcc, 2, (width, height))
            
            # 写入每一帧
            for frame in self.animation_frames:
                # 将RGB转换为BGR（OpenCV使用BGR格式）
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(bgr_frame)
            
            # 释放视频编写器
            video.release()
            print(f"MP4视频保存成功: {self.animation_filename}")
            print(f"共保存了 {len(self.animation_frames)} 帧")
                
        except ImportError:
            print("OpenCV 库未安装，无法保存MP4视频")
            print("请安装: pip install opencv-python")
            # 如果OpenCV不可用，尝试使用matplotlib的动画保存功能
            self.save_animation_with_matplotlib()
        except Exception as e:
            print(f"使用OpenCV保存MP4失败: {e}")
            # 如果OpenCV失败，尝试使用matplotlib的动画保存功能
            self.save_animation_with_matplotlib()
    
    def save_animation_with_matplotlib(self):
        """使用matplotlib保存动画（备选方法）"""
        try:
            # 使用matplotlib的动画功能保存MP4
            def animate(frame):
                # 这里我们实际上不需要做任何事情，因为我们已经有了所有帧
                pass
            
            # 创建动画对象
            anim = animation.FuncAnimation(self.fig, animate, frames=len(self.animation_frames), 
                                          interval=500, repeat=False)
            
            # 尝试使用FFMpegWriter
            try:
                writer = animation.FFMpegWriter(fps=2, metadata=dict(artist='Drone Station Selection'), bitrate=1800)
                anim.save(self.animation_filename, writer=writer)
                print(f"使用FFMpeg保存MP4成功: {self.animation_filename}")
            except:
                # 如果FFMpeg不可用，尝试使用PillowWriter保存GIF
                print("FFMpeg不可用，尝试保存为GIF")
                gif_filename = self.animation_filename.replace('.mp4', '.gif')
                writer = animation.PillowWriter(fps=2)
                anim.save(gif_filename, writer=writer)
                print(f"GIF视频保存成功: {gif_filename}")
                self.animation_filename = gif_filename
                
        except Exception as e:
            print(f"使用matplotlib保存动画也失败: {e}")
    
    def close(self):
        """关闭可视化并保存动画"""
        if self.save_animation:
            self.save_animation_file()
        
        plt.ioff()  # 关闭交互模式
        plt.close(self.fig)
        print("实时可视化窗口已关闭")

def build_reverse_index(cache_manager, data_hash, station_coords, demand_coords,
                       building_coords, building_heights, building_elevations, building_areas,
                       station_heights, station_elevations, demand_elevations,
                       batch_size=5000):
    """
    构建反向索引 - 修复版本
    """
    print("构建反向索引（使用多进程并行计算）...")
    
    # 尝试加载缓存
    cached_reverse_index, _ = cache_manager.load_reverse_index_cache(data_hash)
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
            
            for result in pool.imap_unordered(compute_demand_coverage, tasks):
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
    cache_manager.save_reverse_index_cache(data_hash, all_reverse_index, batch_files_info)
    
    print(f"✅ 反向索引构建完成，共 {len(all_reverse_index)} 个需求点的覆盖信息")
    
    return all_reverse_index

def compute_connectivity_matrix(
    cache_manager,
    data_hash,
    station_coords, station_heights, station_elevations,
    building_coords, building_heights, building_elevations, building_areas,
    relay_radii, demand_coords, demand_elevations,
    batch_size=2000,
):
    """
    分批 + 多进程 加速可达性连通矩阵计算（完整修复全局变量问题）
    修改进度条显示风格，与反向索引保持一致
    """
    # 1. 尝试读取缓存
    cached = cache_manager.load_connectivity_matrix_cache(data_hash)
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

    # 外层进度条：分批处理 - 修改为与反向索引相同的风格
    batch_range = range(0, n, batch_size)
    batch_pbar = tqdm(batch_range, desc="构建连通矩阵", unit="批", ncols=100)

    # --- 分批 ---
    for start in batch_pbar:
        end = min(start + batch_size, n)

        tasks = []

        # 内层进度条：当前批次内任务构建 - 修改为与反向索引相同的风格
        task_pbar = tqdm(range(start, end), desc=f"批次 {start//batch_size + 1}", 
                        leave=False, unit="站", ncols=80)

        # 构造任务列表（包含所有需要的参数）
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
            # 处理任务进度条 - 修改为与反向索引相同的风格
            process_pbar = tqdm(total=len(tasks), desc=f"处理批次 {start//batch_size + 1}", 
                               leave=False, unit="任务", ncols=80)

            # 使用imap_unordered获取结果
            results = []
            for result in pool.imap_unordered(connectivity_worker, tasks):
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
    cache_manager.save_connectivity_matrix_cache(data_hash, connectivity)

    print("连通矩阵构建完成")
    return connectivity

def integrated_efficiency_selection_with_aggregated_data(aggregated_data, grid_original_info,
                                                       cover_radii, relay_radii, city_shapefile,
                                                       batch_size=5000, enable_realtime_visualization=True, 
                                                       use_cache=True, external_cache_manager=None):
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
    
    # 初始化缓存管理器
    cache_manager = CacheManager()
    
    # 初始化实时可视化 - 注意：可视化使用集计数据，但这里我们先用集计数据展示进度
    visualizer = None
    if enable_realtime_visualization:
        # 使用集计数据创建可视化（近似显示）
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
    
    # 用于存储临时文件信息（如果需要）
    temp_files_created = False
    
    # 定义reverse_index_data和connectivity_matrix变量
    reverse_index_data = None
    connectivity_matrix = None
    
    try:
        # 计算数据哈希 - 使用集计数据
        data_hash = cache_manager.get_data_hash(
            station_coords, demand_coords, building_coords,
            building_heights, building_elevations, building_areas
        )
        
        # 步骤1: 构建反向索引 - 使用集计数据
        reverse_index_data = build_reverse_index(
            cache_manager, data_hash, station_coords, demand_coords,
            building_coords, building_heights, building_elevations, building_areas,
            station_heights, station_elevations, demand_elevations,
            batch_size
        )
        
        # 检查反向索引数据
        if not reverse_index_data:
            print("❌ 错误：反向索引数据为空，无法继续")
            return [], iteration_data, None, None
        
        # 步骤2: 计算连通性矩阵 - 使用集计数据
        connectivity_matrix = compute_connectivity_matrix(
            cache_manager=cache,
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
            
            # 更新实时可视化 - 使用集计数据
            if visualizer:
                visualizer.update_plot(iteration, selected_stations, current_coverage)
            
            # 计算动态权重
            progress = min(current_coverage / target_coverage_percentage, 1.0)
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
            if current_coverage >= target_coverage_percentage:
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
                cost = station_heights[station_idx] * cost_multiplier + fixed_cost
                
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
                cost = station_heights[station_idx] * cost_multiplier + fixed_cost
                
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
            cost = station_heights[best_station] * cost_multiplier + fixed_cost
            
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

def mst_connectivity_repair_with_aggregated_data(selected_stations, aggregated_data, grid_original_info):
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
        cost = station_heights[station_idx] * cost_multiplier + fixed_cost
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
                    reachable1, _, _, _ = calculate_reachability(
                        node_i, station_coords[node_j], 'relay',
                        building_coords, building_heights, building_elevations, building_areas,
                        station_coords, station_heights, station_elevations, 
                        demand_coords, demand_elevations
                    )
                    reachability_cache[cache_key1] = reachable1
                
                if cache_key2 in reachability_cache:
                    reachable2 = reachability_cache[cache_key2]
                else:
                    reachable2, _, _, _ = calculate_reachability(
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
                reachable1, _, _, _ = calculate_reachability(
                    station, station_coords[candidate], 'relay',
                    building_coords, building_heights, building_elevations, building_areas,
                    station_coords, station_heights, station_elevations, 
                    demand_coords, demand_elevations
                )
                reachable2, _, _, _ = calculate_reachability(
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
                reachable1, _, _, _ = calculate_reachability(
                    candidate, station_coords[isolated_node], 'relay',
                    building_coords, building_heights, building_elevations, building_areas,
                    station_coords, station_heights, station_elevations, 
                    demand_coords, demand_elevations
                )
                reachable2, _, _, _ = calculate_reachability(
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
                    reachable, _, _, _ = calculate_reachability(
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

def redundancy_pruning_with_aggregated_data(selected_stations, aggregated_data, grid_original_info,
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
                is_reachable, _, _, _ = calculate_reachability(
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
                    reachable1, _, _, _ = calculate_reachability(
                        s1, station_coords[s2], 'relay',
                        building_coords, building_heights, building_elevations, building_areas,
                        station_coords, station_heights, station_elevations,
                        demand_coords, demand_elevations                                      
                    )
                    reachability_cache[cache_key1] = reachable1
                
                if cache_key2 in reachability_cache:
                    reachable2 = reachability_cache[cache_key2]
                else:
                    reachable2, _, _, _ = calculate_reachability(
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
                is_reachable, _, _, _ = calculate_reachability(
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
        cost = station_heights[station_idx] * cost_multiplier + fixed_cost
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
                    is_reachable, _, _, _ = calculate_reachability(
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
        if temp_coverage >= target_coverage_percentage - coverage_tolerance and is_connected:
            # 可以安全移除
            pruned_stations.remove(station_idx)
            removed_stations.append(station_idx)
            
            # 更新进度条描述
            progress_bar.set_description(f"剪枝进度 (移除{len(removed_stations)}个)")
            
            print(f"✅ 移除站点 {station_idx} (成本: {cost:,.0f}元)")
            print(f"   移除后覆盖率: {temp_coverage:.4%} (目标覆盖率: {target_coverage_percentage:.4%})")
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
                is_reachable, _, _, _ = calculate_reachability(
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
    print(f"最终覆盖率: {final_coverage:.4%} (目标: {target_coverage_percentage:.4%})")
    print(f"网络连通性: {'保持连通' if is_final_connected else '断开'}")
    print(f"冗余剪枝耗时: {prune_time:.2f}秒")
    
    # 验证覆盖率没有显著降低
    coverage_loss = initial_coverage - final_coverage
    if final_coverage >= target_coverage_percentage - coverage_tolerance and is_final_connected:
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

def improved_normalization(station_idx, station_coverage, connectivity_count, 
                         uncovered_count, total_candidates, cost, total_demand_points):
    """改进的归一化方法"""
    
    # 计算原始效率
    raw_coverage_efficiency = station_coverage[station_idx] / cost
    raw_connectivity_efficiency = connectivity_count / cost
    
    # 新的归一化方法
    # 使用未覆盖需求点比例作为基准，而不是绝对值
    uncovered_ratio = uncovered_count / total_demand_points
    normalized_coverage = raw_coverage_efficiency / (uncovered_ratio + 0.01)  # +0.01避免除零
    
    # 连通效率使用候选点总数归一化
    normalized_connectivity = raw_connectivity_efficiency / total_candidates
    
    # 放大效率值
    normalized_coverage *= 10000
    normalized_connectivity *= 10000
    
    return normalized_coverage, normalized_connectivity

def calculate_coverage(stations, station_coords, demand_coords, cover_radii):
    """快速计算覆盖率（优化版）"""
    if not stations:
        return 0.0
    
    station_tree = KDTree(station_coords[stations])
    cover_radii_subset = cover_radii[stations]
    max_radius = np.max(cover_radii_subset)
    
    covered_count = 0
    for demand_point in demand_coords:
        # 使用球查询加速
        indices = station_tree.query_ball_point(demand_point, max_radius)
        is_covered = False
        for idx in indices:
            if np.linalg.norm(demand_point - station_coords[stations[idx]]) <= cover_radii_subset[idx]:
                is_covered = True
                break
        if is_covered:
            covered_count += 1
    
    return covered_count / len(demand_coords)

# ========================== 可视化函数（使用原始数据） ==========================

def add_compass(ax, labelsize=18, loc_x=0.88, loc_y=0.85, width=0.04, height=0.13, pad=0.14):
    """添加指北针"""
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    ylen = maxy - miny
    xlen = maxx - minx
    left = [minx + xlen*(loc_x - width*.5), miny + ylen*(loc_y - pad)]
    right = [minx + xlen*(loc_x + width*.5), miny + ylen*(loc_y - pad)]
    top = [minx + xlen*loc_x, miny + ylen*(loc_y - pad + height)]
    center = [minx + xlen*loc_x, left[1] + (top[1] - left[1])*.4]
    triangle = mpatches.Polygon([left, top, right, center], color='k')
    ax.text(s='N', x=minx + xlen*loc_x, y=miny + ylen*(loc_y - pad + height),
            fontsize=labelsize, horizontalalignment='center', verticalalignment='bottom')
    ax.add_patch(triangle)

def calculate_cover_boundary(station_idx, reverse_index, demand_coords):
    """快速计算普通任务覆盖边界 - 使用反向索引"""
    if station_idx not in reverse_index:
        return None
    
    # 获取该站点覆盖的所有需求点索引
    covered_demand_indices = []
    for demand_idx, stations in reverse_index.items():
        if station_idx in stations:
            covered_demand_indices.append(demand_idx)
    
    if not covered_demand_indices:
        return None
    
    # 使用凸包算法计算边界
    from scipy.spatial import ConvexHull
    points = demand_coords[covered_demand_indices]
    
    if len(points) < 3:
        return points
    
    try:
        hull = ConvexHull(points)
        boundary_points = points[hull.vertices]
        return np.vstack([boundary_points, boundary_points[0]])
    except:
        return points

def calculate_relay_boundary(station_idx, connectivity_matrix, station_coords):
    """快速计算接力任务覆盖边界 - 使用连通矩阵"""
    # 获取与该站点连通的所有其他站点
    connected_indices = np.where(connectivity_matrix[station_idx, :])[0]
    
    # 移除自己
    connected_indices = connected_indices[connected_indices != station_idx]
    
    if len(connected_indices) == 0:
        return None
    
    # 使用凸包算法计算边界
    from scipy.spatial import ConvexHull
    points = station_coords[connected_indices]
    
    if len(points) < 3:
        return points
    
    try:
        hull = ConvexHull(points)
        boundary_points = points[hull.vertices]
        return np.vstack([boundary_points, boundary_points[0]])
    except:
        return points

def plot_irregular_coverage(ax, station_coord, boundary_points, color, linestyle, label):
    """绘制不规则覆盖区域"""
    if boundary_points is not None and len(boundary_points) > 2:
        polygon = mpatches.Polygon(boundary_points, fill=False, color=color, 
                                linestyle=linestyle, linewidth=1.5, label=label)
        ax.add_patch(polygon)

def plot_solution_with_original_data(original_data, selected_original_stations, 
                                   reverse_index_data, connectivity_matrix, city_shapefile,
                                   aggregated_data, selected_stations):
    """
    图1：完整解决方案可视化 - 包含所有元素
    """
    print("生成完整解决方案可视化（图1：全元素图）...")
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    
    # 设置字体
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 从原始数据中提取
    station_coords = original_data['station_coords']
    demand_coords = original_data['demand_coords']
    station_heights = original_data['station_heights']
    station_fids = original_data['station_fids']
    
    # 加载城市边界
    city_boundary = load_city_boundary(city_shapefile)
    city_boundary.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.5, linewidth=2)
    
    # 加载等高线数据
    elevation_data = load_elevation_data(elevation_file)
    # 生成热力图背景
    xi, yi, zi = generate_elevation_heatmap(elevation_data)
    heatmap = ax.pcolormesh(xi, yi, zi, shading='auto', cmap='terrain', alpha=0.6)
    cbar = plt.colorbar(heatmap, ax=ax, label='高程')
    cbar.set_label('高程', fontsize=18)
    cbar.ax.set_yticks([])  # 删除刻度线

    # 绘制需求点（蓝色）
    demand_scatter = ax.scatter(demand_coords[:, 0], demand_coords[:, 1], 
                            c='blue', label='需求点', s=10, marker='o', alpha=0.7)

    # 绘制所有候选站点（绿色）
    candidate_scatter = ax.scatter(station_coords[:, 0], station_coords[:, 1], 
                                c='green', s=8, alpha=0.6, label='候选起降点')
    
    # 高亮显示被选中的起降点（红色五角星）
    selected_scatter = ax.scatter(station_coords[selected_original_stations, 0], 
                                station_coords[selected_original_stations, 1], 
                                c='red', label='选中起降点', s=150, marker='*', linewidth=1.5)
    
    # 绘制真实的服务范围边界（使用集计数据计算）
    print("计算真实的服务范围边界...")
    cover_boundaries = []
    relay_boundaries = []
    
    # 为每个选中的集计站点计算边界
    for station_idx in selected_stations:
        # 计算普通任务覆盖边界
        cover_boundary = calculate_cover_boundary(station_idx, reverse_index_data, aggregated_data['demand_coords'])
        if cover_boundary is not None:
            cover_boundaries.append(cover_boundary)
        
        # 计算接力任务覆盖边界  
        relay_boundary = calculate_relay_boundary(station_idx, connectivity_matrix, aggregated_data['station_coords'])
        if relay_boundary is not None:
            relay_boundaries.append(relay_boundary)
    
    # 绘制普通任务覆盖边界（红色虚线）
    for boundary in cover_boundaries:
        if len(boundary) >= 3:
            polygon = mpatches.Polygon(boundary, fill=False, color='red', 
                                    linestyle='--', linewidth=1.5, alpha=0.7)
            ax.add_patch(polygon)
    
    # 绘制接力任务覆盖边界（紫色点线）
    for boundary in relay_boundaries:
        if len(boundary) >= 3:
            polygon = mpatches.Polygon(boundary, fill=False, color='purple', 
                                    linestyle=':', linewidth=1.5, alpha=0.7)
            ax.add_patch(polygon)
    
    # 组合图例
    legend_elements = [
        demand_scatter,
        candidate_scatter,
        selected_scatter,
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='普通任务覆盖范围'),
        Line2D([0], [0], color='purple', linestyle=':', linewidth=1.5, label='接力任务覆盖范围')
    ]
    
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(1, 1),
        title='图例说明',
        title_fontsize=12,
        frameon=True,
        framealpha=0.9,
        fontsize=10
    )
    
    plt.title(f"无人机起降点选址完整方案 - 选中{len(selected_original_stations)}个起降点", fontsize=16)
    plt.xlabel('UTM X坐标', fontsize=12)
    plt.ylabel('UTM Y坐标', fontsize=12)
    plt.axis('equal')
    ax.grid(False)
    add_compass(ax, labelsize=12)
    plt.tight_layout()
    plt.show()

def plot_selected_with_both_coverage_with_original_data(original_data, selected_original_stations, 
                                                      reverse_index_data, connectivity_matrix, city_shapefile,
                                                      aggregated_data, selected_stations):
    """
    图2：去掉候选起降点的双任务覆盖图
    """
    print("生成双任务覆盖范围可视化（图2：无候选起降点）...")
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    
    # 设置字体
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 从原始数据中提取
    station_coords = original_data['station_coords']
    demand_coords = original_data['demand_coords']
    
    # 加载城市边界
    city_boundary = load_city_boundary(city_shapefile)
    city_boundary.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.5, linewidth=2)
    
    # 加载等高线数据
    elevation_data = load_elevation_data(elevation_file)
    # 生成热力图背景
    xi, yi, zi = generate_elevation_heatmap(elevation_data)
    heatmap = ax.pcolormesh(xi, yi, zi, shading='auto', cmap='terrain', alpha=0.6)
    cbar = plt.colorbar(heatmap, ax=ax, label='高程')
    cbar.set_label('高程', fontsize=18)
    cbar.ax.set_yticks([])  # 删除刻度线
    
    # 绘制需求点（蓝色）
    demand_scatter = ax.scatter(demand_coords[:, 0], demand_coords[:, 1], 
                            c='blue', label='需求点', s=10, marker='o', alpha=0.7)

    # 高亮显示被选中的起降点（红色五角星）
    selected_scatter = ax.scatter(station_coords[selected_original_stations, 0], 
                                station_coords[selected_original_stations, 1], 
                                c='red', label='选中起降点', s=150, marker='*', linewidth=1.5)

    # 绘制真实的服务范围边界
    cover_boundaries = []
    relay_boundaries = []
    
    for station_idx in selected_stations:
        cover_boundary = calculate_cover_boundary(station_idx, reverse_index_data, aggregated_data['demand_coords'])
        if cover_boundary is not None:
            cover_boundaries.append(cover_boundary)
        
        relay_boundary = calculate_relay_boundary(station_idx, connectivity_matrix, aggregated_data['station_coords'])
        if relay_boundary is not None:
            relay_boundaries.append(relay_boundary)
    
    # 绘制普通任务覆盖边界（红色虚线）
    for boundary in cover_boundaries:
        if len(boundary) >= 3:
            polygon = mpatches.Polygon(boundary, fill=False, color='red', 
                                    linestyle='--', linewidth=1.5, alpha=0.7)
            ax.add_patch(polygon)
    
    # 绘制接力任务覆盖边界（紫色点线）
    for boundary in relay_boundaries:
        if len(boundary) >= 3:
            polygon = mpatches.Polygon(boundary, fill=False, color='purple', 
                                    linestyle=':', linewidth=1.5, alpha=0.7)
            ax.add_patch(polygon)

    # 组合图例
    legend_elements = [
        demand_scatter,
        selected_scatter,
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='普通任务覆盖范围'),
        Line2D([0], [0], color='purple', linestyle=':', linewidth=1.5, label='接力任务覆盖范围')
    ]
    
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(1, 1),
        title='图例说明',
        title_fontsize=12,
        frameon=True,
        framealpha=0.9,
        fontsize=10
    )
    
    add_compass(ax, labelsize=12)

    plt.title(f"选中起降点与双任务覆盖范围 - {len(selected_original_stations)}个站点", fontsize=16)
    plt.xlabel('UTM X坐标', fontsize=12)
    plt.ylabel('UTM Y坐标', fontsize=12)
    plt.axis('equal')
    ax.grid(False)
    plt.tight_layout()
    plt.show()

def plot_selected_with_normal_coverage_with_original_data(original_data, selected_original_stations, 
                                                        reverse_index_data, city_shapefile,
                                                        aggregated_data, selected_stations):
    """
    图3：只有普通任务覆盖范围
    """
    print("生成普通任务覆盖范围可视化（图3：仅普通任务）...")
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    
    # 设置字体
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 从原始数据中提取
    station_coords = original_data['station_coords']
    demand_coords = original_data['demand_coords']
    
    # 加载城市边界
    city_boundary = load_city_boundary(city_shapefile)
    city_boundary.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.5, linewidth=2)
    
    # 加载等高线数据
    elevation_data = load_elevation_data(elevation_file)
    # 生成热力图背景
    xi, yi, zi = generate_elevation_heatmap(elevation_data)
    heatmap = ax.pcolormesh(xi, yi, zi, shading='auto', cmap='terrain', alpha=0.6)
    cbar = plt.colorbar(heatmap, ax=ax, label='高程')
    cbar.set_label('高程', fontsize=18)
    cbar.ax.set_yticks([])  # 删除刻度线
    
    # 绘制需求点（蓝色）
    demand_scatter = ax.scatter(demand_coords[:, 0], demand_coords[:, 1], 
                            c='blue', label='需求点', s=10, marker='o', alpha=0.7)

    # 高亮显示被选中的起降点（红色五角星）
    selected_scatter = ax.scatter(station_coords[selected_original_stations, 0], 
                                station_coords[selected_original_stations, 1], 
                                c='red', label='选中起降点', s=150, marker='*', linewidth=1.5)

    # 绘制普通任务覆盖边界
    cover_boundaries = []
    
    for station_idx in selected_stations:
        cover_boundary = calculate_cover_boundary(station_idx, reverse_index_data, aggregated_data['demand_coords'])
        if cover_boundary is not None:
            cover_boundaries.append(cover_boundary)
    
    # 绘制普通任务覆盖边界（红色虚线）
    for boundary in cover_boundaries:
        if len(boundary) >= 3:
            polygon = mpatches.Polygon(boundary, fill=False, color='red', 
                                    linestyle='--', linewidth=1.5, alpha=0.7)
            ax.add_patch(polygon)

    # 组合图例
    legend_elements = [
        demand_scatter,
        selected_scatter,
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='普通任务覆盖范围')
    ]
    
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(1, 1),
        title='图例说明',
        title_fontsize=12,
        frameon=True,
        framealpha=0.9,
        fontsize=10
    )
    
    add_compass(ax, labelsize=12)
    
    plt.title(f"普通任务覆盖范围 - {len(selected_original_stations)}个起降点", fontsize=16)
    plt.xlabel('UTM X坐标', fontsize=12)
    plt.ylabel('UTM Y坐标', fontsize=12)
    plt.axis('equal')
    ax.grid(False)
    plt.tight_layout()
    plt.show()

def plot_selected_with_relay_coverage_with_original_data(original_data, selected_original_stations, 
                                                       connectivity_matrix, city_shapefile,
                                                       aggregated_data, selected_stations):
    """
    图4：只有选中起降点和接力任务覆盖范围
    """
    print("生成接力任务覆盖范围可视化（图4：仅接力任务）...")
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    
    # 设置字体
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 从原始数据中提取
    station_coords = original_data['station_coords']
    
    # 加载城市边界
    city_boundary = load_city_boundary(city_shapefile)
    city_boundary.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.5, linewidth=2)
    
    # 加载等高线数据
    elevation_data = load_elevation_data(elevation_file)
    # 生成热力图背景
    xi, yi, zi = generate_elevation_heatmap(elevation_data)
    heatmap = ax.pcolormesh(xi, yi, zi, shading='auto', cmap='terrain', alpha=0.6)
    cbar = plt.colorbar(heatmap, ax=ax, label='高程')
    cbar.set_label('高程', fontsize=18)
    cbar.ax.set_yticks([])  # 删除刻度线
    
    # 高亮显示被选中的起降点（红色五角星）
    selected_scatter = ax.scatter(station_coords[selected_original_stations, 0], 
                                station_coords[selected_original_stations, 1], 
                                c='red', label='选中起降点', s=150, marker='*', linewidth=1.5)

    # 绘制接力任务覆盖边界
    relay_boundaries = []
    
    for station_idx in selected_stations:
        relay_boundary = calculate_relay_boundary(station_idx, connectivity_matrix, aggregated_data['station_coords'])
        if relay_boundary is not None:
            relay_boundaries.append(relay_boundary)
    
    # 绘制接力任务覆盖边界（紫色点线）
    for boundary in relay_boundaries:
        if len(boundary) >= 3:
            polygon = mpatches.Polygon(boundary, fill=False, color='purple', 
                                    linestyle=':', linewidth=1.5, alpha=0.7)
            ax.add_patch(polygon)

    # 组合图例
    legend_elements = [
        selected_scatter,
        Line2D([0], [0], color='purple', linestyle=':', linewidth=1.5, label='接力任务覆盖范围')
    ]
    
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(1, 1),
        title='图例说明',
        title_fontsize=12,
        frameon=True,
        framealpha=0.9,
        fontsize=10
    )
    
    add_compass(ax, labelsize=12)
    
    plt.title(f"接力任务覆盖范围 - {len(selected_original_stations)}个起降点", fontsize=16)
    plt.xlabel('UTM X坐标', fontsize=12)
    plt.ylabel('UTM Y坐标', fontsize=12)
    plt.axis('equal')
    ax.grid(False)
    plt.tight_layout()
    plt.show()

def plot_full_connectivity_with_original_data(original_data, selected_original_stations, 
                                            connectivity_matrix, city_shapefile,
                                            aggregated_data, selected_stations):
    """
    图5：选中起降点、接力任务服务范围和最小生成树连线
    """
    print("生成全连通网络可视化（图5：最小生成树 + 接力任务）...")
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    
    # 设置字体
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 从原始数据中提取
    station_coords = original_data['station_coords']
    
    # 加载城市边界
    city_boundary = load_city_boundary(city_shapefile)
    city_boundary.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.5, linewidth=2)
    
    # 加载等高线数据
    elevation_data = load_elevation_data(elevation_file)
    # 生成热力图背景
    xi, yi, zi = generate_elevation_heatmap(elevation_data)
    heatmap = ax.pcolormesh(xi, yi, zi, shading='auto', cmap='terrain', alpha=0.6)
    cbar = plt.colorbar(heatmap, ax=ax, label='高程')
    cbar.set_label('高程', fontsize=18)
    cbar.ax.set_yticks([])  # 删除刻度线
    
    # 高亮显示被选中的起降点（红色五角星）
    selected_scatter = ax.scatter(station_coords[selected_original_stations, 0], 
                                station_coords[selected_original_stations, 1],
                                c='red', label='选中起降点', s=150, marker='*', linewidth=1.5)

    # 绘制接力任务覆盖边界 - 使用集计索引
    relay_boundaries = []
    
    for station_idx in selected_stations:  # 使用集计索引
        if station_idx < len(aggregated_data['station_coords']):
            relay_boundary = calculate_relay_boundary(station_idx, connectivity_matrix, aggregated_data['station_coords'])
            if relay_boundary is not None:
                relay_boundaries.append(relay_boundary)
    
    # 绘制接力任务覆盖边界（紫色点线）
    for boundary in relay_boundaries:
        if len(boundary) >= 3:
            polygon = mpatches.Polygon(boundary, fill=False, color='purple', 
                                    linestyle=':', linewidth=1.5, alpha=0.7)
            ax.add_patch(polygon)

    # 构建最小生成树 - 使用集计数据的连通矩阵
    G = nx.Graph()
    
    # 创建集计索引到原始索引的映射
    agg_to_orig_mapping = {}
    for i, agg_idx in enumerate(selected_stations):
        G.add_node(i)
        agg_to_orig_mapping[i] = selected_original_stations[i]

    # 添加边（直接从连通矩阵获取）- 使用集计索引
    edges_added = []
    for i in range(len(selected_stations)):
        for j in range(i + 1, len(selected_stations)):
            station_i = selected_stations[i]  # 集计索引
            station_j = selected_stations[j]  # 集计索引
            
            # 检查索引是否在连通性矩阵范围内
            if (station_i < connectivity_matrix.shape[0] and 
                station_j < connectivity_matrix.shape[1] and
                connectivity_matrix[station_i, station_j]):
                
                # 使用原始坐标计算距离
                orig_i = agg_to_orig_mapping[i]
                orig_j = agg_to_orig_mapping[j]
                distance = np.linalg.norm(
                    station_coords[orig_i] - station_coords[orig_j]
                )
                G.add_edge(i, j, weight=distance)
                edges_added.append((i, j))

    # 计算最小生成树
    mst_edges = []
    if G.number_of_edges() > 0:
        try:
            mst = nx.minimum_spanning_tree(G)
            mst_edges = list(mst.edges())
            
            # 绘制最小生成树的边
            for edge in mst_edges:
                i, j = edge
                orig_i = agg_to_orig_mapping[i]
                orig_j = agg_to_orig_mapping[j]
                x_coords = [station_coords[orig_i, 0], 
                          station_coords[orig_j, 0]]
                y_coords = [station_coords[orig_i, 1], 
                          station_coords[orig_j, 1]]
                plt.plot(x_coords, y_coords, 'k-', lw=2, alpha=0.8,
                        label='最小生成树边' if '最小生成树边' not in [l.get_label() for l in ax.lines] else "")
        except Exception as e:
            print(f"计算最小生成树时出错: {e}")
            # 如果最小生成树计算失败，绘制所有连通边
            for edge in edges_added:
                i, j = edge
                orig_i = agg_to_orig_mapping[i]
                orig_j = agg_to_orig_mapping[j]
                x_coords = [station_coords[orig_i, 0], 
                          station_coords[orig_j, 0]]
                y_coords = [station_coords[orig_i, 1], 
                          station_coords[orig_j, 1]]
                plt.plot(x_coords, y_coords, 'k-', lw=2, alpha=0.8,
                        label='连通边' if '连通边' not in [l.get_label() for l in ax.lines] else "")

    # 组合图例
    legend_elements = [
        selected_scatter,
        Line2D([0], [0], color='purple', linestyle=':', linewidth=1.5, label='接力任务覆盖范围'),
        Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='最小生成树边')
    ]
    
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(1, 1),
        title='图例说明',
        title_fontsize=12,
        frameon=True,
        framealpha=0.9,
        fontsize=10,
    )
    
    # 添加指北针
    add_compass(ax, labelsize=12)
    
    # 图形设置
    connected_count = len(mst_edges) if mst_edges else len(edges_added)
    
    plt.title(f"最小生成树网络与接力任务覆盖 - {len(selected_original_stations)}个站点，{connected_count}条连接", fontsize=16)
    plt.xlabel('UTM X坐标', fontsize=12)
    plt.ylabel('UTM Y坐标', fontsize=12)
    plt.axis('equal')
    ax.grid(False)
    plt.tight_layout()
    plt.show()

# ========================== 3D可视化配置 ==========================
SAMPLE_RATE = 0.01
ASPECT_RATIO = (5, 5, 2) 
GROUND_COLOR = [0.5, 0.0, 0.5, 1.0]  
SELECTED_COLOR = [1.0, 0.0, 0.0, 1.0]
BATCH_SIZE = 20000
MAX_Z = 400  # 最大Z轴显示范围
CUBE_FACES = np.array([
    [0,1,2], [0,2,3], [4,5,6], [4,6,7],
    [0,1,5], [0,5,4], [1,2,6], [1,6,5],
    [2,3,7], [2,7,6], [3,0,4], [3,4,7]
], dtype=np.uint32)

class VispyVisualizer:
    def __init__(self, building_data, station_data, selected_fids):
        # 解包数据
        self.b_coords, self.b_heights, self.b_elev, self.b_areas, self.b_fids = building_data
        self.s_coords, self.s_heights, self.s_elev, self.s_fids = station_data
        self.selected_fids = selected_fids

        # 🔧 关键修复：确保正确计算选中的建筑索引
        print(f"传入的选中fids: {self.selected_fids}")
        print(f"建筑fids总数: {len(self.b_fids)}")
        
        # 方法1：精确匹配
        self.selected_indices = []
        for fid in self.selected_fids:
            matches = np.where(self.b_fids == fid)[0]
            if len(matches) > 0:
                self.selected_indices.append(matches[0])
                print(f"✅ 找到匹配建筑: FID={fid}, 索引={matches[0]}")
            else:
                print(f"❌ 未找到匹配建筑: FID={fid}")
        
        # 如果方法1找不到，尝试方法2：模糊匹配（处理数据类型不一致）
        if len(self.selected_indices) == 0:
            print("尝试模糊匹配...")
            # 将fid转换为字符串进行比较
            building_fids_str = [str(fid) for fid in self.b_fids]
            selected_fids_str = [str(fid) for fid in self.selected_fids]
            
            for i, selected_fid_str in enumerate(selected_fids_str):
                for j, building_fid_str in enumerate(building_fids_str):
                    if selected_fid_str == building_fid_str:
                        self.selected_indices.append(j)
                        print(f"✅ 通过模糊匹配找到建筑: FID={selected_fid_str}, 索引={j}")
                        break
        
        # 打印选中建筑数量
        print(f"选中的建筑数量: {len(self.selected_indices)}")
        
        if len(self.selected_indices) == 0:
            print("⚠️ 警告：没有找到任何选中的建筑，3D可视化将不会高亮任何建筑")
            # 作为备选，选择前几个建筑进行显示
            if len(self.b_fids) > 0:
                self.selected_indices = [0]  # 至少选择一个建筑
                print("使用第一个建筑作为备选显示")

        # ==== 初始化场景 ====
        self.canvas = scene.SceneCanvas(keys='interactive', size=(1600, 1200), bgcolor='white', show=True)
        self.view = self.canvas.central_widget.add_view()
        
        # ==== 计算数据中心点 ====
        all_coords = np.vstack([self.b_coords, self.s_coords])
        self.x_center = (all_coords[:, 0].min() + all_coords[:, 0].max()) / 2
        self.y_center = (all_coords[:, 1].min() + all_coords[:, 1].max()) / 2
        self.z_center = 0  # 假设数据在平面内

        # ==== 初始化FlyCamera ====
        self.view.camera = scene.FlyCamera(fov=60, center=(self.x_center, self.y_center, self.z_center))
        
        # ==== 设置初始位置和视角 ====
        self.initial_transform = transforms.MatrixTransform()
        self.initial_transform.translate((0, 0, -2000))  # 初始位置（沿Z轴负方向）
        self.view.camera.transform = self.initial_transform
        
        # ==== 保存初始状态 ====
        self._init_transform = self.view.camera.transform.matrix.copy()  # 位置矩阵
        self._init_center = self.view.camera.center
        # ==== 绑定事件 ====
        self.canvas.events.mouse_press.connect(self.on_mouse_press)
        self.canvas.events.mouse_move.connect(self.on_mouse_move)
        self.canvas.events.mouse_wheel.connect(self.on_mouse_wheel)
        self.canvas.events.key_press.connect(self.on_key_press)
        
        # ==== 状态变量 ====
        self._mouse_last_pos = None
        self._current_button = None

        # 设置空间索引
        self.kdtree = KDTree(self.b_coords)
        
        # 颜色映射参数
        self.max_height = np.max(self.b_heights)
        self.cmap = self.create_colormap()
        
        # 初始化可视化对象
        self.building_visuals = []
        self.ground_stations = []
        self.labels = []
        
        # 计算建筑自身高度的最大值（不包含高程）
        self.max_building_height = np.max(self.b_heights)
        
        # 创建颜色映射（0到最大建筑高度）
        self.cmap = self.create_colormap()

        # 性能监控
        self.mem_start = psutil.Process().memory_info().rss
    
    def on_mouse_press(self, event):
        """记录按下的鼠标键"""
        self._mouse_last_pos = event.pos
        self._current_button = event.button  

    def on_mouse_move(self, event):
        if event.is_dragging and self._current_button:  # 使用正确的变量名_current_button
            dx = event.pos[0] - self._mouse_last_pos[0]
            dy = event.pos[1] - self._mouse_last_pos[1]
            self._mouse_last_pos = event.pos

            # 左键：旋转操作（保留默认行为）
            if self._current_button == 1:
                return

            # 右键：平移操作
            elif self._current_button == 2:
                tr = self.view.camera.transform
                scale_factor = np.linalg.norm(self.view.camera.scale_factor or 1.0)
                delta_ndc = np.array([dx, -dy, 0, 0]) * (0.002 * scale_factor)
                world_delta = tr.map(delta_ndc) - tr.map([0, 0, 0, 0])
                self.view.camera.center -= world_delta[:3]

    def on_mouse_wheel(self, event):
        """滚轮缩放（沿Z轴移动）"""
        delta = event.delta[1]
        zoom_speed = 100
        if delta > 0:  # 向上滚动：靠近数据
            self.view.camera.transform.translate((0, 0, zoom_speed))
        else:  # 向下滚动：远离数据
            self.view.camera.transform.translate((0, 0, -zoom_speed))

    def on_key_press(self, event):
        if event.text == ' ':
            self.reset_view()
        elif event.text.lower() == 'r':
            self.focus_on_scene()

    def focus_on_scene(self):
        # 自动调整视图范围
        self.view.camera.set_range(
            x=(self.b_coords[:, 0].min(), self.b_coords[:, 0].max()),
            y=(self.b_coords[:, 1].min(), self.b_coords[:, 1].max()),
            z=(0, MAX_Z)
        )

    def reset_view(self):
        """完全重置视角（位置+角度+缩放）"""
        self.view.camera.transform.matrix = self._init_transform.copy()
        self.view.camera.center = self._init_center
        self.canvas.update()

    def create_colormap(self):
        """创建从蓝到红的渐变色表"""
        colors = np.array([
            [0.0, 0.0, 1.0, 1.0],  # 蓝色
            [0.0, 1.0, 1.0, 1.0],  # 青色
            [0.0, 1.0, 0.0, 1.0],  # 绿色
            [1.0, 1.0, 0.0, 1.0],  # 黄色
            [1.0, 0.0, 0.0, 1.0]   # 红色
        ])
        return Colormap(colors)
    
    def height_to_color(self, height):
        """将高度映射到颜色"""
        ratio = np.array([height / self.max_height])  # 转换为数组
        return self.cmap.map(ratio)[0][:4]  # 提取第一个结果的颜色值
    
    def create_building_mesh(self, coords, heights, elevs, areas, colors=None):
        """生成建筑网格（底部到顶部渐变）"""
        vertices = []
        indices = []
        valid_colors = []
        
        for idx in range(len(coords)):
            x, y = coords[idx]
            z = elevs[idx]  # 建筑基底高程
            h = heights[idx]  # 建筑自身高度
            area = areas[idx]
            side = np.sqrt(area) / 2
            
            # 生成立方体顶点（高程作为基底）
            verts = np.array([
                [x-side, y-side, z],    [x+side, y-side, z],     # 底部四顶点
                [x+side, y+side, z],    [x-side, y+side, z],
                [x-side, y-side, z+h], [x+side, y-side, z+h],    # 顶部四顶点
                [x+side, y+side, z+h], [x-side, y+side, z+h]
            ], dtype=np.float32)
            
            if colors is not None:
                # 使用传入的固定颜色（如红色或紫色）
                vertex_color = colors[idx]
                valid_colors.extend([vertex_color] * 8)  # 所有顶点同一颜色
            else:
                # 原逻辑：基于高度渐变
                color_bottom = self.cmap.map(np.array([0.0]))[0][:4]
                color_top = self.cmap.map(np.array([h / self.max_building_height]))[0][:4] 
                valid_colors.extend([color_bottom]*4 + [color_top]*4)
            
            # 添加顶点和面索引
            base_idx = len(vertices)
            vertices.extend(verts)
            indices.extend(CUBE_FACES + base_idx)
        
        return scene.visuals.Mesh(
            vertices=np.array(vertices),
            faces=np.array(indices).reshape(-1,3),
            vertex_colors=np.array(valid_colors),
            shading=None,
            parent=self.view.scene
        )
    
    def add_highlight_beam(self, coords, elevs, heights, color):
        """添加垂直光柱"""
        for idx in range(len(coords)):
            x, y = coords[idx]
            z_base = elevs[idx]
            beam_height = 400  # 光柱高度
            
            # 光柱顶点（从建筑顶部到400米高处）
            vertices = np.array([
                [x, y, z_base + heights[idx]],  # 起点
                [x, y, z_base + heights[idx] + beam_height]  # 终点
            ], dtype=np.float32)
            
            # 创建线状光柱
            beam = scene.visuals.Line(
                pos=vertices,
                color=color,
                width=5,  # 线宽
                parent=self.view.scene
            )
            beam.transform = transforms.MatrixTransform()  # 确保坐标正确
            beam.set_gl_state('translucent')

    def visualize(self):
        print("开始构建场景...")
        start_time = time.time()
        
        # 获取总建筑数
        total = len(self.b_coords)

        # 分批次处理建筑数据（修复越界问题）
        for i in range(0, total, BATCH_SIZE):
            end_idx = min(i + BATCH_SIZE, total)  # 确保不越界
            batch_slice = slice(i, end_idx)
            
            # 提取当前批次数据
            batch_coords = self.b_coords[batch_slice]
            batch_heights = self.b_heights[batch_slice]
            batch_elev = self.b_elev[batch_slice]
            batch_areas = self.b_areas[batch_slice]
            
            # 处理颜色
            colors = []
            for h in batch_heights:
                colors.append(self.height_to_color(h))
            
            # 创建批次可视化对象
            batch_mesh = self.create_building_mesh(batch_coords, batch_heights, batch_elev, batch_areas)
            self.building_visuals.append(batch_mesh)
            print(f"已处理 {end_idx}/{total} 栋建筑")
        
        # 添加选中建筑高亮
        selected_indices = np.where(np.isin(self.b_fids, self.selected_fids))[0]
        if len(selected_indices) > 0:
            selected_mesh = self.create_building_mesh(
                self.b_coords[selected_indices],
                self.b_heights[selected_indices],
                self.b_elev[selected_indices],
                self.b_areas[selected_indices],
                colors=np.tile(SELECTED_COLOR, (len(selected_indices), 1))
            )
            self.building_visuals.append(selected_mesh)
        
        # 添加地面站
        ground_mask = self.s_heights == 0
        if ground_mask.any():
            # 提取地面站数据
            ground_coords = self.s_coords[ground_mask]
            ground_elev = self.s_elev[ground_mask]
            ground_areas = np.full(ground_coords.shape[0], 10.0)  # 固定面积
            if len(ground_coords) > 0:
                # 生成地面站网格
                ground_heights = np.full(len(ground_coords), 5.0)  # 高度5米
                ground_mesh = self.create_building_mesh(
                    ground_coords,
                    ground_heights,  
                    ground_elev,
                    ground_areas, 
                    colors=np.tile(GROUND_COLOR, (len(selected_indices), 1))
                )
                self.ground_stations.append(ground_mesh)
        
        # 为选中目标添加红色光柱
        selected_coords = self.b_coords[selected_indices]
        selected_elev = self.b_elev[selected_indices]
        selected_heights = self.b_heights[selected_indices]
        self.add_highlight_beam(selected_coords, selected_elev, selected_heights, (1,0,0,0.7))
        
        # 为地面站添加紫色光柱
        ground_mask = self.s_heights == 0
        ground_coords = self.s_coords[ground_mask]
        ground_elev = self.s_elev[ground_mask]
        self.add_highlight_beam(ground_coords, ground_elev, np.zeros(len(ground_coords)), (0.5,0,0.5,0.7))
        
        # 设置Z轴范围
        self.view.camera.set_range(z=(0, MAX_Z))
        
        # 性能统计
        mem_used = (psutil.Process().memory_info().rss - self.mem_start) // 1024**2
        print(f"场景构建完成! 耗时: {time.time()-start_time:.2f}s, 内存占用: {mem_used}MB")

def create_3d_visualization_with_original_data(original_data, selected_original_stations):
    """
    3D可视化 - 使用原始数据
    """
    print("准备3D可视化数据（使用原始数据）...")
    
    # 从原始数据中提取
    building_coords = original_data['building_coords']
    building_heights = original_data['building_heights']
    building_elevations = original_data['building_elevations']
    building_areas = original_data['building_areas']
    building_fids = original_data['building_fids']
    
    station_coords = original_data['station_coords']
    station_heights = original_data['station_heights']
    station_elevations = original_data['station_elevations']
    station_fids = original_data['station_fids']
    
    # 获取选中的建筑fid
    selected_building_fids = []
    for station_idx in selected_original_stations:
        station_fid = station_fids[station_idx]
        if station_fid in building_fids:
            selected_building_fids.append(station_fid)
    
    print(f"选中的建筑数量: {len(selected_building_fids)}")
    
    if len(selected_building_fids) > 0:
        building_data = (building_coords, building_heights, building_elevations, building_areas, building_fids)
        station_data = (station_coords, station_heights, station_elevations, station_fids)
        
        visualizer = VispyVisualizer(building_data, station_data, selected_building_fids)
        visualizer.visualize()
        app.run()
    else:
        print("❌ 没有找到任何选中的建筑，跳过3D可视化")

# ========================== 收敛性分析与性能可视化 ==========================
def plot_convergence_analysis(performance_data, selected_stations, station_coords, demand_coords, cover_radii):
    """
    收敛性分析 - 移除理论最大覆盖率相关内容
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 设置字体
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    iteration_data = performance_data['iteration']
    
    # 1. 合并图：覆盖率收敛曲线 + 权重变化（双Y轴）
    if (iteration_data['iterations'] and iteration_data['coverage_rates'] and
        iteration_data['coverage_weights'] and iteration_data['connectivity_weights']):
        
        iterations = iteration_data['iterations']
        cover_rates = iteration_data['coverage_rates']
        coverage_weights = iteration_data['coverage_weights']
        connectivity_weights = iteration_data['connectivity_weights']
        
        # 确保数组长度一致
        min_len = min(len(iterations), len(cover_rates), len(coverage_weights), len(connectivity_weights))
        iterations = iterations[:min_len]
        cover_rates = cover_rates[:min_len]
        coverage_weights = coverage_weights[:min_len]
        connectivity_weights = connectivity_weights[:min_len]
        
        # 创建双Y轴
        ax1_left = ax1
        ax1_right = ax1.twinx()
        
        # 左侧Y轴：覆盖率
        line1 = ax1_left.plot(iterations, cover_rates, 'b-', linewidth=3, label='覆盖率')[0]
        ax1_left.axhline(y=target_coverage_percentage, color='r', linestyle='--', 
                        linewidth=2, label=f'目标覆盖率 ({target_coverage_percentage:.1%})')
        ax1_left.set_xlabel('迭代次数', fontsize=14)
        ax1_left.set_ylabel('覆盖率', fontsize=14, color='black')
        ax1_left.tick_params(axis='y', labelcolor='black')
        ax1_left.set_ylim(0, 1)
        
        # 右侧Y轴：权重
        line2 = ax1_right.plot(iterations, coverage_weights, 'orange', linewidth=2, 
                            linestyle='-', label='覆盖权重')[0]
        line3 = ax1_right.plot(iterations, connectivity_weights, 'purple', linewidth=2, 
                            linestyle='-', label='连通权重')[0]
        ax1_right.set_ylabel('权重值', fontsize=14, color='black')
        ax1_right.tick_params(axis='y', labelcolor='black')
        ax1_right.set_ylim(0, max(max(coverage_weights), max(connectivity_weights)) * 1.1)
        
        # 合并图例
        lines = [line1, ax1_left.get_lines()[1], line2, line3]
        labels = [l.get_label() for l in lines]
        ax1_left.legend(lines, labels, loc='center right')
        
        ax1_left.set_title('覆盖率收敛与权重变化', fontsize=16)
        ax1_left.grid(True, alpha=0.3)
    
    # 2. 算法效率统计（移除理论最大覆盖率相关项）
    total_stations = len(station_coords)
    final_coverage = calculate_coverage(selected_stations, station_coords, demand_coords, cover_radii)
    
    efficiency_metrics = {
        '候选站点总数': total_stations,
        '最终选中站点': len(selected_stations),
        '选择比例': f'{len(selected_stations) / total_stations * 100:.2f}%',
        '目标覆盖率': f'{target_coverage_percentage * 100:.2f}%',
        '最终覆盖率': f'{final_coverage * 100:.2f}%',
        '总运行时间': f'{performance_data["total_time"]:.2f}秒',
        '总迭代次数': len(iteration_data['iterations'])
    }
    
    # 添加连通修复和冗余剪枝信息
    if 'repair' in performance_data:
        efficiency_metrics['连通修复添加站点'] = performance_data['repair']['added_stations']
    if 'prune' in performance_data:
        efficiency_metrics['冗余剪枝移除站点'] = performance_data['prune']['removed_stations']
    
    metrics_text = "\n".join([f"{k}: {v}" for k, v in efficiency_metrics.items()])
    
    ax2.text(0.1, 0.9, metrics_text, transform=ax2.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('算法效率统计', fontsize=16)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 输出详细性能报告
    print("\n" + "="*60)
    print("算法性能分析报告")
    print("="*60)
    for key, value in efficiency_metrics.items():
        print(f"{key}: {value}")

def validate_cost_selection(grid_original_info, selected_original_stations, original_data):
    """验证成本选择策略是否正确执行"""
    print("\n=== 成本选择策略验证 ===")
    
    correctly_selected = 0
    total_checked = 0
    
    station_heights = original_data['station_heights']
    
    for orig_idx in selected_original_stations[:10]:  # 检查前10个站点
        # 查找这个原始站点属于哪个网格
        for grid_key, grid_info in grid_original_info.items():
            grid_originals = grid_info['all_originals']
            grid_original_indices = [orig['original_index'] for orig in grid_originals]
            
            if orig_idx in grid_original_indices:
                total_checked += 1
                
                # 找到这个网格内所有站点的成本
                grid_costs = []
                for orig_data in grid_originals:
                    cost = orig_data['height'] * cost_multiplier + fixed_cost
                    grid_costs.append((orig_data['original_index'], cost))
                
                # 按成本排序
                grid_costs.sort(key=lambda x: x[1])
                lowest_cost_idx = grid_costs[0][0]
                
                # 检查选中的是否是最低成本的
                if orig_idx == lowest_cost_idx:
                    correctly_selected += 1
                    print(f"✅ 网格 {grid_key}: 正确选择了成本最低的站点 (成本: {grid_costs[0][1]:,.0f}元)")
                else:
                    selected_cost = next(cost for idx, cost in grid_costs if idx == orig_idx)
                    lowest_cost = grid_costs[0][1]
                    print(f"❌ 网格 {grid_key}: 错误! 选中成本: {selected_cost:,.0f}元, 最低成本: {lowest_cost:,.0f}元")
                
                break
    
    if total_checked > 0:
        accuracy = correctly_selected / total_checked * 100
        print(f"成本选择准确率: {accuracy:.1f}% ({correctly_selected}/{total_checked})")
    else:
        print("未找到匹配的网格信息进行验证")

def calculate_final_cost_with_original_data(original_data, selected_original_stations):
    """计算最终建设成本 - 使用原始数据"""
    station_heights = original_data['station_heights']
    station_fids = original_data['station_fids']
    
    total_cost = 0
    cost_details = []
    
    for station_idx in selected_original_stations:
        cost = station_heights[station_idx] * cost_multiplier + fixed_cost
        total_cost += cost
        cost_details.append({
            'index': station_idx,
            'fid': station_fids[station_idx],
            'height': station_heights[station_idx],
            'cost': cost
        })
    
    # 按成本排序
    cost_details.sort(key=lambda x: x['cost'])
    
    print(f"\n最终成本分析（基于原始数据）:")
    print(f"总建设成本: {total_cost:,.0f} 元")
    print(f"选中起降点数量: {len(selected_original_stations)}")
    print(f"平均每个站点成本: {total_cost/len(selected_original_stations):,.0f} 元")
    
    print(f"\n成本最低的5个站点:")
    for i, detail in enumerate(cost_details[:5]):
        print(f"  {i+1}. 索引{detail['index']}, FID{detail['fid']}, 高度{detail['height']:.1f}m, 成本{detail['cost']:,.0f}元")
    
    print(f"\n成本最高的5个站点:")
    for i, detail in enumerate(cost_details[-5:]):
        print(f"  {i+1}. 索引{detail['index']}, FID{detail['fid']}, 高度{detail['height']:.1f}m, 成本{detail['cost']:,.0f}元")
    
    return total_cost, cost_details

# ========================== 主程序执行 ==========================
if __name__ == '__main__':
    print("开始执行低空起降点选址优化算法（两套数据系统）...")
    overall_start_time = time.time()

    # 性能数据收集
    performance_data = {}

    # 步骤1-2: 基于综合效率迭代选点 - 使用集计数据
    print("=== 算法计算阶段（使用集计数据）===")
    selected_stations, iteration_data, reverse_index_data, connectivity_matrix = integrated_efficiency_selection_with_aggregated_data(
        aggregated_data, grid_original_info,
        cover_radii, relay_radii, city_shapefile,
        enable_realtime_visualization=True,
        external_cache_manager=cache
    )

    performance_data['iteration'] = iteration_data
    print(f"初始选点完成，选中 {len(selected_stations)} 个集计起降点")

    # 步骤3: 全局连通性修复 - 使用集计数据
    selected_stations, repair_data = mst_connectivity_repair_with_aggregated_data(
        selected_stations, aggregated_data, grid_original_info
    )

    performance_data['repair'] = repair_data
    print(f"连通性修复后，共有 {len(selected_stations)} 个起降点")

    # 步骤4: 冗余剪枝 - 使用集计数据  
    selected_stations, prune_data = redundancy_pruning_with_aggregated_data(
        selected_stations, aggregated_data, grid_original_info
    )

    performance_data['prune'] = prune_data
    print(f"冗余剪枝后，最终选中 {len(selected_stations)} 个起降点")

    # 总时间统计
    total_time = time.time() - overall_start_time
    performance_data['total_time'] = total_time
    print(f"算法总运行时间: {total_time:.2f}秒")

    # ========================== 结果映射和可视化 ==========================
    
    print("=== 结果映射和可视化阶段（使用原始数据）===")
    
    # 将选中的集计站点映射回原始站点
    selected_original_stations, selected_original_fids = get_original_stations_from_mapping_enhanced(
        selected_stations, grid_original_info, aggregated_station_fids
    )

    # 计算成本 - 使用原始数据
    total_cost, cost_details = calculate_final_cost_with_original_data(original_data, selected_original_stations)

    # 验证成本选择策略
    validate_cost_selection(grid_original_info, selected_original_stations, original_data)

    # 可视化 - 使用原始数据
    print("=== 生成可视化结果（使用原始数据）===")
    
    # 1. 完整解决方案可视化
    plot_solution_with_original_data(original_data, selected_original_stations, 
                                reverse_index_data, connectivity_matrix, city_shapefile,
                                aggregated_data, selected_stations)

    # 2. 选中站点与双任务覆盖范围可视化（去掉候选起降点）
    plot_selected_with_both_coverage_with_original_data(original_data, selected_original_stations, 
                                                    reverse_index_data, connectivity_matrix, city_shapefile,
                                                    aggregated_data, selected_stations)

    # 3. 普通任务覆盖范围可视化（去掉接力任务）
    plot_selected_with_normal_coverage_with_original_data(original_data, selected_original_stations, 
                                                        reverse_index_data, city_shapefile,
                                                        aggregated_data, selected_stations)

    # 4. 接力任务覆盖范围可视化（只有选中起降点和接力任务）
    plot_selected_with_relay_coverage_with_original_data(original_data, selected_original_stations, 
                                                    connectivity_matrix, city_shapefile,
                                                    aggregated_data, selected_stations)

    # 5. 最小生成树可视化（只有选中起降点和最小生成树连线）
    plot_full_connectivity_with_original_data(original_data, selected_original_stations, 
                                            connectivity_matrix, city_shapefile,
                                            aggregated_data, selected_stations)

    # 6. 3D可视化 - 使用原始数据
    if District == '全部区' and len(original_data['building_coords']) > 0:
        print("是否运行3D可视化？(y/n)")
        user_input = input().lower()
        if user_input == 'y':
            create_3d_visualization_with_original_data(original_data, selected_original_stations)

    # 7. 收敛性分析
    print("开始算法收敛性分析...")
    plot_convergence_analysis(performance_data, selected_stations, station_coords, demand_coords, cover_radii)

    print("======= 算法执行完成 =======")
    print(f"最终选中起降点数量: {len(selected_original_stations)} (原始数据)")
    print(f"总建设成本: {total_cost:,.0f} 元")
    print(f"总运行时间: {performance_data['total_time']:.2f}秒")

    # 输出选中起降点的详细信息
    print("\n选中起降点详细信息:")
    for i, station_idx in enumerate(selected_original_stations):
        height = original_data['station_heights'][station_idx]
        elevation = original_data['station_elevations'][station_idx]
        cost = height * cost_multiplier + fixed_cost
        cover_radius = cover_radii[station_idx] if station_idx < len(cover_radii) else 0
        relay_radius = relay_radii[station_idx] if station_idx < len(relay_radii) else 0
        print(f"起降点 {i+1}: FID={original_data['station_fids'][station_idx]}, "
            f"高度={height:.1f}m, 高程={elevation:.1f}m, "
            f"覆盖半径={cover_radius:.0f}m, 接力半径={relay_radius:.0f}m, "
            f"成本={cost:,.0f}元")