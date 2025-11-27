import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Proj
from tqdm import tqdm
import os
import math
from collections import defaultdict

class DataLoader:
    """数据加载和预处理类"""
    
    def __init__(self, config=None, city_shapefile=None, station_file=None, 
                 demand_file=None, building_file=None, elevation_file=None, district=None):
        """
        初始化数据加载器
        
        支持两种初始化方式：
        1. 通过config字典
        2. 通过单独的文件路径参数
        """
        if config is not None:
            # 从配置字典初始化
            from .utils import build_data_paths
            paths = build_data_paths(config)
            self.city_shapefile = paths['city_shapefile']
            self.station_file = paths['station_file']
            self.demand_file = paths['demand_file']
            self.building_file = paths['building_file']
            self.elevation_file = paths['elevation_file']
            self.district = config['data_files']['district']
        else:
            # 从单独参数初始化
            self.city_shapefile = city_shapefile
            self.station_file = station_file
            self.demand_file = demand_file
            self.building_file = building_file
            self.elevation_file = elevation_file
            self.district = district if district else '全部区'
        
        # 从配置或默认值获取参数
        if config and 'model' in config:
            self.cost_multiplier = config['model'].get('cost_multiplier', 5000)
            self.fixed_cost = config['model'].get('fixed_cost', 150000)
            self.b = config['physical'].get('b', 200)
        else:
            self.cost_multiplier = 5000
            self.fixed_cost = 150000
            self.b = 200
        
    def gcj02_to_utm(self, lons, lats):
        """坐标转换函数"""
        proj = Proj(
            proj='utm',
            zone=54,
            ellips='krass',
            towgs84='0,0,0,0,0,0,0',
            preserve_units=True
        )
        if isinstance(lons, (list, tuple, np.ndarray)):
            return np.array([proj(lon, lat) for lon, lat in zip(lons, lats)])
        else:
            return proj(lons, lats)
    
    def load_data(self):
        """加载所有原始数据"""
        print(f"------读取起降点和需求点数据------")
        
        # 读取数据
        station_df = pd.read_excel(self.station_file)
        demand_df = pd.read_excel(self.demand_file)
        building_df = pd.read_excel(self.building_file)
        
        # 提取起降点数据
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
        
        # 提取建筑数据
        building_areas = np.array([
            row['roof_area'] for _, row in tqdm(building_df.iterrows(), total=building_df.shape[0], 
                                                desc="读取屋顶面积", unit="行", ncols=100)
        ])
        
        if self.district == '全部区':
            building_heights = np.array([
                row['Height'] for _, row in tqdm(building_df.iterrows(), total=building_df.shape[0], 
                                               desc="读取建筑高度", unit="行", ncols=100)
            ])  
            building_elevations = np.array([
                row['DEM高度'] for _, row in tqdm(building_df.iterrows(), total=building_df.shape[0], 
                                                desc="读取高程高度", unit="行", ncols=100)
            ]) 
            
            # 转换建筑坐标
            print("读取建筑坐标...")
            building_lons = building_df['经度'].values
            building_lats = building_df['纬度'].values
            building_coords = np.array([
                self.gcj02_to_utm(lon, lat) 
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
        
        # 提取需求点数据
        demand_elevations = np.array([
            row['DEM高度'] if 'DEM高度' in row else 0
            for _, row in tqdm(demand_df.iterrows(), total=demand_df.shape[0],
                              desc="读取需求点高程", unit="行", ncols=100)
        ])
        
        # 转换起降点坐标
        print("读取起降点坐标...")
        station_lons = station_df['经度'].values
        station_lats = station_df['纬度'].values
        station_coords = np.array([
            self.gcj02_to_utm(lon, lat) 
            for lon, lat in tqdm(zip(station_lons, station_lats), 
                               total=len(station_lons), desc="转换起降点坐标", unit="行", ncols=100)
        ])
        
        # 转换需求点坐标
        print("读取需求点坐标...")
        demand_lons = demand_df['经度'].values
        demand_lats = demand_df['纬度'].values
        demand_coords = np.array([
            self.gcj02_to_utm(lon, lat) 
            for lon, lat in tqdm(zip(demand_lons, demand_lats), 
                               total=len(demand_lons), desc="转换需求点坐标", unit="行", ncols=100)
        ])
        
        # 返回原始数据
        original_data = {
            'station_coords': station_coords,
            'demand_coords': demand_coords,
            'building_coords': building_coords,
            'station_heights': station_heights,
            'station_elevations': station_elevations,
            'station_fids': station_fids,
            'demand_elevations': demand_elevations,
            'building_heights': building_heights,
            'building_elevations': building_elevations,
            'building_areas': building_areas,
            'building_fids': building_fids,
            'station_df': station_df
        }
        
        return original_data
    
    def load_and_process(self, grid_size=300):
        """加载数据并进行网格化处理"""
        # 加载原始数据
        original_data = self.load_data()
        
        # 创建网格化处理器
        aggregator = GridAggregator(grid_size=grid_size, cost_multiplier=self.cost_multiplier, fixed_cost=self.fixed_cost)
        
        # 进行网格化处理
        aggregated_data, selected_mapping, grid_original_info = aggregator.aggregate_all_data(original_data)
        
        print("数据集计完成！")
        print(f"集计后数据量:")
        print(f"  建筑: {len(aggregated_data['building_coords'])}")
        print(f"  起降点: {len(aggregated_data['station_coords'])}") 
        print(f"  需求点: {len(aggregated_data['demand_coords'])}")
        
        return aggregated_data, original_data, grid_original_info


class GridAggregator:
    """网格化数据聚合器"""
    
    def __init__(self, grid_size=300, cost_multiplier=5000, fixed_cost=150000):
        self.grid_size = grid_size
        self.cost_multiplier = cost_multiplier
        self.fixed_cost = fixed_cost
        self.b = 200  # 平飞高度
    
    def aggregate_all_data(self, original_data, min_building_height_threshold=None):
        """集计所有数据"""
        if min_building_height_threshold is None:
            min_building_height_threshold = self.b
        
        print(f"开始网格化集计，网格大小: {self.grid_size}m，超高建筑阈值: {min_building_height_threshold}m")
        
        # 提取数据
        building_coords = original_data['building_coords']
        building_heights = original_data['building_heights']
        building_elevations = original_data['building_elevations']
        building_areas = original_data['building_areas']
        building_fids = original_data['building_fids']
        
        station_coords = original_data['station_coords']
        station_heights = original_data['station_heights']
        station_elevations = original_data['station_elevations']
        station_fids = original_data['station_fids']
        
        demand_coords = original_data['demand_coords']
        demand_elevations = original_data['demand_elevations']
        station_df = original_data['station_df']
        
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
        x_bins = np.arange(min_x, max_x + self.grid_size, self.grid_size)
        y_bins = np.arange(min_y, max_y + self.grid_size, self.grid_size)
        
        print(f"网格范围: X[{min_x:.0f}, {max_x:.0f}], Y[{min_y:.0f}, {max_y:.0f}]")
        print(f"网格数量: {len(x_bins)-1} x {len(y_bins)-1} = {(len(x_bins)-1)*(len(y_bins)-1)}")
        
        # 集计各类数据
        aggregated_buildings = self.aggregate_buildings_by_grid(
            building_coords, building_heights, building_elevations, building_areas, building_fids,
            x_bins, y_bins, min_building_height_threshold
        )
        
        aggregated_stations, selected_mapping, grid_original_info = self.aggregate_stations_by_grid_with_selection(
            station_coords, station_heights, station_elevations, station_fids,
            x_bins, y_bins, station_df, strategy="lowest_cost"
        )
        
        aggregated_demands = self.aggregate_demands_by_grid(
            demand_coords, demand_elevations,
            x_bins, y_bins
        )
        
        # 组合所有集计数据
        (building_coords_agg, building_heights_agg, building_elevations_agg, 
         building_areas_agg, building_fids_agg) = aggregated_buildings
        
        (station_coords_agg, station_heights_agg, station_elevations_agg, 
         station_fids_agg) = aggregated_stations
        
        (demand_coords_agg, demand_elevations_agg) = aggregated_demands
        
        aggregated_data = {
            'station_coords': station_coords_agg,
            'demand_coords': demand_coords_agg,
            'building_coords': building_coords_agg,
            'station_heights': station_heights_agg,
            'station_elevations': station_elevations_agg,
            'station_fids': station_fids_agg,
            'demand_elevations': demand_elevations_agg,
            'building_heights': building_heights_agg,
            'building_elevations': building_elevations_agg,
            'building_areas': building_areas_agg,
            'building_fids': building_fids_agg
        }
        
        return aggregated_data, selected_mapping, grid_original_info
    
    def aggregate_buildings_by_grid(self, building_coords, building_heights, building_elevations, 
                                   building_areas, building_fids, x_bins, y_bins, height_threshold):
        """集计建筑数据"""
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
        
        return [
            np.array(aggregated_coords),
            np.array(aggregated_heights),      # 原始建筑物高度
            np.array(aggregated_elevations),   # 高程
            np.array(aggregated_areas),
            np.array(aggregated_fids)
        ]
    
    def aggregate_stations_by_grid_with_selection(self, station_coords, station_heights, station_elevations, 
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
                selected_original_idx = self.select_representative_station_in_grid_direct(
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
    
    def select_representative_station_in_grid_direct(self, grid_data, strategy="lowest_cost"):
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
                cost = height * self.cost_multiplier + self.fixed_cost
                costs.append(cost)
            
            # 选择成本最低的站点
            min_cost_idx = np.argmin(costs)
            return grid_data['original_indices'][min_cost_idx]
        
        else:
            # 其他策略的实现（如果需要）
            costs = []
            for height in grid_data['heights']:
                cost = height * self.cost_multiplier + self.fixed_cost
                costs.append(cost)
            
            min_cost_idx = np.argmin(costs)
            return grid_data['original_indices'][min_cost_idx]
    
    def aggregate_demands_by_grid(self, demand_coords, demand_elevations, x_bins, y_bins):
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

    def get_original_stations_from_mapping_enhanced(self, selected_grid_stations, grid_original_info, aggregated_station_fids):
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
                        cost = original_data['height'] * self.cost_multiplier + self.fixed_cost
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