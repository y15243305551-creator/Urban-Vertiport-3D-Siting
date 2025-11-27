import numpy as np
import math
from pyproj import Proj
import time
from tqdm import tqdm
import geopandas as gpd
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib

# 坐标转换函数
def gcj02_to_utm(lons, lats):
    """坐标转换工具函数"""
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

def load_city_boundary(city_shapefile):
    """加载城市边界数据"""
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

def load_elevation_data(elevation_file):
    """加载等高线数据"""
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

def validate_cost_selection(grid_original_info, selected_original_stations, original_data):
    """验证成本选择策略是否正确执行"""
    print("\n=== 成本选择策略验证 ===")
    
    correctly_selected = 0
    total_checked = 0
    
    station_heights = original_data['station_heights']
    cost_multiplier = 5000
    fixed_cost = 150000
    
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
    cost_multiplier = 5000
    fixed_cost = 150000
    
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

def plot_convergence_analysis(performance_data, selected_stations, station_coords, demand_coords, cover_radii):
    """
    收敛性分析 - 移除理论最大覆盖率相关内容
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 设置字体
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    iteration_data = performance_data['iteration']
    target_coverage_percentage = 0.95
    
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
    
def calculate_theoretical_max_coverage(reverse_index_data, total_demand_points):
    """
    使用反向索引计算理论最大覆盖率（所有候选站点都能建设时能达到的覆盖率）
    """
    print("计算理论最大覆盖率（使用反向索引）...")
    
    # 使用反向索引计算：只要需求点在反向索引中，就表示至少有一个站点可以覆盖它
    covered_demand_count = len(reverse_index_data)
    theoretical_max = covered_demand_count / total_demand_points
    
    print(f"理论最大覆盖率: {theoretical_max:.2%}")
    print(f"可覆盖需求点: {covered_demand_count}/{total_demand_points}")
    
    return theoretical_max