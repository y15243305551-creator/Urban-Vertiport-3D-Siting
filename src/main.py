"""
城市低空起降点三维选址优化系统 - 主程序入口
"""

import sys
import os
import yaml
import time

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataLoader
from optimization_algorithm import VertiportOptimizer
from utils import (validate_cost_selection, calculate_final_cost_with_original_data,
                  plot_convergence_analysis)

def load_config(config_path="config/parameters.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    print("="*60)
    print("城市低空起降点三维选址优化系统")
    print("="*60)
    
    # 加载配置
    config = load_config()
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 初始化数据加载器
        data_loader = DataLoader(
            city_shapefile=config['data_files']['city_shapefile'],
            station_file=config['data_files']['station_file'],
            demand_file=config['data_files']['demand_file'],
            building_file=config['data_files']['building_file'],
            elevation_file=config['data_files']['elevation_file'],
            district='深圳市'
        )
        
        # 加载和处理数据
        print("\n1. 数据预处理...")
        aggregated_data, original_data, grid_original_info = data_loader.load_and_process(
            grid_size=config['model']['grid_size']
        )
        
        # 初始化优化器
        optimizer = VertiportOptimizer(
            target_coverage=config['model']['target_coverage'],
            cover_weight=1.5,
            connectivity_weight=0.5
        )
        
        # 运行优化算法
        print("\n2. 运行优化算法...")
        results = optimizer.solve(
            aggregated_data=aggregated_data,
            grid_original_info=grid_original_info,
            city_shapefile=config['data_files']['city_shapefile'],
            physical_params=config['physical'],
            drone_params=config['drone'],
            model_params=config['model']
        )
        
        # 将选中的集计站点映射回原始站点
        print("\n3. 结果映射...")
        from data_preprocessing import GridAggregator
        aggregator = GridAggregator()
        selected_original_stations, selected_original_fids = aggregator.get_original_stations_from_mapping_enhanced(
            results['selected_stations'], grid_original_info, aggregated_data['station_fids']
        )
        
        # 计算成本
        print("\n4. 成本分析...")
        total_cost, cost_details = calculate_final_cost_with_original_data(original_data, selected_original_stations)
        
        # 验证成本选择策略
        validate_cost_selection(grid_original_info, selected_original_stations, original_data)
        
        # 收敛性分析
        print("\n5. 生成收敛性分析...")
        plot_convergence_analysis(
            performance_data=results['performance_data'],
            selected_stations=results['selected_stations'],
            station_coords=aggregated_data['station_coords'],
            demand_coords=aggregated_data['demand_coords'],
            cover_radii=results['cover_radii']
        )
        
        # 总时间统计
        total_time = time.time() - start_time
        
        # 输出最终结果
        print("\n" + "="*60)
        print("🎉 优化完成!")
        print("="*60)
        print(f"最终选中起降点数量: {len(selected_original_stations)} (原始数据)")
        print(f"总建设成本: {total_cost:,.0f} 元")
        print(f"最终覆盖率: {results['performance_data']['prune']['final_coverage']:.2%}")
        print(f"网络连通性: {'是' if results['performance_data']['prune']['is_connected'] else '否'}")
        print(f"总运行时间: {total_time:.2f} 秒")
        print("="*60)
        
        # 保存结果到文件
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存选中站点信息
        import pandas as pd
        selected_stations_info = []
        for i, station_idx in enumerate(selected_original_stations):
            station_info = {
                '序号': i + 1,
                '原始索引': station_idx,
                'FID': original_data['station_fids'][station_idx],
                '高度(m)': original_data['station_heights'][station_idx],
                '高程(m)': original_data['station_elevations'][station_idx],
                'UTM_X': original_data['station_coords'][station_idx][0],
                'UTM_Y': original_data['station_coords'][station_idx][1],
                '成本(元)': original_data['station_heights'][station_idx] * 5000 + 150000
            }
            selected_stations_info.append(station_info)
        
        df = pd.DataFrame(selected_stations_info)
        df.to_excel(os.path.join(output_dir, "selected_stations.xlsx"), index=False)
        print(f"选中站点信息已保存到: {os.path.join(output_dir, 'selected_stations.xlsx')}")
        
        # 保存性能数据
        performance_info = {
            '总运行时间(秒)': total_time,
            '算法运行时间(秒)': results['performance_data']['total_time'],
            '连通修复时间(秒)': results['performance_data']['repair']['repair_time'],
            '冗余剪枝时间(秒)': results['performance_data']['prune']['prune_time'],
            '最终选中站点数': len(selected_original_stations),
            '总建设成本(元)': total_cost,
            '最终覆盖率': results['performance_data']['prune']['final_coverage'],
            '网络连通性': '是' if results['performance_data']['prune']['is_connected'] else '否',
            '总迭代次数': len(results['performance_data']['iteration']['iterations'])
        }
        
        performance_df = pd.DataFrame([performance_info])
        performance_df.to_excel(os.path.join(output_dir, "performance_summary.xlsx"), index=False)
        print(f"性能摘要已保存到: {os.path.join(output_dir, 'performance_summary.xlsx')}")
        
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()