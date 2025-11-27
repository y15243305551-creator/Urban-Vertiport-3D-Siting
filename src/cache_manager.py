import os
import pickle
import hashlib
import json
import shutil
import numpy as np

class CacheManager:
    """缓存管理类"""
    
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, name, data_hash):
        return os.path.join(self.cache_dir, f"{name}_{data_hash}.pkl")
    
    def _compute_data_hash(self):
        """原有的数据哈希计算方法"""
        # 所有影响可达性计算的参数
        reachability_params = {
            # 物理参数
            'g': 9.81,
            'v_w': -10,
            'b': 200,
            'rho': 1.225,
            
            # 无人机参数
            'm_d': 32,
            'm_p_max': 10,
            'v_u': 4,
            'v_d': 3,
            'v_h': 14,
            'A': 1.7,
            'Ah': 1,
            'r': 0.47,
            'Ap': 3.141592653589793 * 0.47**2,
            'N': 8,
            'E_total': 7668320,
            
            # 系数
            'eta': 0.85,
            'Cd': 0.7,
            'Ct': 0.04,
            'electricity_consumption_rate': 0.8,
            
            # 模型参数
            'cover_K': 1000,
            'relay_K': 1000,
            'MAX_OBSTACLE_COUNT': 10,
            'elevation_min': -41,
        }
        
        # 数据统计信息
        data_stats = {
            'district': '全部区',
            'station_count': 0,
            'demand_count': 0,
            'building_count': 0,
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
            'MAX_OBSTACLE_COUNT': 10,
            'MAX_OBSTACLE_THRESHOLD': 10,
            'electricity_consumption_rate': 0.8,
            'b': 200,
            'E_total': 7668320
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
                if fname.startswith("reverse_index_") or fname.startswith("connectivity_"):
                    cache_path = os.path.join(self.cache_dir, fname)
                    if os.path.isfile(cache_path):
                        mtime = os.path.getmtime(cache_path)
                        cache_files.append((mtime, cache_path))
            
            # 按修改时间排序，保留最新的
            cache_files.sort(reverse=True)
            for mtime, cache_path in cache_files[keep_recent:]:
                print(f"清理旧缓存: {os.path.basename(cache_path)}")
                os.remove(cache_path)
                
        except Exception as e:
            print(f"清理缓存时出错: {e}")
    
    def clear_all_caches(self):
        """清理所有缓存"""
        try:
            for fname in os.listdir(self.cache_dir):
                cache_path = os.path.join(self.cache_dir, fname)
                if os.path.isfile(cache_path) and (fname.startswith("reverse_index_") or fname.startswith("connectivity_")):
                    os.remove(cache_path)
            print("✅ 所有缓存已清理")
        except Exception as e:
            print(f"清理缓存时出错: {e}")