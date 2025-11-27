import numpy as np
import math
from scipy.spatial import KDTree
from tqdm import tqdm
import time

class ReachabilityCalculator:
    """可达性计算类"""
    
    def __init__(self, physical_params, drone_params, model_params):
        """
        初始化可达性计算器
        
        Args:
            physical_params: 物理参数字典
            drone_params: 无人机参数字典  
            model_params: 模型参数字典
        """
        # 物理参数
        self.pi = physical_params.get('pi', math.pi)
        self.g = physical_params.get('g', 9.81)
        self.v_w = physical_params.get('v_w', -10)
        self.b = physical_params.get('b', 200)
        self.rho = physical_params.get('rho', 1.225)
        
        # 无人机参数
        self.m_d = drone_params.get('mass', 32)
        self.m_p_max = drone_params.get('max_payload', 10)
        self.v_u = drone_params.get('takeoff_speed', 4)
        self.v_d = drone_params.get('landing_speed', 3)
        self.v_h = drone_params.get('cruise_speed', 14)
        self.A = drone_params.get('cross_section', 1.7)
        self.Ah = drone_params.get('cruise_cross_section', 1)
        self.r = drone_params.get('propeller_radius', 0.47)
        self.Ap = self.pi * self.r**2
        self.N = drone_params.get('propeller_count', 8)
        self.E_total = drone_params.get('battery_energy', 7668320)
        
        # 系数
        self.eta = drone_params.get('efficiency', 0.85)
        self.Cd = drone_params.get('drag_coefficient', 0.7)
        self.Ct = drone_params.get('thrust_coefficient', 0.04)
        self.electricity_consumption_rate = drone_params.get('electricity_consumption_rate', 0.8)
        
        # 模型参数
        self.cover_K = model_params.get('cover_K', 1000)
        self.relay_K = model_params.get('relay_K', 1000)
        self.MAX_OBSTACLE_COUNT = model_params.get('max_obstacle_count', 10)
        self.MAX_OBSTACLE_THRESHOLD = model_params.get('max_obstacle_threshold', 10)
        self.elevation_min = model_params.get('elevation_min', -41)
        
        # 高建筑KDTree（将在外部设置）
        self.high_building_kdtree = None
        self.high_building_indices = None
    
    def set_high_building_kdtree(self, high_building_kdtree, high_building_indices):
        """设置高建筑KDTree"""
        self.high_building_kdtree = high_building_kdtree
        self.high_building_indices = high_building_indices
    
    def calculate_power(self, omega):
        """计算功率（使用已经缩放好的功率公式）"""
        return 2.33e-3 * omega**3 + 1.528e-2 * omega**2 + 1.027e-2 * omega + 64.63

    def calculate_omega_u(self, l_i):
        """计算起飞角速度（修正公式）"""
        numerator = ((2 * (self.m_d + self.m_p_max) * self.g + self.rho * self.A * self.Cd * self.v_u**2)**2 + (self.rho * self.A * self.Cd * self.v_w**2)**2)**(1/2)
        denominator = self.N * self.rho * self.Ap * self.Ct * self.r**2
        return math.sqrt(numerator / denominator)

    def calculate_omega_d(self, l_i):
        """计算降落角速度（修正公式）"""
        numerator = ((2 * (self.m_d + self.m_p_max) * self.g - self.rho * self.A * self.Cd * self.v_d**2)**2 + (self.rho * self.A * self.Cd * self.v_w**2)**2)**(1/2)
        denominator = self.N * self.rho * self.Ap * self.Ct * self.r**2
        return math.sqrt(numerator / denominator)

    def calculate_omega_h(self):
        """计算平飞角速度（修正公式）"""
        numerator = (4 * (self.m_d + self.m_p_max)**2 * self.g**2 + self.rho**2 * self.Ah**2 * self.Cd**2 * (self.v_h - self.v_w)**4)**(1/4)
        denominator = math.sqrt(self.N * self.rho * self.Ap * self.Ct * self.r**2)
        return numerator / denominator

    def calculate_vertical_energy(self, l_i_start, l_i_end, task_type='cover'):
        """计算垂直起降能耗 - 区分起点和终点高度"""
        omega_u = self.calculate_omega_u(max(l_i_start, l_i_end))
        omega_d = self.calculate_omega_d(max(l_i_start, l_i_end))
        
        P_u = self.calculate_power(omega_u)
        P_d = self.calculate_power(omega_d)
        
        if task_type == 'cover':
            # 普通任务：起点起飞 + 终点降落 + 终点起飞 + 起点降落
            return (P_u * l_i_start / self.v_u + P_d * l_i_end / self.v_d + 
                    P_u * l_i_end / self.v_u + P_d * l_i_start / self.v_d)
        else:  # relay
            # 接力任务：起点起飞 + 终点降落
            return P_u * l_i_start / self.v_u + P_d * l_i_end / self.v_d

    def calculate_horizontal_energy(self, distance, task_type='cover'):
        """计算水平飞行能耗"""
        omega_h = self.calculate_omega_h()
        P_h = self.calculate_power(omega_h)
        
        if task_type == 'cover':
            # 普通任务：往返距离
            return P_h * 2 * distance / self.v_h
        else:  # relay
            # 接力任务：单程距离
            return P_h * distance / self.v_h

    def is_point_on_segment(self, p, a, b):
        """判断点p是否在线段ab上"""
        # 使用向量叉积和点积判断
        cross = np.cross(b - a, p - a)
        if abs(cross) > 1e-10:  # 不在直线上
            return False
        
        # 在线段上
        dot1 = np.dot(p - a, b - a)
        dot2 = np.dot(p - b, a - b)
        return dot1 >= 0 and dot2 >= 0

    def find_obstacles_on_path(self, start_coord, end_coord, building_coords, building_heights, building_elevations, building_areas):
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
        if self.high_building_kdtree is None:
            return obstacles
        
        # 路径方向单位向量
        path_dir = path_vector / path_length
        
        # 查询路径附近的潜在障碍物
        # 计算路径的边界框，扩大搜索范围确保覆盖所有可能障碍物
        path_center = (start_coord + end_coord) / 2
        search_radius = path_length / 2 + 50  # 路径长度一半+50米缓冲
        
        # 使用KDTree快速找到路径附近的高建筑
        nearby_indices = self.high_building_kdtree.query_ball_point(path_center, search_radius)
        
        # 对这些潜在障碍物进行精确几何计算
        for kd_index in nearby_indices:
            # 获取原始建筑索引
            original_idx = self.high_building_indices[kd_index]
            
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

    def calculate_detour_energy(self, obstacles, start_coord, end_coord, task_type='cover'):
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
        omega_h = self.calculate_omega_h()
        P_h = self.calculate_power(omega_h)
        energy_per_meter = P_h / self.v_h
        
        if task_type == 'cover':
            # 普通任务：往返都要绕行
            return 2 * extra_distance * energy_per_meter
        else:
            # 接力任务：单程绕行
            return extra_distance * energy_per_meter

    def calculate_climb_energy(self, obstacles, start_coord, end_coord, task_type='cover'):
        """计算爬升能耗 - 对每个障碍物单独计算"""
        if not obstacles:
            return 0
        
        climb_energy = 0
        omega_u = self.calculate_omega_u(0)  # 使用0作为基础高度，实际高度在循环中计算
        omega_d = self.calculate_omega_d(0)
        P_u = self.calculate_power(omega_u)
        P_d = self.calculate_power(omega_d)
        
        for obstacle in obstacles:
            # 需要爬升到障碍物上方10米
            required_height = obstacle['height'] + 10
            
            # 计算额外爬升高度
            extra_climb_height = required_height - self.b
            
            if extra_climb_height > 0:
                # 计算单个障碍物的爬升能耗
                single_climb_energy = P_u * extra_climb_height / self.v_u + P_d * extra_climb_height / self.v_d
                
                if task_type == 'cover':
                    # 普通任务：往返都要爬升
                    climb_energy += 2 * single_climb_energy
                else:
                    # 接力任务：单程爬升
                    climb_energy += single_climb_energy
        
        return climb_energy

    def calculate_obstacle_energy(self, obstacles, start_coord, end_coord, task_type='cover'):
        """计算避障能耗 - 选择绕行或爬升中能耗较小的策略"""
        if not obstacles:
            return 0, 0, 'none'
        
        detour_energy = self.calculate_detour_energy(obstacles, start_coord, end_coord, task_type)
        climb_energy = self.calculate_climb_energy(obstacles, start_coord, end_coord, task_type)
        
        # 选择能耗较小的策略
        if detour_energy < climb_energy:
            return detour_energy, len(obstacles), 'detour'
        else:
            return climb_energy, len(obstacles), 'climb'

    def calculate_reachability(self, station_idx, target_coord, task_type,
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
        l_i_start = max(self.b - start_total_height, 0)
        l_i_end = max(self.b - end_total_height, 0)

        # 首先检测路径上的所有障碍物
        t1 = time.time()
        all_obstacles = self.find_obstacles_on_path(station_coord, target_coord,
                                                  building_coords, building_heights,
                                                  building_elevations, building_areas)
        total_obstacle_count = len(all_obstacles)
        t2 = time.time()

        # 障碍物数量阈值检查
        if total_obstacle_count > self.MAX_OBSTACLE_THRESHOLD:
            return False, float('inf'), total_obstacle_count, 'exceed_threshold'

        # 如果障碍物数量在阈值内，继续计算能耗
        # 只取前MAX_OBSTACLE_COUNT个障碍物用于能耗计算
        obstacles_for_calculation = all_obstacles[:self.MAX_OBSTACLE_COUNT]
        obstacle_count_for_calculation = len(obstacles_for_calculation)

        # 垂直能耗
        t3 = time.time()
        vertical_energy = self.calculate_vertical_energy(l_i_start, l_i_end, task_type)
        t4 = time.time()

        # 水平能耗
        distance = np.linalg.norm(target_coord - station_coord)
        horizontal_energy = self.calculate_horizontal_energy(distance, task_type)
        t5 = time.time()

        # 避障能耗（只计算前MAX_OBSTACLE_COUNT个障碍物）
        obstacle_energy, _, strategy = self.calculate_obstacle_energy(
            obstacles_for_calculation, station_coord, target_coord, task_type)
        t6 = time.time()

        total_energy = vertical_energy + horizontal_energy + obstacle_energy
        reachable = (total_energy <= self.E_total * self.electricity_consumption_rate) and (obstacle_count_for_calculation <= self.MAX_OBSTACLE_COUNT)

        return reachable, total_energy, total_obstacle_count, strategy

    def calculate_max_service_radius(self, station_idx, station_heights, station_elevations, task_type):
        """计算最大服务半径 - 使用论文中的公式"""
        station_height = station_heights[station_idx]
        station_elevation = station_elevations[station_idx]
        total_height = station_height - station_elevation
        l_i = max(self.b - total_height, 10)
        
        # 角速度计算
        ω_u = self.calculate_omega_u(l_i)
        ω_d = self.calculate_omega_d(l_i)
        ω_h = self.calculate_omega_h()
        
        # 功率计算
        P_u = self.calculate_power(ω_u)
        P_d = self.calculate_power(ω_d)
        P_h = self.calculate_power(ω_h)
        
        if task_type == 'cover':
            # 普通任务覆盖半径 r_i
            r_i = self.v_h * (self.E_total * self.electricity_consumption_rate - 
                         (P_u * l_i / self.v_u + P_u * (self.b - self.elevation_min) / self.v_u + 
                          P_d * (self.b - self.elevation_min) / self.v_d + P_d * l_i / self.v_d)) / (2 * P_h)
            return max(r_i, 0)
        else:  # relay
            # 接力任务覆盖半径 R_i
            R_i = self.v_h * (self.E_total * self.electricity_consumption_rate - 
                         (P_u * l_i / self.v_u + P_d * (self.b - self.elevation_min) / self.v_d)) / P_h
            return max(R_i, 0)

    def compute_cover_radii(self, station_coords, station_heights, station_elevations):
        """计算所有站点的覆盖半径"""
        print("计算最大服务半径用于预筛选...")
        cover_radii = np.array([self.calculate_max_service_radius(i, station_heights, station_elevations, 'cover') 
                              for i in tqdm(range(len(station_coords)), desc="计算覆盖半径")])
        return cover_radii

    def compute_relay_radii(self, station_coords, station_heights, station_elevations):
        """计算所有站点的接力半径"""
        relay_radii = np.array([self.calculate_max_service_radius(i, station_heights, station_elevations, 'relay') 
                              for i in tqdm(range(len(station_coords)), desc="计算接力半径")])
        return relay_radii