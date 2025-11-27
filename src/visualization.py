import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from tqdm import tqdm
import time
import os
import cv2
import matplotlib.animation as animation
from PIL import Image
from scipy.spatial import KDTree
from .utils import load_city_boundary, load_elevation_data, generate_elevation_heatmap

# 3D可视化相关导入
try:
    from vispy import scene, app
    from vispy.visuals import transforms
    from vispy.color import Colormap
    import psutil
    VISPY_AVAILABLE = True
except ImportError:
    VISPY_AVAILABLE = False
    print("警告: vispy 不可用，3D可视化功能将禁用")

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
                img = Image.open(temp_filename)
                # 转换为RGB
                img = img.convert('RGB')
                # 转换为numpy数组
                img_array = np.array(img)
                
                self.animation_frames.append(img_array)
                
                # 删除临时文件
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
        from scipy.spatial import KDTree
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

# 3D可视化配置
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

if VISPY_AVAILABLE:
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
            from scipy.spatial import KDTree
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


# 2D可视化函数
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
    elevation_data = load_elevation_data("等高线.geojson")
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
    elevation_data = load_elevation_data("等高线.geojson")
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
    elevation_data = load_elevation_data("等高线.geojson")
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
    elevation_data = load_elevation_data("等高线.geojson")
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
    elevation_data = load_elevation_data("等高线.geojson")
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
    import networkx as nx
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
        ax1_left.axhline(y=0.95, color='r', linestyle='--', 
                        linewidth=2, label=f'目标覆盖率 (95.0%)')
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
        '目标覆盖率': '95.00%',
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

def create_3d_visualization_with_original_data(original_data, selected_original_stations):
    """
    3D可视化 - 使用原始数据
    """
    if not VISPY_AVAILABLE:
        print("❌ vispy 不可用，跳过3D可视化")
        return
    
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