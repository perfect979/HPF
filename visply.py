import open3d as o3d

# 1. 读取 PLY 文件
file_path = "your_model.ply"  # 替换为你的 .ply 文件路径
mesh = o3d.io.read_triangle_mesh(file_path)

# 2. 检查文件是否成功加载
if not mesh.has_triangles():
    print("Error: The file does not contain a valid 3D mesh.")
else:
    print("Mesh successfully loaded!")
    print(mesh)

# 3. 计算法线（如果需要更好的视觉效果）
mesh.compute_vertex_normals()

# 4. 创建可视化窗口并移除背景网格线
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="3D Model Viewer", width=800, height=600)
vis.add_geometry(mesh)

# 取消背景网格线
render_option = vis.get_render_option()
render_option.show_coordinate_frame = False  # 不显示坐标系
render_option.mesh_show_back_face = True     # 显示背面
render_option.background_color = [0, 0, 0]   # 设置背景为黑色 (RGB)

# 渲染
vis.run()
vis.destroy_window()
