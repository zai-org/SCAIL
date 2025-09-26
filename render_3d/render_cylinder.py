import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


def render_colored_cylinders(cylinder_specs, focal, princpt, image_size=(1280, 1280), img=None):
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    import pyrender
    import trimesh
    H, W = image_size
    if isinstance(focal, float) or isinstance(focal, int):
        fx, fy = focal, focal
    else:
        fx, fy = focal[0], focal[1]
    cx, cy = princpt

    # 初始化场景
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.1, 0.1, 0.1])

    # 设置相机
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.5, zfar=10000)
    pyrender2opencv = np.array([[1.0, 0, 0, 0],
                                 [0, -1, 0, 0],
                                 [0, 0, -1, 0],
                                 [0, 0, 0, 1]])
    cam_pose = pyrender2opencv @ np.eye(4)
    scene.add(camera, pose=cam_pose)

    # 添加光源
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=cam_pose)

    points_to_draw = []

    for start, end, color in cylinder_specs:
        start = np.array(start)
        end = np.array(end)
        vec = end - start
        height = np.linalg.norm(vec)
        if height == 0:
            continue

        tm = trimesh.creation.cylinder(radius=12, height=height, sections=16)

        # 旋转对齐z轴
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, vec)
        if np.linalg.norm(axis) > 1e-6:
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.dot(z_axis, vec) / height)
            rot = trimesh.transformations.rotation_matrix(angle, axis)
            tm.apply_transform(rot)

        tm.apply_translation(start + vec / 2)

        # 材质颜色（支持 RGBA）
        rgba = np.array(color)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.1,
            roughnessFactor=0.5,
            baseColorFactor=rgba
        )

        mesh = pyrender.Mesh.from_trimesh(tm, material=material)
        scene.add(mesh)

        # 投影点用于可视化，检查投射是否正确
        x1 = fx * (start[0] / start[2]) + cx
        y1 = fy * (start[1] / start[2]) + cy
        x2 = fx * (end[0] / end[2]) + cx
        y2 = fy * (end[1] / end[2]) + cy
        points_to_draw.append((x1, y1))
        points_to_draw.append((x2, y2))

    # 渲染
    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H, point_size=1.0)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)

    # 后处理
    color = color.astype(np.float32) / 255.0
    # 转 uint8
    final_img = (color * 255).astype(np.uint8)

    # 画点，检查投射是否正确
    for (x, y) in points_to_draw:
        print(f" debug point: {x}, {y}")
        x_draw = int(x)
        y_draw = int(y)
        cv2.circle(final_img, (x_draw, y_draw), radius=4, color=(0, 255, 0), thickness=-1)

    return Image.fromarray(final_img)


# test
if __name__ == "__main__":
    # 构造一个空白背景
    H, W = 480, 640
    img = np.zeros((H, W, 3), dtype=np.uint8) + 255  # 白色背景

    # 构造几组3D点对和颜色
    cylinder_specs = [
        # 起点 (0,0,100), Y轴方向的红色圆柱终点 Y 调整为 40
        (np.array([0, 20, 120]), np.array([0, 40, 100]), [1.0, 0.0, 0.0, 1.0]),  # 红色

        # 起点 (0,0,100), X轴方向的绿色圆柱终点 X 调整为 60
        (np.array([0, 0, 100]), np.array([60, 40, 100]), [0.0, 1.0, 0.0, 1.0]),  # 绿色

        # Z轴方向的蓝色圆柱长度调整为50 (从100到150)
        (np.array([0, 0, 100]), np.array([0, 0, 150]), [0.0, 0.0, 1.0, 1.0]),  # 蓝色
    ]

    # 简单的相机参数
    fx, fy = 500, 500
    cx, cy = W // 2, H // 2

    # 调用渲染函数
    img_pil = render_colored_cylinders(
        cylinder_specs=cylinder_specs,
        focal=(fx, fy),
        princpt=(cx, cy),
        image_size=(H, W),
        img=img
    )

    # 显示或保存结果
    img_pil.save("test_render_cylinder.png")
