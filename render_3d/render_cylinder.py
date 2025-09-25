import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import pyrender
import trimesh


def overlay_image_onto_background(image_rgb, alpha_mask, background_color=(0.0, 0.0, 0.0)):
    """
    将渲染得到的图像（RGB部分）根据其alpha_mask叠加到指定的背景色上。
    image_rgb: torch.Tensor, 形状为 (H, W, 3) 的渲染前景RGB图像 (0-1范围)。
    alpha_mask: torch.Tensor, 形状为 (H, W) 的透明度掩码 (0-1范围，或布尔值)。
    background_color: tuple, RGB元组，表示背景颜色 (0-1范围)。
    """
    H, W, _ = image_rgb.shape

    # 修改此处：创建背景张量
    background = torch.tensor(background_color, device=image_rgb.device, dtype=image_rgb.dtype).view(1, 1, 3).expand(H, W, 3)

    # 确保 alpha_mask 是 float 类型，方便乘法运算
    if alpha_mask.dtype == torch.bool:
        alpha_mask = alpha_mask.float()

    # 使用 alpha_mask 进行线性插值叠加
    # out_pixel = alpha * foreground_pixel + (1 - alpha) * background_pixel
    out_image = image_rgb * alpha_mask.unsqueeze(-1) + background * (1 - alpha_mask.unsqueeze(-1))
    return out_image

def render_colored_cylinders(cylinder_specs, image_size=(1280, 1280), scene=None):    
    H, W = image_size
    points_to_draw = []
    added_nodes = []

    for start, end, color in cylinder_specs:
        start = np.array(start)
        end = np.array(end)
        vec = end - start
        height = np.linalg.norm(vec)
        if height == 0:
            continue

        tm = trimesh.creation.cylinder(radius=12, height=height, sections=32)

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
        node = scene.add(mesh)
        added_nodes.append(node)

        # # [Optional]: 投影点用于可视化，debug投射是否正确
        # x1 = fx * (start[0] / start[2]) + cx
        # y1 = fy * (start[1] / start[2]) + cy
        # x2 = fx * (end[0] / end[2]) + cx
        # y2 = fy * (end[1] / end[2]) + cy
        # points_to_draw.append((x1, y1))
        # points_to_draw.append((x2, y2))

    # 渲染
    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H, point_size=1.0)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)

    # 后处理
    color = color.astype(np.float32) / 255.0
    # 转 uint8
    final_img = (color * 255).astype(np.uint8)

    # 渲染完成后删除添加的mesh
    for node in added_nodes:
        scene.remove_node(node)

    # # [Optional]: 2D画点，debug投射是否正确
    # for (x, y) in points_to_draw:
    #     x_draw = int(x)
    #     y_draw = int(y)
    #     cv2.circle(final_img, (x_draw, y_draw), radius=4, color=(0, 255, 0), thickness=-1)

    return Image.fromarray(final_img)


def main():
    """
    Main function to set up the scene, define test cylinders,
    render them, and save the output.
    """
    IMG_SIZE = (1280, 1280)

    # 1. Create the scene
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.3, 0.3, 0.3])

    # 2. Add a camera
    # ================================================================= #
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
    #         THIS IS THE CORRECTED LINE                              #
    # We must specify znear and zfar to ensure objects are visible      #
    camera = pyrender.OrthographicCamera(xmag=300, ymag=300, znear=0.1, zfar=1000)
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ #
    # ================================================================= #

    # Position the camera to look at the origin from a distance
    camera_pose = np.array([
       [1.0, 0.0, 0.0, 0.0],
       [0.0, 1.0, 0.0, 0.0],
       [0.0, 0.0, 1.0, 500.0], # Move camera back along Z-axis
       [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)

    # 3. Add a light source
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=camera_pose) # Add light from the camera's position

    # 4. Define the test cylinders (a simple coordinate frame)
    # X-axis in Red, Y-axis in Green, Z-axis in Blue
    cylinder_specs = [
        # ((start_x, start_y, start_z), (end_x, end_y, end_z), (R, G, B, Alpha))
        ((0, 0, 0), (100, 0, 0), (1.0, 0.0, 0.0, 1.0)), # Red cylinder along X
        ((0, 0, 0), (0, 100, 0), (0.0, 1.0, 0.0, 1.0)), # Green cylinder along Y
        ((0, 0, 0), (0, 0, 100), (0.0, 0.0, 1.0, 1.0)), # Blue cylinder along Z
    ]

    # 5. Render the image
    print("Rendering scene...")
    rendered_image = render_colored_cylinders(
        cylinder_specs=cylinder_specs,
        image_size=IMG_SIZE,
        scene=scene
    )

    # 6. Save the output
    output_path = "test.jpg"
    rendered_image.save(output_path)
    print(f"Image saved successfully to {output_path}")


if __name__ == "__main__":
    main()