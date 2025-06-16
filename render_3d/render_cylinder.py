import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


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

# def render_colored_cylinders(cylinder_specs, focal, princpt, image_size=(1280, 1280)):
#     all_verts = []
#     all_faces = []
#     all_colors = []
#     vert_offset = 0
#     # 相机设置
#     fx, fy = focal
#     px, py = princpt
#     H, W = image_size

#     points_to_draw = []
#     for (start, end, color) in cylinder_specs:
#         x_in2d_start = fx * (start[0] / start[2]) + px
#         y_in2d_start = fy * (start[1] / start[2]) + py
#         x_in2d_end = fx * (end[0] / end[2]) + px
#         y_in2d_end = fy * (end[1] / end[2]) + py

#         points_to_draw.append((x_in2d_start, y_in2d_start))
#         points_to_draw.append((x_in2d_end, y_in2d_end))


#         start = np.array(start)
#         end = np.array(end)
#         vec = end - start
#         height = np.linalg.norm(vec)
#         if height == 0:
#             continue

#         tm = trimesh.creation.cylinder(radius=0.01, height=height, sections=32)

#         z_axis = np.array([0, 0, 1])
#         axis = np.cross(z_axis, vec)
#         if np.linalg.norm(axis) > 1e-6:
#             axis = axis / np.linalg.norm(axis)
#             angle = np.arccos(np.dot(z_axis, vec) / height)
#             rot = trimesh.transformations.rotation_matrix(angle, axis)
#             tm.apply_transform(rot)

#         tm.apply_translation(start + vec / 2)

#         verts = torch.tensor(tm.vertices, dtype=torch.float32)
#         faces = torch.tensor(tm.faces, dtype=torch.int64) + vert_offset
#         verts_rgb = torch.tensor(color, dtype=torch.float32).view(1, 3).expand(verts.shape[0], -1)

#         all_verts.append(verts)
#         all_faces.append(faces)
#         all_colors.append(verts_rgb)
#         vert_offset += verts.shape[0]

#     verts = torch.cat(all_verts, dim=0).to(device)
#     faces = torch.cat(all_faces, dim=0).to(device)
#     verts_rgb = torch.cat(all_colors, dim=0).to(device)
#     textures = TexturesVertex(verts_features=[verts_rgb])
#     mesh = Meshes(verts=[verts], faces=[faces], textures=textures)



#     cameras = PerspectiveCameras(
#         device=device,
#         focal_length=((fx, fy),),
#         principal_point=((px, py),),
#         image_size=((H, W),),
#         in_ndc=False
#     )

#     lights = PointLights(device=device, location=[[2.0, 2.0, 2.0]])
#     raster_settings = RasterizationSettings(image_size=(H, W), blur_radius=0, faces_per_pixel=2)
#     renderer = MeshRenderer(
#         rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
#         shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
#     )

#     images = renderer(mesh)
#     image_rgb = images[0, ..., :3]
#     alpha_mask = images[0, ..., 3]

#     final_image = overlay_image_onto_background(image_rgb, alpha_mask, background_color=(0.0, 0.0, 0.0))
#     final_image_np = (final_image.cpu().numpy() * 255).astype(np.uint8)
#     # 翻转图像
#     final_image_np = cv2.flip(final_image_np, -1)
#     green_color = (0, 255, 0) # Green in RGB (R, G, B)
#     radius = 2
#     thickness = -1 # Fills the circle

#     # Draw circles for each stored point
#     for (point_x, point_y) in points_to_draw:
#         # If you flipped the image, the coordinates might also need to be adjusted
#         # For a horizontal and vertical flip (-1), (x,y) becomes (W-1-x, H-1-y)
#         # However, `cv2.flip` modifies the array in place, so subsequent operations
#         # like `cv2.circle` should operate on the *new* coordinate system.
#         # So we draw on the already flipped image.
#         cv2.circle(final_image_np, (int(point_x), int(point_y)), radius, green_color, thickness)


#     final_image_pil = Image.fromarray(final_image_np)

#     return final_image_pil


def render_colored_cylinders(cylinder_specs, focal, princpt, image_size=(1280, 1280), img=None):
    import pyrender
    import trimesh
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    H, W = image_size
    fx, fy = focal
    cx, cy = princpt

    # 初始化场景
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3])

    # 设置相机
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
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

        tm = trimesh.creation.cylinder(radius=0.01, height=height, sections=32)

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
        rgba = np.array(color + (0.8,))
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.5,
            baseColorFactor=rgba
        )

        mesh = pyrender.Mesh.from_trimesh(tm, material=material)
        scene.add(mesh)

        # 投影点用于可视化
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
    final_img = cv2.add(final_img, img)

    # 画点
    for (x, y) in points_to_draw:
        x_draw = int(x)
        y_draw = int(y)
        cv2.circle(final_img, (x_draw, y_draw), radius=4, color=(0, 255, 0), thickness=-1)

    return Image.fromarray(final_img)
