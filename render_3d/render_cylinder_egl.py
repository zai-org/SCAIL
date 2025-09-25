import os
import sys
import types

os.environ["PYOPENGL_PLATFORM"] = "egl"
# To regard pyrender.viewer
fake_viewer = types.ModuleType("pyrender.viewer")
class DummyViewer:
    def __init__(self, *a, **kw):
        raise RuntimeError("Viewer is disabled in headless EGL/OSMesa mode")
fake_viewer.Viewer = DummyViewer
sys.modules["pyrender.viewer"] = fake_viewer

import torch
import numpy as np
from PIL import Image
import pyrender
import trimesh

def try_create_renderer(W, H):
    from pyrender import OffscreenRenderer
    try:
        print("🔹 Try EGL...")
        r = OffscreenRenderer(viewport_width=W, viewport_height=H, point_size=1.0)
        return r
    except Exception as e:
        print("❌ EGL fails:", e)

def render_colored_cylinders(cylinder_specs, image_size=(1280, 1280), scene=None):
    H, W = image_size
    added_nodes = []

    for start, end, color in cylinder_specs:
        start = np.array(start)
        end = np.array(end)
        vec = end - start
        height = np.linalg.norm(vec)
        if height == 0:
            continue

        tm = trimesh.creation.cylinder(radius=12, height=height, sections=32)

        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, vec)
        if np.linalg.norm(axis) > 1e-6:
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.dot(z_axis, vec) / height)
            rot = trimesh.transformations.rotation_matrix(angle, axis)
            tm.apply_transform(rot)

        tm.apply_translation(start + vec / 2)

        rgba = np.array(color)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.1,
            roughnessFactor=0.5,
            baseColorFactor=rgba
        )

        mesh = pyrender.Mesh.from_trimesh(tm, material=material)
        node = scene.add(mesh)
        added_nodes.append(node)

    r = try_create_renderer(W, H)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)

    color = color.astype(np.float32) / 255.0
    final_img = (color * 255).astype(np.uint8)

    for node in added_nodes:
        scene.remove_node(node)

    return Image.fromarray(final_img)

def main():
    IMG_SIZE = (1280, 1280)

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                           ambient_light=[0.3, 0.3, 0.3])

    camera = pyrender.OrthographicCamera(xmag=300, ymag=300, znear=0.1, zfar=1000)
    camera_pose = np.array([
       [1.0, 0.0, 0.0, 0.0],
       [0.0, 1.0, 0.0, 0.0],
       [0.0, 0.0, 1.0, 500.0],
       [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=camera_pose)

    cylinder_specs = [
        ((0, 0, 0), (100, 0, 0), (1.0, 0.0, 0.0, 1.0)),
        ((0, 0, 0), (0, 100, 0), (0.0, 1.0, 0.0, 1.0)),
        ((0, 0, 0), (0, 0, 100), (0.0, 0.0, 1.0, 1.0)),
    ]

    print("Rendering scene...")
    rendered_image = render_colored_cylinders(
        cylinder_specs=cylinder_specs,
        image_size=IMG_SIZE,
        scene=scene
    )

    # RGBA
    rendered_image.save("test_egl.png")
    # RGB
    rendered_image.convert("RGB").save("test_egl.jpg")

    print("✅ finish rendering")


if __name__ == "__main__":
    main()