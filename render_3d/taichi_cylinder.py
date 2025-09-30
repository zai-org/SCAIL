import taichi as ti
import numpy as np
from PIL import Image
import random
import math
import time
import imageio
from pose_draw.draw_pose_main import save_videos_from_pil
ti.init(arch=ti.cuda)

def flatten_specs(specs_list):
    """把 specs_list 拉平为 numpy 数组 + 索引表"""
    starts, ends, colors = [], [], []
    frame_offset, frame_count = [], []
    offset = 0
    for specs in specs_list:
        frame_offset.append(offset)
        frame_count.append(len(specs))
        for (s, e, c) in specs:
            starts.append(s)
            ends.append(e)
            colors.append(c)
        offset += len(specs)
    return (
        np.array(starts, dtype=np.float32),
        np.array(ends, dtype=np.float32),
        np.array(colors, dtype=np.float32),
        np.array(frame_offset, dtype=np.int32),
        np.array(frame_count, dtype=np.int32),
    )

def render_whole(specs_list, H=480, W=640, fx=500, fy=500, cx=240,  cy=320, output_path=None):
    img = ti.Vector.field(4, dtype=ti.f32, shape=(H, W))
    starts, ends, colors, frame_offset, frame_count = flatten_specs(specs_list)
    total_cyl = len(starts)
    n_frames = len(specs_list)
    z_min = min(starts[:, 2].min(), ends[:, 2].min())
    z_max = max(starts[:, 2].max(), ends[:, 2].max())

    # ========= 相机内参 =========
    znear = 0.1
    zfar = max(min(z_max, 25000), 10000)
    C = ti.Vector([0.0, 0.0, 0.0])  # 相机中心
    light_dir = ti.Vector([0.0, 0.0, 1.0])

    c_start = ti.Vector.field(3, dtype=ti.f32, shape=total_cyl)
    c_end   = ti.Vector.field(3, dtype=ti.f32, shape=total_cyl)
    c_rgba  = ti.Vector.field(4, dtype=ti.f32, shape=total_cyl)
    n_cyl   = ti.field(dtype=ti.i32, shape=())  # 实际数量
    f_offset = ti.field(dtype=ti.i32, shape=n_frames)
    f_count  = ti.field(dtype=ti.i32, shape=n_frames)
    frame_id = ti.field(dtype=ti.i32, shape=())  # 当前帧号
    z_min_field = ti.field(dtype=ti.f32, shape=())
    z_max_field = ti.field(dtype=ti.f32, shape=())

    z_min_field[None] = z_min
    z_max_field[None] = z_max

    # # ====== 拷贝数据一次 ======
    c_start.from_numpy(starts)
    c_end.from_numpy(ends)
    c_rgba.from_numpy(colors)
    f_offset.from_numpy(frame_offset)
    f_count.from_numpy(frame_count)

    @ti.func
    def sd_cylinder(p, a, b, r):
        pa = p - a
        ba = b - a
        h = ba.norm()
        eps = 1e-8
        res = 0.0
        if h < eps:
            res = pa.norm() - r
        else:
            ba_n = ba / h
            proj = pa.dot(ba_n)
            proj_clamped = min(max(proj, 0.0), h)
            res = (pa - proj_clamped * ba_n).norm() - r
        return res

    @ti.func
    def scene_sdf(p):
        best_d = 1e6
        best_col = ti.Vector([0.0, 0.0, 0.0, 0.0])
        fid = frame_id[None]  # 从 field 里读出来，变成一个普通 int
        off = f_offset[fid]
        cnt = f_count[fid]
        for i in range(cnt):  # 只遍历实际数量
            a = c_start[off + i]
            b = c_end[off + i]
            r = 11.0
            col = c_rgba[off + i]
            d = sd_cylinder(p, a, b, r)
            if d < best_d:
                best_d = d
                best_col = col
        return best_d, best_col

    @ti.func
    def get_normal(p):
        e = 1e-3
        dx = scene_sdf(p + ti.Vector([e, 0.0, 0.0]))[0] - scene_sdf(p - ti.Vector([e, 0.0, 0.0]))[0]
        dy = scene_sdf(p + ti.Vector([0.0, e, 0.0]))[0] - scene_sdf(p - ti.Vector([0.0, e, 0.0]))[0]
        dz = scene_sdf(p + ti.Vector([0.0, 0.0, e]))[0] - scene_sdf(p - ti.Vector([0.0, 0.0, e]))[0]
        n = ti.Vector([dx, dy, dz])
        return n.normalized()

    @ti.func
    def pixel_to_ray(xi, yi):
        u = (xi - cx) / fx
        v = (yi - cy) / fy
        dir_cam = ti.Vector([u, v, 1.0]).normalized()
        Rcw = ti.Matrix.identity(ti.f32, 3)
        rd_world = Rcw @ dir_cam
        ro_world = C
        return ro_world, rd_world

    @ti.kernel
    def render():
        depth_near, depth_far = ti.max(z_min_field[None], 0.1), ti.min(z_max_field[None] + 6000, 20000)  # 能渲染出来的点，最大12000
        for y, x in img:
            ro, rd = pixel_to_ray(x, y)
            t = znear
            col_out = ti.Vector([0.0, 0.0, 0.0, 0.0])
            for _ in range(300):
                p = ro + rd * t
                d, col = scene_sdf(p)
                if d < 1e-3:
                #     n = get_normal(p)
                #     diff = max(n.dot(-light_dir), 0.0)
                #     lit = 0.3 + 0.7 * diff
                #     col_out = ti.Vector([col.x * lit, col.y * lit, col.z * lit, col.w])
                #     break

                    n = get_normal(p)
                    diff = max(n.dot(-light_dir), 0.0)

                    # === Blinn-Phong 镜面反射 ===
                    view_dir = -rd.normalized()
                    half_dir = (view_dir + -light_dir).normalized()
                    spec = max(n.dot(half_dir), 0.0) ** 32   # shininess=32，越小越散，越大越锐

                    depth_factor = 1.0 - (p.z - depth_near) / (depth_far - znear)
                    depth_factor = ti.max(0.0, ti.min(1.0, depth_factor))

                    # 原来的 diffuse/ambient 光照
                    diffuse_term = 0.3 + 0.7 * diff
                    base = col.xyz * diffuse_term * depth_factor

                    # 镜面高光（叠加到原有结果上）
                    highlight = ti.Vector([1.0, 1.0, 1.0]) * (0.5 * spec) * depth_factor

                    col_out = ti.Vector([base.x + highlight.x,
                                        base.y + highlight.y,
                                        base.z + highlight.z,
                                        col.w])
                    break

                if t > zfar:
                    break
                t += max(d, 1e-4)
            img[y, x] = col_out

    frames = []
    for f in range(len(specs_list)):
        # start_time = time.time()
        frame_id[None] = f
        render()
        arr = np.clip(img.to_numpy(), 0, 1)
        # end_time = time.time()
        # print(f"Frame {f} time: {end_time - start_time} seconds")
        arr8 = (arr * 255).astype(np.uint8)
        frames.append(Image.fromarray(arr8, mode="RGBA"))

    save_videos_from_pil(frames, output_path, fps=16)


def random_cylinder():
    """生成一根随机圆柱 (start, end, color)。"""
    # 起点 [-200,200]^2, z 在 [-300,-100]
    ax = random.uniform(-200, 200)
    ay = random.uniform(-200, 200)
    az = random.uniform(300, 400)
    start = [ax, ay, az]

    # 随机方向和长度
    theta = random.uniform(0, 2*math.pi)
    phi = random.uniform(-math.pi/4, math.pi/4)  # 倾斜角
    L = 100
    dx = math.cos(phi) * math.cos(theta)
    dy = math.cos(phi) * math.sin(theta)
    dz = math.sin(phi)
    end = [ax + dx * L, ay + dy * L, az + dz * L]

    # 随机颜色 (RGB + alpha=1)
    color = [random.random(), random.random(), random.random(), 1.0]

    return (start, end, color)

def generate_specs_list(num_frames=120, min_cyl=10, max_cyl=120):
    """生成 specs_list，每帧有若干随机圆柱."""
    specs_list = []
    for _ in range(num_frames):
        n_cyl = random.randint(min_cyl, max_cyl)
        specs = [random_cylinder() for _ in range(n_cyl)]
        specs_x_shift = [([spec[0][0] + 50, spec[0][1], spec[0][2]], [spec[1][0] + 50, spec[1][1], spec[1][2]], spec[2]) for spec in specs]
        specs_list.append(specs)
        specs_list.append(specs_x_shift)
    return specs_list


if __name__ == "__main__":
    specs_list = generate_specs_list(num_frames=24, min_cyl=10, max_cyl=120)
    render_whole(specs_list, output_path="/workspace/yanwenhao/dwpose_draw/render_preview/test.mp4")
