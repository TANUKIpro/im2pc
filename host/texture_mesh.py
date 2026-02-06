#!/usr/bin/env python3
"""
Multi-view texture projection onto mesh.

Takes a geometry-only mesh (PLY), camera poses, frames, masks, and intrinsics,
and bakes a UV-mapped texture from multiple views.

Pipeline:
  1. Load mesh and generate UV atlas (xatlas)
  2. Build texel → 3D position mapping
  3. For each view: project texels, check visibility (normal + mask), sample color
  4. Blend all views weighted by cos(view angle)
  5. Apply seam padding
  6. Export as OBJ + MTL + PNG

Usage:
    python host/texture_mesh.py \
        --mesh /path/to/object_mesh_final.ply \
        --intrinsics data/output_sam2/intrinsics.json \
        --poses data/output_sam2/camera_poses.json \
        --frames-dir data/output_sam2/frames \
        --masks-dir data/output_sam2/masks \
        --output-dir data/output_textured
"""

import argparse
import functools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData

print = functools.partial(print, flush=True)


# ---------------------------------------------------------------------------
# Mesh I/O
# ---------------------------------------------------------------------------


def load_mesh_ply(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load mesh from PLY. Returns (vertices (V,3) float64, faces (F,3) int32)."""
    ply = PlyData.read(path)
    v = ply["vertex"]
    vertices = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    f = ply["face"]
    faces = np.vstack(f["vertex_indices"]).astype(np.int32)
    return vertices, faces


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-face unit normals. Returns (F, 3)."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norms, 1e-10)


# ---------------------------------------------------------------------------
# UV Atlas (xatlas)
# ---------------------------------------------------------------------------


def generate_uv_atlas(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate UV atlas. Returns (new_verts, new_faces, uvs, vmapping)."""
    import xatlas

    print("Generating UV atlas with xatlas...")
    vmapping, new_faces, uvs = xatlas.parametrize(
        vertices.astype(np.float32), faces.astype(np.uint32)
    )
    new_vertices = vertices[vmapping]
    print(f"  {len(new_vertices)} vertices, {len(new_faces)} faces")
    print(f"  UV: u=[{uvs[:,0].min():.3f},{uvs[:,0].max():.3f}] "
          f"v=[{uvs[:,1].min():.3f},{uvs[:,1].max():.3f}]")
    return new_vertices, new_faces, uvs, vmapping


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


def load_intrinsics(path: str) -> np.ndarray:
    with open(path) as f:
        return np.array(json.load(f)["K"], dtype=np.float64)


def load_poses(path: str) -> np.ndarray:
    with open(path) as f:
        return np.array(json.load(f)["poses"], dtype=np.float64)


def project_points(pts: np.ndarray, c2w: np.ndarray, K: np.ndarray):
    """Project 3D→2D. Returns (uv (N,2), depths (N,))."""
    w2c = np.linalg.inv(c2w)
    cam = (w2c[:3, :3] @ pts.T).T + w2c[:3, 3]
    d = cam[:, 2].copy()
    sz = np.maximum(d, 1e-10)
    u = K[0, 0] * cam[:, 0] / sz + K[0, 2]
    v = K[1, 1] * cam[:, 1] / sz + K[1, 2]
    return np.column_stack([u, v]), d


# ---------------------------------------------------------------------------
# Texel → Face mapping (rasterize UV triangles into texture space)
# ---------------------------------------------------------------------------


def build_texel_mapping(
    faces: np.ndarray, uvs: np.ndarray, tex_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map each texel to a face + barycentric coords using OpenCV rasterization.
    Much faster than per-face Python loops.

    Returns:
        texel_face: (tex_size, tex_size) int32, -1 = empty
        texel_bary: (tex_size, tex_size, 3) float32
    """
    print(f"  Building texel mapping ({tex_size}x{tex_size})...")

    # Render face index buffer using OpenCV
    # Strategy: draw each face with its index as pixel value
    # Use int32 image with fillConvexPoly
    face_id_buf = np.full((tex_size, tex_size), -1, dtype=np.int32)

    # Scale UVs to texture coordinates
    uv_scaled = uvs * tex_size  # (V, 2)

    num_faces = len(faces)
    report_iv = max(num_faces // 10, 1)

    for fi in range(num_faces):
        if fi % report_iv == 0:
            print(f"    Face rasterize: {fi}/{num_faces} ({100*fi/num_faces:.0f}%)")

        i0, i1, i2 = faces[fi]
        pts = np.array([
            [uv_scaled[i0, 0], uv_scaled[i0, 1]],
            [uv_scaled[i1, 0], uv_scaled[i1, 1]],
            [uv_scaled[i2, 0], uv_scaled[i2, 1]],
        ], dtype=np.int32).reshape(3, 1, 2)
        cv2.fillConvexPoly(face_id_buf, pts, int(fi))

    # Now compute barycentric coords for all assigned texels
    valid = face_id_buf >= 0
    ys, xs = np.where(valid)
    fids = face_id_buf[ys, xs]

    # Texel center coordinates
    px = xs + 0.5
    py = ys + 0.5

    # Get UV coords of face vertices
    fi0 = faces[fids, 0]
    fi1 = faces[fids, 1]
    fi2 = faces[fids, 2]

    uv0 = uv_scaled[fi0]  # (N, 2)
    uv1 = uv_scaled[fi1]
    uv2 = uv_scaled[fi2]

    # Barycentric computation (vectorized)
    denom = (uv1[:, 1] - uv2[:, 1]) * (uv0[:, 0] - uv2[:, 0]) + \
            (uv2[:, 0] - uv1[:, 0]) * (uv0[:, 1] - uv2[:, 1])
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)

    w0 = ((uv1[:, 1] - uv2[:, 1]) * (px - uv2[:, 0]) +
          (uv2[:, 0] - uv1[:, 0]) * (py - uv2[:, 1])) / denom
    w1 = ((uv2[:, 1] - uv0[:, 1]) * (px - uv2[:, 0]) +
          (uv0[:, 0] - uv2[:, 0]) * (py - uv2[:, 1])) / denom
    w2 = 1.0 - w0 - w1

    texel_bary = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
    texel_bary[ys, xs, 0] = w0.astype(np.float32)
    texel_bary[ys, xs, 1] = w1.astype(np.float32)
    texel_bary[ys, xs, 2] = w2.astype(np.float32)

    print(f"    Valid texels: {valid.sum()} ({100*valid.sum()/(tex_size*tex_size):.1f}%)")
    return face_id_buf, texel_bary


# ---------------------------------------------------------------------------
# Texture Baking
# ---------------------------------------------------------------------------


def bake_texture(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    face_normals: np.ndarray,
    poses: np.ndarray,
    K: np.ndarray,
    frames_dir: str,
    masks_dir: str,
    img_w: int,
    img_h: int,
    tex_size: int = 2048,
    max_views: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bake texture from multiple views.
    Visibility = face normal facing camera + pixel inside mask.
    No explicit depth buffer (works well for mostly-convex objects).

    Returns: (texture (H,W,3) float32, texel_face (H,W) int32)
    """
    # Build texel mapping
    texel_face, texel_bary = build_texel_mapping(faces, uvs, tex_size)
    valid_texels = texel_face >= 0
    n_valid = valid_texels.sum()

    # Precompute texel 3D positions and normals
    print("Precomputing texel 3D positions...")
    ys, xs = np.where(valid_texels)
    fids = texel_face[ys, xs]
    barys = texel_bary[ys, xs]  # (N, 3)

    v0 = vertices[faces[fids, 0]]
    v1 = vertices[faces[fids, 1]]
    v2 = vertices[faces[fids, 2]]
    pos3d = barys[:, 0:1] * v0 + barys[:, 1:2] * v1 + barys[:, 2:3] * v2  # (N, 3)
    normals = face_normals[fids]  # (N, 3)
    print(f"  {len(pos3d)} texels with 3D positions")

    # Accumulate
    color_sum = np.zeros((n_valid, 3), dtype=np.float64)
    weight_sum = np.zeros(n_valid, dtype=np.float64)

    n_poses = len(poses)
    if max_views is not None and max_views < n_poses:
        view_idx = np.linspace(0, n_poses - 1, max_views, dtype=int)
    else:
        view_idx = np.arange(n_poses)

    for vi, vidx in enumerate(view_idx):
        print(f"\nView {vi+1}/{len(view_idx)} (frame {vidx:05d})...")
        c2w = poses[vidx]
        cam_pos = c2w[:3, 3]

        # Load frame + mask
        frame = cv2.imread(str(Path(frames_dir) / f"{vidx:05d}.jpg"))
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

        mask = cv2.imread(str(Path(masks_dir) / f"{vidx:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask_bool = mask > 127

        # View direction and cos angle (vectorized for all texels)
        view_dirs = cam_pos - pos3d  # (N, 3)
        dists = np.linalg.norm(view_dirs, axis=1, keepdims=True)
        view_dirs_n = view_dirs / np.maximum(dists, 1e-10)
        cos_angle = np.sum(normals * view_dirs_n, axis=1)

        # Face must be facing camera (cos > threshold)
        facing = cos_angle > 0.1

        if facing.sum() == 0:
            print("  No facing texels")
            continue

        # Project facing texels to image
        facing_pos = pos3d[facing]
        uv2d, depths = project_points(facing_pos, c2w, K)
        px = uv2d[:, 0]
        py = uv2d[:, 1]

        # Bounds + depth check
        ok = (depths > 0.01) & \
             (px >= 0) & (px < img_w - 1) & \
             (py >= 0) & (py < img_h - 1)

        # Mask check
        pxi = np.clip(px.astype(np.int32), 0, img_w - 1)
        pyi = np.clip(py.astype(np.int32), 0, img_h - 1)
        in_mask = mask_bool[pyi, pxi]
        ok = ok & in_mask

        n_ok = ok.sum()
        print(f"  Usable texels: {n_ok}")
        if n_ok == 0:
            continue

        # Sample colors (bilinear)
        colors = bilinear_sample(frame, px[ok], py[ok])
        w = cos_angle[facing][ok]

        # Map back to full texel indices
        facing_indices = np.where(facing)[0]
        final_indices = facing_indices[ok]

        color_sum[final_indices] += colors * w[:, None]
        weight_sum[final_indices] += w

    # Normalize
    has_color = weight_sum > 0
    color_sum[has_color] /= weight_sum[has_color, None]

    texture = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
    texture[ys, xs] = color_sum.astype(np.float32)

    cov = has_color.sum()
    print(f"\nTexture coverage: {cov}/{n_valid} ({100*cov/max(n_valid,1):.1f}%)")
    return texture, texel_face


def bilinear_sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear interpolation at fractional pixel coords."""
    h, w = img.shape[:2]
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    fx = (x - x0.astype(np.float64))[:, None]
    fy = (y - y0.astype(np.float64))[:, None]
    return (
        (1 - fx) * (1 - fy) * img[y0, x0]
        + fx * (1 - fy) * img[y0, x1]
        + (1 - fx) * fy * img[y1, x0]
        + fx * fy * img[y1, x1]
    )


# ---------------------------------------------------------------------------
# Seam Padding
# ---------------------------------------------------------------------------


def pad_texture_seams(
    texture: np.ndarray, texel_face: np.ndarray, iterations: int = 8
) -> np.ndarray:
    """Dilate texture islands to fill seam gaps."""
    print(f"Padding seams ({iterations} iterations)...")
    valid = (texel_face >= 0) & (np.sum(texture, axis=2) > 0)
    result = texture.copy()
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)

    for _ in range(iterations):
        empty = ~valid
        if not np.any(empty):
            break
        for c in range(3):
            ns = cv2.filter2D(result[:, :, c], -1, kernel, borderType=cv2.BORDER_CONSTANT)
            nc = cv2.filter2D(valid.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
            fill = empty & (nc > 0)
            if np.any(fill):
                result[:, :, c][fill] = ns[fill] / nc[fill]
                valid[fill] = True

    print(f"  Padded {valid.sum() - ((texel_face >= 0) & (np.sum(texture, axis=2) > 0)).sum()} texels")
    return result


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def save_obj_mtl(
    output_dir: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    texture: np.ndarray,
    basename: str = "textured_mesh",
):
    """Save as OBJ + MTL + texture PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    obj_path = output_dir / f"{basename}.obj"
    mtl_path = output_dir / f"{basename}.mtl"
    tex_path = output_dir / "texture.png"

    # Texture (flip V for OBJ convention)
    tex_u8 = np.clip(texture * 255, 0, 255).astype(np.uint8)
    tex_bgr = cv2.cvtColor(tex_u8, cv2.COLOR_RGB2BGR)[::-1]
    cv2.imwrite(str(tex_path), tex_bgr)
    print(f"Saved: {tex_path}")

    # MTL
    with open(mtl_path, "w") as f:
        f.write("newmtl material_0\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\n"
                "Ks 0.0 0.0 0.0\nd 1.0\nillum 1\nmap_Kd texture.png\n")
    print(f"Saved: {mtl_path}")

    # OBJ
    with open(obj_path, "w") as f:
        f.write(f"mtllib {basename}.mtl\nusemtl material_0\n\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for face in faces:
            a, b, c = face + 1
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
    print(f"Saved: {obj_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Multi-view texture baking")
    p.add_argument("--mesh", required=True)
    p.add_argument("--intrinsics", required=True)
    p.add_argument("--poses", required=True)
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--masks-dir", required=True)
    p.add_argument("--output-dir", default="data/output_textured")
    p.add_argument("--tex-size", type=int, default=2048)
    p.add_argument("--max-views", type=int, default=None)
    p.add_argument("--img-width", type=int, default=1080)
    p.add_argument("--img-height", type=int, default=1920)
    p.add_argument("--pad-iterations", type=int, default=8)
    args = p.parse_args()

    print("Loading mesh...")
    verts, faces = load_mesh_ply(args.mesh)
    print(f"  {len(verts)} verts, {len(faces)} faces")

    print("Computing normals...")
    fn = compute_face_normals(verts, faces)

    new_v, new_f, uvs, vmap = generate_uv_atlas(verts, faces)
    new_fn = compute_face_normals(new_v, new_f)

    K = load_intrinsics(args.intrinsics)
    poses = load_poses(args.poses)
    print(f"{len(poses)} poses, K: fx={K[0,0]:.1f} fy={K[1,1]:.1f}")

    texture, texel_face = bake_texture(
        new_v, new_f, uvs, new_fn, poses, K,
        args.frames_dir, args.masks_dir,
        args.img_width, args.img_height,
        tex_size=args.tex_size, max_views=args.max_views,
    )

    texture = pad_texture_seams(texture, texel_face, iterations=args.pad_iterations)

    save_obj_mtl(Path(args.output_dir), new_v, new_f, uvs, texture)
    print(f"\nDone! → {args.output_dir}")


if __name__ == "__main__":
    main()
