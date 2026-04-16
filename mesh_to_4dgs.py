"""
mesh_to_4dgs.py — Convert dynamic ActionMesh output to 4D Gaussian Splatting.

Samples Gaussians uniformly by surface area (area-weighted triangle selection +
random barycentric coordinates). Scale and orientation are derived from the
containing triangle's local frame (UV-to-3D Jacobian). Deformation offsets use
barycentric interpolation of per-vertex deformations.

Outputs:
  <output_dir>/gaussians.ply         — 3DGS PLY (Inpaint360GS compatible)
  <output_dir>/deformation_offsets.npy — (T, N_tri, 3) xyz offsets

Usage:
  python mesh_to_4dgs.py --input_mesh output/2026.03.03/actionmesh_gs_replace_corgi --output_dir output/2026.04.16/corgi_gaussians
  python mesh_to_4dgs.py --input_mesh ... --output_dir ... --n_gaussians 40000 --sigma 0.65
"""

import sys
import os
import glob

# Fix static TLS block issues (libgomp + libGLdispatch) by preloading before
# the dynamic linker locks in TLS layout. Re-exec to apply.
if "LIBGOMP_PRELOADED" not in os.environ:
    preload_files = []
    libgomp_files = glob.glob(
        "/home/ubuntu/miniconda3/envs/mvadapter/lib/python*/site-packages/scikit_learn.libs/libgomp*.so*"
    )
    if libgomp_files:
        preload_files.append(libgomp_files[0])
    libgl_path = "/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0"
    if os.path.exists(libgl_path):
        preload_files.append(libgl_path)
    if preload_files:
        os.environ["LD_PRELOAD"] = ":".join(preload_files)
        os.environ["LIBGOMP_PRELOADED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "actionmesh"))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "actionmesh", "third_party", "TripoSG"))

import argparse
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

from visualise_output import apply_mesh_transforms
from mvadapter.utils.mesh_utils import load_mesh

# SH DC coefficient (C0 from sh_utils.py)
C0 = 0.28209479177387814


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert dynamic ActionMesh output to 4D Gaussian Splatting."
    )
    parser.add_argument(
        "--input_mesh", required=True,
        help="ActionMesh output dir (from run_actionmesh.py / texture_actionmesh.py)."
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to write gaussians.ply and deformation_offsets.npy."
    )
    parser.add_argument(
        "--sigma", type=float, default=0.65,
        help="Fraction of the average 3D triangle edge length used as Gaussian scale (default: 0.65)."
    )
    parser.add_argument(
        "--n_gaussians", type=int, default=40000,
        help="Number of Gaussians to sample from the mesh surface (default: 40000)."
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for surface sampling (default: 0)."
    )
    return parser.parse_args()


def rotation_matrix_to_quaternion(rot):
    """
    Convert batch of rotation matrices to quaternions (wxyz).

    Args:
        rot: (N, 3, 3) rotation matrices (columns are local axes)
    Returns:
        (N, 4) quaternions in wxyz order
    """
    N = rot.shape[0]
    # Using the Shepperd method for numerical stability
    trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]  # (N,)

    quat = torch.zeros(N, 4, dtype=rot.dtype, device=rot.device)

    # Case: trace > 0
    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2.0  # s = 4*w
    quat[:, 0] = 0.25 * s
    quat[:, 1] = (rot[:, 2, 1] - rot[:, 1, 2]) / s
    quat[:, 2] = (rot[:, 0, 2] - rot[:, 2, 0]) / s
    quat[:, 3] = (rot[:, 1, 0] - rot[:, 0, 1]) / s

    # Case: rot[0,0] largest diagonal element
    mask1 = (rot[:, 0, 0] > rot[:, 1, 1]) & (rot[:, 0, 0] > rot[:, 2, 2]) & (trace <= 0)
    s1 = torch.sqrt(torch.clamp(1.0 + rot[:, 0, 0] - rot[:, 1, 1] - rot[:, 2, 2], min=1e-10)) * 2.0
    quat[mask1, 0] = (rot[mask1, 2, 1] - rot[mask1, 1, 2]) / s1[mask1]
    quat[mask1, 1] = 0.25 * s1[mask1]
    quat[mask1, 2] = (rot[mask1, 0, 1] + rot[mask1, 1, 0]) / s1[mask1]
    quat[mask1, 3] = (rot[mask1, 0, 2] + rot[mask1, 2, 0]) / s1[mask1]

    # Case: rot[1,1] largest diagonal element
    mask2 = (rot[:, 1, 1] > rot[:, 2, 2]) & ~mask1 & (trace <= 0)
    s2 = torch.sqrt(torch.clamp(1.0 + rot[:, 1, 1] - rot[:, 0, 0] - rot[:, 2, 2], min=1e-10)) * 2.0
    quat[mask2, 0] = (rot[mask2, 0, 2] - rot[mask2, 2, 0]) / s2[mask2]
    quat[mask2, 1] = (rot[mask2, 0, 1] + rot[mask2, 1, 0]) / s2[mask2]
    quat[mask2, 2] = 0.25 * s2[mask2]
    quat[mask2, 3] = (rot[mask2, 1, 2] + rot[mask2, 2, 1]) / s2[mask2]

    # Case: rot[2,2] largest diagonal element
    mask3 = ~mask1 & ~mask2 & (trace <= 0)
    s3 = torch.sqrt(torch.clamp(1.0 + rot[:, 2, 2] - rot[:, 0, 0] - rot[:, 1, 1], min=1e-10)) * 2.0
    quat[mask3, 0] = (rot[mask3, 1, 0] - rot[mask3, 0, 1]) / s3[mask3]
    quat[mask3, 1] = (rot[mask3, 0, 2] + rot[mask3, 2, 0]) / s3[mask3]
    quat[mask3, 2] = (rot[mask3, 1, 2] + rot[mask3, 2, 1]) / s3[mask3]
    quat[mask3, 3] = 0.25 * s3[mask3]

    # Normalise
    quat = torch.nn.functional.normalize(quat, dim=-1)
    return quat


def sample_texture_at_uvs(texture_np, uvs):
    """
    Bilinear sample texture at UV coordinates.

    Args:
        texture_np: (H, W, 3) float32 array in [0, 1]
        uvs: (N, 2) float32 array in [0, 1] (u, v) in image convention — v=0 at top
             (load_mesh with flip_uv=True already converts OBJ v=0-at-bottom to this)
    Returns:
        (N, 3) float32 RGB values
    """
    H, W = texture_np.shape[:2]
    u = uvs[:, 0]
    v = uvs[:, 1]

    # load_mesh(flip_uv=True) already converted v to image convention (v=0 at top),
    # so sample directly without any additional flip.
    px = u * (W - 1)
    py = v * (H - 1)

    x0 = np.floor(px).astype(np.int32).clip(0, W - 2)
    y0 = np.floor(py).astype(np.int32).clip(0, H - 2)
    x1 = x0 + 1
    y1 = y0 + 1

    fx = (px - x0).astype(np.float32)[:, None]  # (N, 1)
    fy = (py - y0).astype(np.float32)[:, None]

    c00 = texture_np[y0, x0]  # (N, 3)
    c10 = texture_np[y1, x0]
    c01 = texture_np[y0, x1]
    c11 = texture_np[y1, x1]

    rgb = (c00 * (1 - fy) * (1 - fx)
           + c10 * fy * (1 - fx)
           + c01 * (1 - fy) * fx
           + c11 * fy * fx)
    return rgb.astype(np.float32)


def sample_surface_gaussians(mesh, texture_np, sigma, n_gaussians, seed=0):
    """
    Sample Gaussians uniformly by surface area from the frame-0 mesh.

    Each Gaussian is placed at a random point inside a triangle, with triangles
    selected proportional to their area. Barycentric coordinates are stored for
    use in deformation offset computation.

    Returns dict with keys:
      centroid    (N, 3)  — 3D positions
      log_scale   (N, 3)
      quaternion  (N, 4)  — wxyz
      f_dc        (N, 3)
      opacity     (N, 1)
      tri_indices (N,)    — which triangle each Gaussian belongs to
      bary_coords (N, 3)  — barycentric weights (w0, w1, w2)
    """
    rng = np.random.default_rng(seed)

    v_pos = mesh.v_pos.cpu().numpy()          # (V, 3)
    t_pos_idx = mesh.t_pos_idx.cpu().numpy()  # (F, 3)
    v_tex = mesh.v_tex.cpu().numpy()          # (V_uv, 2)
    t_tex_idx = mesh.t_tex_idx.cpu().numpy()  # (F, 3)

    F = t_pos_idx.shape[0]
    print(f"  Triangles: {F}")

    # Corner positions and UVs for all triangles
    idx0, idx1, idx2 = t_pos_idx[:, 0], t_pos_idx[:, 1], t_pos_idx[:, 2]
    v0, v1, v2 = v_pos[idx0], v_pos[idx1], v_pos[idx2]  # (F, 3) each

    uv_idx0, uv_idx1, uv_idx2 = t_tex_idx[:, 0], t_tex_idx[:, 1], t_tex_idx[:, 2]
    uv0, uv1, uv2 = v_tex[uv_idx0], v_tex[uv_idx1], v_tex[uv_idx2]  # (F, 2) each

    # --- 3D edge vectors (used for scale, rotation, area) ---
    dp1 = v1 - v0  # (F, 3)
    dp2 = v2 - v0

    # --- Surface normals ---
    normal_raw = np.cross(dp1, dp2)  # (F, 3)
    norm_len = np.linalg.norm(normal_raw, axis=-1, keepdims=True)
    degenerate_tri = (norm_len[:, 0] < 1e-10)
    areas = norm_len[:, 0] / 2.0  # triangle area = |cross| / 2

    # --- Area-weighted triangle sampling ---
    areas_safe = np.maximum(areas, 0.0)
    total_area = areas_safe.sum()
    if total_area < 1e-12:
        raise ValueError("Mesh has zero total surface area.")
    probs = areas_safe / total_area
    tri_indices = rng.choice(F, size=n_gaussians, p=probs)  # (N,)

    # --- Random barycentric coordinates (uniform in triangle) ---
    r1 = rng.random(n_gaussians).astype(np.float32)
    r2 = rng.random(n_gaussians).astype(np.float32)
    sqrt_r1 = np.sqrt(r1)
    w0 = 1.0 - sqrt_r1
    w1 = sqrt_r1 * (1.0 - r2)
    w2 = sqrt_r1 * r2
    bary_coords = np.stack([w0, w1, w2], axis=-1)  # (N, 3)

    # --- Sampled positions ---
    sv0 = v0[tri_indices]; sv1 = v1[tri_indices]; sv2 = v2[tri_indices]
    centroid = w0[:, None] * sv0 + w1[:, None] * sv1 + w2[:, None] * sv2  # (N, 3)

    # --- Per-triangle geometry for scale / rotation ---
    # Each Gaussian inherits its triangle's local frame and scale.
    sdp1 = dp1[tri_indices]  # (N, 3)
    sdp2 = dp2[tri_indices]
    snorm_raw = normal_raw[tri_indices]  # (N, 3)
    snorm_len = np.maximum(norm_len[tri_indices], 1e-10)  # (N, 1)
    normal = snorm_raw / snorm_len  # (N, 3)

    # --- UV edge vectors for tangent frame ---
    duv1 = uv1 - uv0  # (F, 2)
    duv2 = uv2 - uv0
    sduv1 = duv1[tri_indices]  # (N, 2)
    sduv2 = duv2[tri_indices]

    D = np.stack([
        np.stack([sduv1[:, 0], sduv2[:, 0]], axis=-1),
        np.stack([sduv1[:, 1], sduv2[:, 1]], axis=-1),
    ], axis=-2)  # (N, 2, 2)
    det = D[:, 0, 0] * D[:, 1, 1] - D[:, 0, 1] * D[:, 1, 0]
    degenerate_uv = (np.abs(det) < 1e-10)

    D_inv = np.zeros_like(D)
    ok = ~degenerate_uv
    D_inv[ok, 0, 0] =  D[ok, 1, 1] / det[ok]
    D_inv[ok, 0, 1] = -D[ok, 0, 1] / det[ok]
    D_inv[ok, 1, 0] = -D[ok, 1, 0] / det[ok]
    D_inv[ok, 1, 1] =  D[ok, 0, 0] / det[ok]

    dp_mat = np.stack([sdp1, sdp2], axis=-1)  # (N, 3, 2)
    J = dp_mat @ D_inv  # (N, 3, 2)

    # --- Scale from triangle edge lengths ---
    edge_len_3d = (np.linalg.norm(sdp1, axis=-1) + np.linalg.norm(sdp2, axis=-1)) / 2.0
    scale_tang   = np.maximum(sigma * edge_len_3d, 1e-8)
    scale_bitang = np.maximum(sigma * edge_len_3d, 1e-8)
    scale_norm   = np.maximum(0.01 * sigma * edge_len_3d, 1e-8)

    log_scale = np.stack([
        np.log(scale_tang),
        np.log(scale_bitang),
        np.log(scale_norm),
    ], axis=-1)  # (N, 3)

    # --- Rotation from tangent frame ---
    tangent = J[:, :, 0].copy()
    tang_norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    fallback_tangent = sdp1 / np.maximum(np.linalg.norm(sdp1, axis=-1, keepdims=True), 1e-10)
    use_fallback = (tang_norm[:, 0] < 1e-10) | degenerate_uv
    tangent = np.where(use_fallback[:, None], fallback_tangent, tangent)
    tangent = tangent / np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-10)

    bitangent = np.cross(normal, tangent)  # (N, 3)
    bitan_norm = np.linalg.norm(bitangent, axis=-1, keepdims=True)
    bad = bitan_norm[:, 0] < 1e-10
    if bad.any():
        perp = np.zeros_like(tangent); perp[:, 0] = 1.0
        alt_bitan = np.cross(normal, perp)
        alt_bitan = alt_bitan / np.maximum(np.linalg.norm(alt_bitan, axis=-1, keepdims=True), 1e-10)
        bitangent[bad] = alt_bitan[bad]
        bitan_norm[bad] = np.linalg.norm(bitangent[bad], axis=-1, keepdims=True)
    bitangent = bitangent / np.maximum(bitan_norm, 1e-10)

    rot_matrix = np.stack([tangent, bitangent, normal], axis=-1)  # (N, 3, 3)
    rot_t = torch.from_numpy(rot_matrix.astype(np.float32))
    quaternion = rotation_matrix_to_quaternion(rot_t).numpy()  # (N, 4) wxyz

    # --- Color: interpolate UV then sample texture ---
    suv0 = uv0[tri_indices]; suv1 = uv1[tri_indices]; suv2 = uv2[tri_indices]
    uv_sample = w0[:, None] * suv0 + w1[:, None] * suv1 + w2[:, None] * suv2  # (N, 2)
    uv_sample = np.clip(uv_sample, 0.0, 1.0)
    rgb = sample_texture_at_uvs(texture_np, uv_sample)  # (N, 3)
    f_dc = (rgb - 0.5) / C0

    # --- Opacity ---
    opacity = np.full((n_gaussians, 1), np.log(0.99 / 0.01), dtype=np.float32)

    n_degen = int(degenerate_uv.sum()) + int(degenerate_tri[tri_indices].sum())
    if n_degen > 0:
        print(f"  Gaussians on degenerate triangles: {n_degen}")

    return {
        "centroid":    centroid.astype(np.float32),
        "log_scale":   log_scale.astype(np.float32),
        "quaternion":  quaternion.astype(np.float32),
        "f_dc":        f_dc.astype(np.float32),
        "opacity":     opacity,
        "tri_indices": tri_indices,       # (N,)
        "bary_coords": bary_coords,       # (N, 3)
    }


def compute_deformation_offsets(mesh, deformations_path, tri_indices, bary_coords, pos_0, offset, scale):
    """
    Compute per-frame position offsets for each sampled Gaussian using barycentric interpolation.

    Args:
        tri_indices: (N,) triangle index for each Gaussian
        bary_coords: (N, 3) barycentric weights (w0, w1, w2)
        pos_0: (N, 3) frame-0 positions (used to compute relative offsets)

    Returns:
        offsets: (T, N, 3) float32 array
    """
    print("  Loading deformations...")
    deformations = np.load(deformations_path)  # (T, V, 3)
    T = deformations.shape[0]
    N = tri_indices.shape[0]
    print(f"  Deformations shape: {deformations.shape}")

    # Undo save_deformation coordinate transform: saved as [-z, x, y] -> restore [x, y, z]
    deformations_orig = np.zeros_like(deformations)
    deformations_orig[:, :, 0] = deformations[:, :, 1]   # x <- second
    deformations_orig[:, :, 1] = deformations[:, :, 2]   # y <- third
    deformations_orig[:, :, 2] = -deformations[:, :, 0]  # z <- neg first

    # Apply the same mesh transforms as load_mesh (offset, scale, front_x_to_y)
    for i in range(T):
        deformations_orig[i] = apply_mesh_transforms(
            deformations_orig[i], offset, scale, front_x_to_y=True
        )

    # Compute nearest-neighbour mapping: mesh verts -> deformation verts
    print("  Computing vertex NN mapping...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v_mesh = mesh.v_pos.to(device)                                         # (V, 3)
    v_orig = torch.from_numpy(deformations_orig[0]).float().to(device)    # (V_orig, 3)

    dist_matrix = torch.cdist(v_mesh.unsqueeze(0), v_orig.unsqueeze(0)).squeeze(0)
    min_dist, mapping_idx = torch.min(dist_matrix, dim=1)
    avg_dist = torch.mean(min_dist).item()
    print(f"  NN mapping avg distance: {avg_dist:.6f}")
    if avg_dist > 1e-4:
        print("  WARNING: High avg distance — possible misalignment between mesh and deformations.")
    mapping_idx_np = mapping_idx.cpu().numpy()  # (V,) maps mesh vert -> deformation vert

    # Precompute per-Gaussian vertex indices in deformation array
    t_pos_idx = mesh.t_pos_idx.cpu().numpy()  # (F, 3)
    v0_idx = mapping_idx_np[t_pos_idx[tri_indices, 0]]  # (N,)
    v1_idx = mapping_idx_np[t_pos_idx[tri_indices, 1]]
    v2_idx = mapping_idx_np[t_pos_idx[tri_indices, 2]]
    w0 = bary_coords[:, 0:1]  # (N, 1)
    w1 = bary_coords[:, 1:2]
    w2 = bary_coords[:, 2:3]

    offsets = np.zeros((T, N, 3), dtype=np.float32)
    for t in range(T):
        def_verts = deformations_orig[t]   # (V_orig, 3)
        pos_t = (w0 * def_verts[v0_idx]
               + w1 * def_verts[v1_idx]
               + w2 * def_verts[v2_idx])   # (N, 3)
        offsets[t] = pos_t - pos_0

        if t % 10 == 0:
            max_off = np.abs(offsets[t]).max()
            print(f"  Frame {t}/{T}: max offset = {max_off:.4f}")

    return offsets


def write_gaussians_ply(path, centroid, log_scale, quaternion, f_dc, opacity, num_sh_rest=45, num_obj_dc=16):
    """
    Write 3DGS PLY file compatible with Inpaint360GS GaussianModel.load_ply().

    Field layout (all float32):
      x, y, z, nx, ny, nz,
      f_dc_0, f_dc_1, f_dc_2,
      f_rest_0 ... f_rest_{num_sh_rest-1},
      opacity,
      scale_0, scale_1, scale_2,
      rot_0, rot_1, rot_2, rot_3,
      obj_dc_0 ... obj_dc_{num_obj_dc-1}
    """
    N = centroid.shape[0]
    normals = np.zeros((N, 3), dtype=np.float32)

    # f_rest: all zeros (SH degree 3 rest coefficients)
    f_rest = np.zeros((N, num_sh_rest), dtype=np.float32)

    # obj_dc: all zeros
    obj_dc = np.zeros((N, num_obj_dc), dtype=np.float32)

    # Build attribute list (matches construct_list_of_attributes in gaussian_model.py)
    attr_names = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(3):
        attr_names.append(f'f_dc_{i}')
    for i in range(num_sh_rest):
        attr_names.append(f'f_rest_{i}')
    attr_names.append('opacity')
    for i in range(3):
        attr_names.append(f'scale_{i}')
    for i in range(4):
        attr_names.append(f'rot_{i}')
    for i in range(num_obj_dc):
        attr_names.append(f'obj_dc_{i}')

    dtype_full = [(name, 'f4') for name in attr_names]

    # Concatenate all arrays in order
    attributes = np.concatenate([
        centroid,          # (N, 3)
        normals,           # (N, 3)
        f_dc,              # (N, 3)
        f_rest,            # (N, 45)
        opacity,           # (N, 1)
        log_scale,         # (N, 3)
        quaternion,        # (N, 4)
        obj_dc,            # (N, 16)
    ], axis=1)             # (N, 3+3+3+45+1+3+4+16 = 78)

    elements = np.empty(N, dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    PlyData([el]).write(path)


def main():
    args = parse_args()
    input_mesh = args.input_mesh
    texture_dir = os.path.join(input_mesh, "texture")
    deformations_path = os.path.join(input_mesh, "deformations_vertices.npy")
    gaussians_dir = args.output_dir

    # --- Load texture image ---
    print("\n--- Loading texture ---")
    texture_img_path = os.path.join(texture_dir, "mesh_00_texture.png")
    if not os.path.exists(texture_img_path):
        raise FileNotFoundError(f"Texture not found: {texture_img_path}")
    texture_img = Image.open(texture_img_path).convert("RGB")
    uv_size = texture_img.width
    print(f"  Texture: {uv_size}x{texture_img.height}")
    texture_np = np.array(texture_img).astype(np.float32) / 255.0  # (H, W, 3)

    # --- Load mesh ---
    print("\n--- Loading mesh ---")
    mesh_path = os.path.join(texture_dir, "mesh_00.obj")
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mesh, offset, scale = load_mesh(
        mesh_path,
        rescale=True,
        move_to_center=True,
        front_x_to_y=True,
        default_uv_size=uv_size,
        device=device,
        return_transform=True,
        merge_vertices=False,
    )
    print(f"  Vertices: {mesh.v_pos.shape[0]}, Faces: {mesh.t_pos_idx.shape[0]}")
    print(f"  UV verts: {mesh.v_tex.shape[0]}, UV faces: {mesh.t_tex_idx.shape[0]}")
    print(f"  offset: {offset}, scale: {scale:.6f}")

    # --- Sample Gaussians uniformly by surface area ---
    print("\n--- Sampling surface Gaussians (frame 0) ---")
    gaussians = sample_surface_gaussians(mesh, texture_np, sigma=args.sigma,
                                         n_gaussians=args.n_gaussians, seed=args.seed)
    N = gaussians["centroid"].shape[0]
    print(f"  N Gaussians: {N}")

    # --- Compute deformation offsets ---
    offsets = None
    if os.path.exists(deformations_path):
        print("\n--- Computing deformation offsets ---")
        offsets = compute_deformation_offsets(
            mesh,
            deformations_path,
            gaussians["tri_indices"],
            gaussians["bary_coords"],
            gaussians["centroid"],
            offset,
            scale,
        )
        print(f"  Offsets shape: {offsets.shape}")
        print(f"  Frame 0 max offset: {np.abs(offsets[0]).max():.6f} (should be ~0)")
    else:
        print(f"\nNo deformations found at {deformations_path}, skipping offset computation.")

    # --- Write outputs ---
    print("\n--- Writing outputs ---")
    os.makedirs(gaussians_dir, exist_ok=True)

    ply_path = os.path.join(gaussians_dir, "gaussians.ply")
    write_gaussians_ply(
        ply_path,
        gaussians["centroid"],
        gaussians["log_scale"],
        gaussians["quaternion"],
        gaussians["f_dc"],
        gaussians["opacity"],
    )
    print(f"  Written: {ply_path}")

    if offsets is not None:
        offsets_path = os.path.join(gaussians_dir, "deformation_offsets.npy")
        np.save(offsets_path, offsets.astype(np.float32))
        print(f"  Written: {offsets_path}")

    print(f"\nDone.")
    print(f"  N Gaussians: {N}")
    if offsets is not None:
        print(f"  Deformation offsets shape: {offsets.shape}")


if __name__ == "__main__":
    main()
