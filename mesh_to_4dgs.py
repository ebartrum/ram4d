"""
mesh_to_4dgs.py — Convert dynamic ActionMesh output to 4D Gaussian Splatting.

One Gaussian per mesh triangle. Uses UV-to-3D Jacobian to derive physically
motivated scale and orientation (mesh2splat approach). Deformation offsets are
computed per frame as centroid displacements.

Outputs:
  <output_path>/gaussians/gaussians.ply         — 3DGS PLY (Inpaint360GS compatible)
  <output_path>/gaussians/deformation_offsets.npy — (T, N_tri, 3) xyz offsets

Usage:
  python mesh_to_4dgs.py --output_path output/2026.03.02/actionmesh_gs_replace_corgi
  python mesh_to_4dgs.py --output_path ... --sigma 0.65
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
        "--output_path", required=True,
        help="ActionMesh output dir (from run_actionmesh.py / texture_actionmesh.py)."
    )
    parser.add_argument(
        "--sigma", type=float, default=0.65,
        help="mesh2splat Gaussian spread in UV space (default: 0.65)."
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
        uvs: (N, 2) float32 array in [0, 1] (u, v) where v=0 is top
    Returns:
        (N, 3) float32 RGB values
    """
    H, W = texture_np.shape[:2]
    u = uvs[:, 0]
    v = uvs[:, 1]

    # Flip v: UV (0,0) is bottom-left in OBJ convention, image (0,0) is top-left
    v_img = 1.0 - v

    # Pixel coords
    px = u * (W - 1)
    py = v_img * (H - 1)

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


def compute_per_triangle_gaussians(mesh, texture_np, sigma):
    """
    Compute static 3DGS parameters for each triangle in frame-0 mesh.

    Returns dict with keys: centroid, log_scale, quaternion, f_dc, opacity
    """
    v_pos = mesh.v_pos.cpu().numpy()      # (V_tex, 3)
    t_pos_idx = mesh.t_pos_idx.cpu().numpy()  # (F, 3)
    v_tex = mesh.v_tex.cpu().numpy()      # (V_uv, 2)
    t_tex_idx = mesh.t_tex_idx.cpu().numpy()  # (F, 3)

    F = t_pos_idx.shape[0]
    print(f"  Triangles: {F}")

    # Position indices for each triangle corner
    idx0, idx1, idx2 = t_pos_idx[:, 0], t_pos_idx[:, 1], t_pos_idx[:, 2]
    v0, v1, v2 = v_pos[idx0], v_pos[idx1], v_pos[idx2]  # (F, 3) each

    # UV indices
    uv_idx0, uv_idx1, uv_idx2 = t_tex_idx[:, 0], t_tex_idx[:, 1], t_tex_idx[:, 2]
    uv0, uv1, uv2 = v_tex[uv_idx0], v_tex[uv_idx1], v_tex[uv_idx2]  # (F, 2) each

    # --- Centroid ---
    centroid = (v0 + v1 + v2) / 3.0  # (F, 3)

    # --- 3D edge vectors ---
    dp1 = v1 - v0  # (F, 3)
    dp2 = v2 - v0

    # --- UV edge vectors ---
    duv1 = uv1 - uv0  # (F, 2)
    duv2 = uv2 - uv0

    # --- Surface normal ---
    normal = np.cross(dp1, dp2)  # (F, 3)
    norm_len = np.linalg.norm(normal, axis=-1, keepdims=True)
    degenerate_tri = (norm_len[:, 0] < 1e-10)
    norm_len = np.maximum(norm_len, 1e-10)
    normal = normal / norm_len  # (F, 3)

    # --- UV-to-3D Jacobian: J (F, 3, 2) ---
    # [dp1 | dp2] = J @ [[duv1.x, duv2.x], [duv1.y, duv2.y]]
    # J = [dp1 | dp2] @ inv(D) where D = [[duv1.x, duv2.x], [duv1.y, duv2.y]]
    D = np.stack([
        np.stack([duv1[:, 0], duv2[:, 0]], axis=-1),  # row 0: [duv1.x, duv2.x]
        np.stack([duv1[:, 1], duv2[:, 1]], axis=-1),  # row 1: [duv1.y, duv2.y]
    ], axis=-2)  # (F, 2, 2)

    det = D[:, 0, 0] * D[:, 1, 1] - D[:, 0, 1] * D[:, 1, 0]  # (F,)
    degenerate_uv = (np.abs(det) < 1e-10)

    # Invert D where non-degenerate
    D_inv = np.zeros_like(D)
    ok = ~degenerate_uv
    D_inv[ok, 0, 0] =  D[ok, 1, 1] / det[ok]
    D_inv[ok, 0, 1] = -D[ok, 0, 1] / det[ok]
    D_inv[ok, 1, 0] = -D[ok, 1, 0] / det[ok]
    D_inv[ok, 1, 1] =  D[ok, 0, 0] / det[ok]

    # [dp1 | dp2] is (F, 3, 2) with columns dp1, dp2
    dp_mat = np.stack([dp1, dp2], axis=-1)  # (F, 3, 2)
    J = dp_mat @ D_inv  # (F, 3, 2)  J[:,0]=tangent, J[:,1]=bitangent in 3D space

    # --- Scale from Jacobian ---
    tang_len = np.linalg.norm(J[:, :, 0], axis=-1)   # (F,) length of tangent in 3D
    bitang_len = np.linalg.norm(J[:, :, 1], axis=-1)  # (F,)

    scale_tang = sigma * tang_len
    scale_bitang = sigma * bitang_len
    scale_norm = 0.01 * np.minimum(scale_tang, scale_bitang)

    # Fallback for degenerate UV triangles: use edge lengths
    edge_len = (np.linalg.norm(dp1, axis=-1) + np.linalg.norm(dp2, axis=-1)) / 2.0
    fallback_scale = sigma * edge_len
    scale_tang = np.where(degenerate_uv, fallback_scale, scale_tang)
    scale_bitang = np.where(degenerate_uv, fallback_scale, scale_bitang)
    scale_norm = np.where(degenerate_uv, 0.01 * fallback_scale, scale_norm)

    # Clamp scales to avoid log(0)
    scale_tang = np.maximum(scale_tang, 1e-8)
    scale_bitang = np.maximum(scale_bitang, 1e-8)
    scale_norm = np.maximum(scale_norm, 1e-8)

    log_scale = np.stack([
        np.log(scale_tang),
        np.log(scale_bitang),
        np.log(scale_norm),
    ], axis=-1)  # (F, 3)

    # --- Rotation quaternion from orthonormal frame ---
    tangent = J[:, :, 0].copy()
    tang_norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    # Fallback tangent when UV is degenerate
    fallback_tangent = dp1 / np.maximum(np.linalg.norm(dp1, axis=-1, keepdims=True), 1e-10)
    use_fallback = (tang_norm[:, 0] < 1e-10) | degenerate_uv
    tangent = np.where(use_fallback[:, None], fallback_tangent, tangent)
    tang_norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    tangent = tangent / np.maximum(tang_norm, 1e-10)

    bitangent = np.cross(normal, tangent)  # (F, 3)
    bitan_norm = np.linalg.norm(bitangent, axis=-1, keepdims=True)
    # Handle near-parallel normal/tangent
    bad = bitan_norm[:, 0] < 1e-10
    if bad.any():
        perp = np.zeros_like(tangent)
        perp[:, 0] = 1.0
        alt_bitan = np.cross(normal, perp)
        alt_bitan = alt_bitan / np.maximum(np.linalg.norm(alt_bitan, axis=-1, keepdims=True), 1e-10)
        bitangent[bad] = alt_bitan[bad]
        bitan_norm[bad] = np.linalg.norm(bitangent[bad], axis=-1, keepdims=True)
    bitangent = bitangent / np.maximum(bitan_norm, 1e-10)

    # Rotation matrix: columns are [tangent, bitangent, normal] -> local X, Y, Z
    rot_matrix = np.stack([tangent, bitangent, normal], axis=-1)  # (F, 3, 3)

    rot_t = torch.from_numpy(rot_matrix.astype(np.float32))
    quaternion = rotation_matrix_to_quaternion(rot_t).numpy()  # (F, 4) wxyz

    # --- Color: bilinear sample texture at UV centroid ---
    uv_centroid = (uv0 + uv1 + uv2) / 3.0  # (F, 2)
    # Clamp UVs to valid range
    uv_centroid = np.clip(uv_centroid, 0.0, 1.0)
    rgb = sample_texture_at_uvs(texture_np, uv_centroid)  # (F, 3)
    f_dc = (rgb - 0.5) / C0  # SH DC coefficient, (F, 3)

    # --- Opacity (fully opaque): inverse sigmoid(0.99) ---
    opacity = np.full((F, 1), np.log(0.99 / 0.01), dtype=np.float32)  # ≈ 4.595

    n_degen = int(degenerate_uv.sum()) + int(degenerate_tri.sum())
    if n_degen > 0:
        print(f"  Degenerate triangles: {n_degen} (used edge-length fallback)")

    return {
        "centroid": centroid.astype(np.float32),
        "log_scale": log_scale.astype(np.float32),
        "quaternion": quaternion.astype(np.float32),
        "f_dc": f_dc.astype(np.float32),
        "opacity": opacity,
        "t_pos_idx": t_pos_idx,
    }


def compute_deformation_offsets(mesh, deformations_path, t_pos_idx, centroid_0, offset, scale):
    """
    Compute per-frame centroid displacement offsets.

    Returns:
        offsets: (T, N_tri, 3) float32 array
    """
    print("  Loading deformations...")
    deformations = np.load(deformations_path)  # (T, V, 3)
    T = deformations.shape[0]
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

    # Compute nearest-neighbour mapping: textured mesh verts -> original deformation verts
    print("  Computing vertex NN mapping...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v_tex = mesh.v_pos.to(device)                                         # (V_tex, 3)
    v_orig = torch.from_numpy(deformations_orig[0]).float().to(device)    # (V_orig, 3)

    dist_matrix = torch.cdist(v_tex.unsqueeze(0), v_orig.unsqueeze(0)).squeeze(0)
    min_dist, mapping_idx = torch.min(dist_matrix, dim=1)
    avg_dist = torch.mean(min_dist).item()
    print(f"  NN mapping avg distance: {avg_dist:.6f}")
    if avg_dist > 1e-4:
        print("  WARNING: High avg distance — possible misalignment between mesh and deformations.")
    mapping_idx_np = mapping_idx.cpu().numpy()  # (V_tex,)

    # Per-frame centroid offsets
    idx0 = t_pos_idx[:, 0]
    idx1 = t_pos_idx[:, 1]
    idx2 = t_pos_idx[:, 2]

    offsets = np.zeros((T, t_pos_idx.shape[0], 3), dtype=np.float32)
    for t in range(T):
        def_verts = deformations_orig[t]   # (V_orig, 3)
        def_v0 = def_verts[mapping_idx_np[idx0]]  # (F, 3)
        def_v1 = def_verts[mapping_idx_np[idx1]]
        def_v2 = def_verts[mapping_idx_np[idx2]]
        centroid_t = (def_v0 + def_v1 + def_v2) / 3.0
        offsets[t] = centroid_t - centroid_0

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
    output_path = args.output_path
    texture_dir = os.path.join(output_path, "texture")
    deformations_path = os.path.join(output_path, "deformations_vertices.npy")
    gaussians_dir = os.path.join(output_path, "gaussians")

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

    # --- Compute per-triangle Gaussians ---
    print("\n--- Computing per-triangle Gaussians (frame 0) ---")
    gaussians = compute_per_triangle_gaussians(mesh, texture_np, sigma=args.sigma)
    N_tri = gaussians["centroid"].shape[0]
    print(f"  N Gaussians: {N_tri}")

    # --- Compute deformation offsets ---
    offsets = None
    if os.path.exists(deformations_path):
        print("\n--- Computing deformation offsets ---")
        offsets = compute_deformation_offsets(
            mesh,
            deformations_path,
            gaussians["t_pos_idx"],
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
    print(f"  N Gaussians (triangles): {N_tri}")
    if offsets is not None:
        print(f"  Deformation offsets shape: {offsets.shape}")


if __name__ == "__main__":
    main()
