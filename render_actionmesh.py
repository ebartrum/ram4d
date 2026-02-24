import sys
import os
import glob

# Fix sklearn TLS issue
if "LIBGOMP_PRELOADED" not in os.environ:
    libgomp_files = glob.glob(
        "/home/ubuntu/miniconda3/envs/mvadapter/lib/python*/site-packages/scikit_learn.libs/libgomp*.so*"
    )
    if libgomp_files:
        os.environ["LD_PRELOAD"] = libgomp_files[0]
        os.environ["LIBGOMP_PRELOADED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, os.path.abspath("actionmesh"))
sys.path.insert(0, os.path.abspath("actionmesh/third_party/TripoSG"))

import argparse
import numpy as np
import trimesh
from actionmesh.render.visualizer import ActionMeshVisualizer
from actionmesh.render.utils import save_video


def load_meshes(output_path):
    """Load per-frame meshes from saved deformation numpy arrays."""
    verts_path = os.path.join(output_path, "deformations_vertices.npy")
    faces_path = os.path.join(output_path, "deformations_faces.npy")

    verts = np.load(verts_path)  # (T, V, 3) — saved as [-z, x, y]
    faces = np.load(faces_path)  # (F, 3)

    print(f"Loaded {verts.shape[0]} frames, {verts.shape[1]} verts, {faces.shape[0]} faces")

    # Undo save_deformation coordinate transform: [-z, x, y] -> [x, y, z]
    v = verts.copy()
    verts_restored = np.stack([v[:, :, 1], v[:, :, 2], -v[:, :, 0]], axis=-1)

    meshes = [
        trimesh.Trimesh(vertices=verts_restored[i], faces=faces, process=False)
        for i in range(verts_restored.shape[0])
    ]
    return meshes


def main():
    parser = argparse.ArgumentParser(description="Render ActionMesh output as a normal-shaded MP4.")
    parser.add_argument("--output_path", required=True,
                        help="ActionMesh output directory containing deformations_*.npy.")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Render resolution (square). Default: 512.")
    parser.add_argument("--fps", type=int, default=16,
                        help="Output video FPS. Default: 16.")
    parser.add_argument("--cameras", nargs="+",
                        default=["Input", "U000", "U004", "U008"],
                        help="Camera views to render. Default: Input U000 U004 U008.")
    args = parser.parse_args()

    meshes = load_meshes(args.output_path)

    print(f"Initializing visualizer (cameras: {args.cameras}, size: {args.image_size})...")
    visualizer = ActionMeshVisualizer(
        image_size=args.image_size,
        bg_color=(1.0, 1.0, 1.0),
        cameras=args.cameras,
    )

    render_dir = os.path.join(args.output_path, "render")
    os.makedirs(render_dir, exist_ok=True)

    print(f"Rendering {len(meshes)} frames...")
    visualizer.render(
        meshes=meshes,
        device="cuda",
        output_dir=render_dir,
        input_frames=None,
    )

    # save_multiview_video_grid writes render_dir/grid_normal.mp4 at fps=12.
    # Re-save at the requested fps using our own writer.
    grid_normal = os.path.join(render_dir, "grid_normal.mp4")
    if os.path.exists(grid_normal) and args.fps != 12:
        import cv2
        import imageio
        frames = []
        cap = cv2.VideoCapture(grid_normal)
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        out_path = os.path.join(args.output_path, "render.mp4")
        imageio.mimsave(out_path, frames, fps=args.fps)
        print(f"Render saved to {out_path} ({len(frames)} frames @ {args.fps}fps)")
    else:
        import shutil
        out_path = os.path.join(args.output_path, "render.mp4")
        shutil.copy(grid_normal, out_path)
        print(f"Render saved to {out_path}")


if __name__ == "__main__":
    main()
