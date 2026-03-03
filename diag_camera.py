import sys, numpy as np
sys.path.insert(0, 'Inpaint360GS')
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat

extrinsics = read_extrinsics_binary('Inpaint360GS/data/inpaint360/bag/sparse/0/images.bin')
intrinsics = read_intrinsics_binary('Inpaint360GS/data/inpaint360/bag/sparse/0/cameras.bin')
sorted_images = sorted(extrinsics.values(), key=lambda x: x.name)
img = sorted_images[28]
cam = intrinsics[img.camera_id]
R_w2c = qvec2rotmat(img.qvec)
tvec  = np.array(img.tvec)
R_c2w = R_w2c.T
cam_center = -(R_c2w @ tvec)

print('Camera name:', img.name)
print('R_c2w:\n', np.round(R_c2w, 4))
print('tvec:', np.round(tvec, 4))
print('cam_center (world):', np.round(cam_center, 4))
print('cam right  in world:', np.round(R_c2w[:,0], 4))
print('cam down   in world:', np.round(R_c2w[:,1], 4))
print('cam fwd    in world:', np.round(R_c2w[:,2], 4))
print('world up  ~        :', np.round(-R_c2w[:,1], 4))

corgi = np.array([0.080, 0.934, 1.592])
to_cam = cam_center - corgi
print('corgi->camera vec  :', np.round(to_cam, 4))
print('corgi->camera dir  :', np.round(to_cam / np.linalg.norm(to_cam), 4))

# Also print a few other cameras to see the overall scene orientation
print('\n--- All cameras Y-down direction in world (approx gravity) ---')
downs = []
for si in sorted_images:
    R = qvec2rotmat(si.qvec).T
    downs.append(R[:,1])
mean_down = np.mean(downs, axis=0)
mean_down /= np.linalg.norm(mean_down)
print('Mean cam-down in world (approx gravity down):', np.round(mean_down, 4))
print('Mean world up (anti-gravity):', np.round(-mean_down, 4))
