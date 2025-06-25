import os
import json
import numpy as np
import imageio.v2 as imageio
import cv2

def load_blender_data(basedir, split='train', half_res=True, n_views=None):
    """Load Blender synthetic dataset"""
    json_file = os.path.join(basedir, f'transforms_{split}.json')
    with open(json_file, 'r') as f:
        meta = json.load(f)
    
    imgs = []
    poses = []
    
    frames = meta['frames']
    if n_views is not None and split == 'train':
        # Select sparse views
        indices = np.linspace(0, len(frames)-1, n_views, dtype=int)
        frames = [frames[i] for i in indices]
    
    for frame in frames:
        fname = os.path.join(basedir, frame['file_path'] + '.png')
        try:
            img = imageio.imread(fname)
            
            # Handle RGBA
            if img.shape[-1] == 4:
                img = img[...,:3] * img[...,-1:] / 255. + (1. - img[...,-1:] / 255.)
            else:
                img = img[...,:3] / 255.
                
            imgs.append(img)
            poses.append(np.array(frame['transform_matrix'], dtype=np.float32)[:3, :4])
        except Exception as e:
            print(f"Could not load {fname}: {e}")
    
    if not imgs:
        raise ValueError(f"No images loaded from {basedir}")
    
    imgs = np.array(imgs, dtype=np.float32)
    poses = np.array(poses, dtype=np.float32)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3), dtype=np.float32)
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
    
    return imgs, poses, [H, W, focal]
