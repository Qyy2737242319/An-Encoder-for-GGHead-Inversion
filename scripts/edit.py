import gc
import sys

from eg3d.projector.PTI.configs.global_config import run_name
from mpmath.identification import transforms

sys.path.append("./src/")

from typing import Optional

import mediapy
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as F1
import cv2
import tyro
import lpips
from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.image import Img
from dreifus.matrix import Pose
from dreifus.trajectory import circle_around_axis
from dreifus.vector import Vec3
from eg3d.datamanager.nersemble import encode_camera_params, decode_camera_params
from elias.util import ensure_directory_exists
from elias.util.batch import batchify_sliced
from tqdm import tqdm

from gghead.constants import DEFAULT_INTRINSICS
from gghead.env import GGH_RENDERINGS_PATH
from gghead.model_manager.finder import find_model_manager

from gghead.config.infer_config import get_parser
from gghead.config.swin_config import get_config
# from gghead.models.encoder_512 import GOAEncoder as Encoder
from gghead.models.stylegan_encoder import Encoder as Encoder
from torchvision import transforms
from gghead.models.psp_encoder import GradualStyleEncoder as psp_Encoder
from gghead.models.triplane_encoder import TriPlane_Encoder as tri_Encoder
import tqdm

from PIL import Image

def flip_yaw(pose_matrix):
    flipped = pose_matrix.copy()
    flipped[:,0, 1] *= -1
    flipped[:,0, 2] *= -1
    flipped[:,1, 0] *= -1
    flipped[:,2, 0] *= -1
    flipped[:,0, 3] *= -1
    return flipped


def train(opts,
         checkpoint: int = -1,
         n_seeds: int = 8,  # If no seeds specified, generate heads for seeds 0-n
         seeds: Optional[str] = None,  # comma-separated list of seeds to sample from
         truncation_psi: float = 0.7,
         batch_size: int = 16,
         resolution: int = 512,
         ):
    device = torch.device('cuda')

    run_name = opts.run_name
    model_manager = find_model_manager(run_name)
    checkpoint = model_manager._resolve_checkpoint_id(checkpoint)
    G = model_manager.load_checkpoint(checkpoint, load_ema=True).to(device)
    G.eval()

    move_z = 2.7
    poses = circle_around_axis(96,axis=Vec3(0, 1, 0), up=Vec3(0, 1, 0), move=Vec3(0, 0, 0), distance=3.5 * move_z / 2.7,
                               theta_from=1.3 * np.pi,theta_to=1.8 * np.pi)

    # poses = circle_around_axis(96, up=Vec3(0, 1, 0), move=Vec3(0, 0, move_z), distance=0.6 * move_z / 2.7,
    #                            theta_to=2 * np.pi)


    cs = [encode_camera_params(pose, DEFAULT_INTRINSICS) for pose in poses]

    cs = torch.stack([torch.from_numpy(c).cuda() for c in cs])

    temp_pose, temp_intrinsics = np.array(cs.cpu()[:, :16]).reshape(cs.shape[0], 4, 4), np.array(
        cs.cpu()[:, 16:]).reshape(cs.shape[0], 3, 3)
    flipped_pose = flip_yaw(temp_pose)
    mirror_camera = np.concatenate(
        [flipped_pose.reshape(cs.shape[0], -1), temp_intrinsics.reshape(cs.shape[0], -1)], axis=1)
    mirror_camera = torch.from_numpy(mirror_camera).cuda()

    c_front = encode_camera_params(
        Pose(matrix_or_rotation=np.eye(3), translation=(0, 0, 3.5), pose_type=PoseType.CAM_2_WORLD,
             camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL), DEFAULT_INTRINSICS)

    c_front = torch.from_numpy(c_front).cuda().unsqueeze(0)
    sh_ref_cam, intrinsics = decode_camera_params(c_front[0].cpu())

    #swin_config = get_config(opts)
    #E = Encoder(swin_config, mlp_layer=opts.mlp_layer, stage_list=[0, 0, 0]).to(device)
    #E = Encoder(512, 512).to(device)
    Encoder = psp_Encoder(50, 'ir_se', opts).to(device)
    if opts.E_ckpt is not None:
        Encoder.load_state_dict(torch.load(opts.E_ckpt, map_location=device))
    triplane_encoder = tri_Encoder(opts).to(device)

    if opts.triplane_ckpt is not None:
        triplane_encoder.load_state_dict(torch.load(opts.triplane_ckpt, map_location=device))

    test_img = np.array(Image.open(f'./test_data/{opts.id_index}.jpg').convert('RGB'))
    test_img = torch.tensor(test_img).permute(2,0,1).unsqueeze(0).to(device)

    test_image = test_img / 255.0
    test_image = F1.normalize(test_image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    test_image_512 = test_image
    test_image = F1.resize(test_image, [256, 256])

    #test_image = torch.tensor(test_img).to(device).to(torch.float32) / 127.5 - 1.

    rec_ws = Encoder(test_image)
    rec_ws = torch.mean(rec_ws,dim = 1)

    sigma = opts.sigma

    edit_w = torch.zeros_like(rec_ws)
    edit_w[opts.w_start, opts.w_end] = sigma * rec_ws[opts.w_start, opts.w_end]  #0 id 10 mouth 11 cloth,eyes,ears 13 mouth
    #50+- biaoqing
    #150+- light guangzhao
    #199 hair style
    #200 nianling+ biaoqing
    ws_edit = rec_ws + edit_w

    w = G.mapping(rec_ws, c_front, truncation_psi=truncation_psi)
    w_edit = G.mapping(ws_edit, c_front, truncation_psi=truncation_psi)


    w = w.repeat(len(cs), 1, 1)
    w_edit = w_edit.repeat(len(cs), 1, 1)

    all_frames = []

    with torch.no_grad():
        for c_batch, w_batch in zip(batchify_sliced(cs, batch_size=batch_size),
                                    batchify_sliced(w, batch_size=batch_size)):
            output_test = G.synthesis(w_batch, c_batch, sh_ref_cam=sh_ref_cam, return_masks=True,
                                      noise_mode='const',
                                      neural_rendering_resolution=resolution)

        x_clone = test_image_512

        for c_batch, w_batch in zip(batchify_sliced(c_front, batch_size=batch_size),
                                    batchify_sliced(w[[0], ...], batch_size=batch_size)):
            output_photo = G.synthesis(w_batch, c_batch, sh_ref_cam=sh_ref_cam, return_masks=True,
                                       noise_mode='const',
                                       neural_rendering_resolution=resolution)

        test = Img.from_normalized_torch(output_photo['image'][:, :3, ...]).to_torch().img
        test_img = test * 255
        test_img = test_img[0, ...].permute(1, 2, 0).cpu().numpy()
        save_img = Image.fromarray(test_img.astype(np.uint8))
        save_img.save(f"./test_result/{opts.id_index}.png")

        y_hat_initial_clone = output_photo['image'][:, :3, ...].clone().detach()


        with torch.no_grad():
            for c_batch, w_batch in zip(batchify_sliced(cs, batch_size=batch_size),
                                        batchify_sliced(w_edit, batch_size=batch_size)):
                output_edit = G.synthesis(w_batch, c_batch, sh_ref_cam=sh_ref_cam, return_masks=True,
                                          noise_mode='const',
                                          neural_rendering_resolution=resolution)
                frames = [Img.from_normalized_torch(image).to_numpy().img[..., :3] for image in output_edit['image']]
                all_frames.extend(frames)
            output_folder = "./test_result/"
            ensure_directory_exists(output_folder)
            mediapy.write_video(f"{output_folder}/edit.mp4", all_frames, fps=24)
        # for c_batch, w_batch in zip(batchify_sliced(mirror_camera, batch_size=batch_size),
        #                             batchify_sliced(w, batch_size=batch_size)):
        #     output_m = G.synthesis(w_batch, c_batch, sh_ref_cam=sh_ref_cam, return_masks=True, noise_mode='const',
        #                            neural_rendering_resolution=resolution)  # 8 22 256 256
        #
        # pre_image_mirror = output_m['image'][:, :3, ...]
        #
        # y_hat_initial_clone_mirror = resize(pre_image_mirror.clone().detach())

        x_input = torch.cat(
            [y_hat_initial_clone, x_clone - y_hat_initial_clone, x_clone - y_hat_initial_clone], dim=1)

        x_input = x_input.repeat(batch_size, 1, 1, 1)

        triplane_offsets = triplane_encoder(x_input, output_test["triplanes"].clone().detach())

        edit_triplane_offsets = triplane_offsets+output_edit["triplanes"]-output_test["triplanes"]


        all_frames = []
        for c_batch, w_batch in zip(batchify_sliced(cs, batch_size=batch_size),
                                    batchify_sliced(w_edit, batch_size=batch_size)):
            outs = G.synthesis(w_batch, c_batch, sh_ref_cam=sh_ref_cam, return_masks=True, noise_mode='const',
                               neural_rendering_resolution=resolution, triplane_offsets=edit_triplane_offsets)
            frames = [Img.from_normalized_torch(image).to_numpy().img[..., :3] for image in outs['image']]
            all_frames.extend(frames)
        output_folder = "./test_result/"
        ensure_directory_exists(output_folder)
        mediapy.write_video(f"{output_folder}/edit_inversion_offset.mp4", all_frames, fps=24)


parser = get_parser()
opts = parser.parse_args()
train(opts)