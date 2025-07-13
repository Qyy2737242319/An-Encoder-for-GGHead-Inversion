import os

from torchvision.transforms.v2.functional import resize

#os.environ['PYTORCH_CUDA_ALLOC_CONF']="expandable_segments:True"

os.environ['PYTORCH_CUDA_ALLOC_CONF']="max_split_size_mb:32"
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
import torch.nn as nn
import torchvision
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

from criteria import id_loss
from criteria.lpips.lpips import LPIPS

from gghead.config.infer_config import get_parser
from gghead.config.swin_config import get_config
#from gghead.models.encoder_512 import GOAEncoder as Encoder
#from gghead.models.stylegan_encoder import Encoder as Encoder
from gghead.models.psp_encoder import GradualStyleEncoder as psp_Encoder
from gghead.models.triplane_encoder import TriPlane_Encoder as tri_Encoder
from torchvision import transforms
import tqdm
import matplotlib.pyplot as plt
from PIL import Image

class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()

        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)

        vgg = torchvision.models.vgg19(pretrained=True).features

        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer

        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().to(device)

    def forward(self, source, target):
        loss = 0
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight * self.criterion(source, target)

        return loss

def calc_loss_psp(x, y_hat, loss_dict,opts,Id_loss,lpips_loss):
    loss = 0.0
    id_logs = None
    if opts.id_lambda_psp > 0:
        loss_id, sim_improvement, _ = Id_loss(y_hat, x, x)
        loss_dict['loss_id_psp'] = float(loss_id)
        loss_dict['id_improve_psp'] = float(sim_improvement)
        loss = loss_id * opts.id_lambda_psp

    if opts.l2_lambda_psp > 0:
        loss_l2 = F.mse_loss(y_hat, x)
        loss_dict['loss_l2_psp'] = float(loss_l2)
        loss += loss_l2 * opts.l2_lambda_psp

    if opts.lpips_lambda_psp > 0:
        loss_lpips = lpips_loss(y_hat, x)
        loss_dict['loss_lpips_psp'] = float(loss_lpips)
        loss += loss_lpips * opts.lpips_lambda_psp
    loss_dict['loss_psp'] = float(loss)
    return loss, loss_dict, id_logs



def calc_loss_triplane(x, y_hat, loss_dict,opts,Id_loss,lpips_loss):
    loss = 0.0
    id_logs = None

    if opts.id_lambda_triplane > 0:
        loss_id, sim_improvement, _ = Id_loss(y_hat, x, x)
        loss_dict['loss_id_triplane'] = float(loss_id)
        loss_dict['id_improve_triplane'] = float(sim_improvement)
        loss = loss_id * opts.id_lambda_triplane

    if opts.l1_lambda_triplane > 0:
        loss_l1 = F.smooth_l1_loss(y_hat, x)
        loss_dict['loss_l1_triplane'] = float(loss_l1)
        loss += loss_l1 * opts.l1_lambda_triplane

    if opts.lpips_lambda_triplane > 0:
        loss_lpips = lpips_loss(y_hat, x)
        loss_dict['loss_lpips_triplane'] = float(loss_lpips)
        loss += loss_lpips * opts.lpips_lambda_triplane

    loss_dict['loss_triplane'] = float(loss)
    return loss, loss_dict, id_logs

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
         batch_size: int = 6,
         resolution: int = 512,
         ):
    device = torch.device('cuda')

    resize = transforms.Resize((256, 256))


    Id_loss = id_loss.IDLoss().to(device).eval()
    lpips_loss = LPIPS(net_type='alex').to(device).eval()




    run_name = opts.run_name
    model_manager = find_model_manager(run_name)
    checkpoint = model_manager._resolve_checkpoint_id(checkpoint)
    G = model_manager.load_checkpoint(checkpoint, load_ema=True).to(device)

    for param in G.parameters():
        param.requires_grad_(False)

    G.eval()

    move_z = 2.7

    poses = circle_around_axis(96, axis=Vec3(0, 1, 0), up=Vec3(0, 1, 0), move=Vec3(0, 0, 0),
                               distance=3.5 * move_z / 2.7,
                               theta_from=1.3 * np.pi, theta_to=1.8 * np.pi)

    cs = [encode_camera_params(pose, DEFAULT_INTRINSICS) for pose in poses]
    cs = torch.stack([torch.from_numpy(c).cuda() for c in cs])

    c_front = encode_camera_params(
        Pose(matrix_or_rotation=np.eye(3), translation=(0, 0, 3.5), pose_type=PoseType.CAM_2_WORLD,
             camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL), DEFAULT_INTRINSICS)

    c_front = torch.from_numpy(c_front).cuda().unsqueeze(0)


    pose, intrinsics = np.array(c_front.cpu()[:,:16]).reshape(c_front.shape[0],4, 4), np.array(c_front.cpu()[:,16:]).reshape(c_front.shape[0],3, 3)
    flipped_pose = flip_yaw(pose)
    mirror_c_front = np.concatenate([flipped_pose.reshape(c_front.shape[0],-1), intrinsics.reshape(c_front.shape[0],-1)],axis=1)
    mirror_c_front = torch.from_numpy(mirror_c_front).cuda()

    sh_ref_cam, intrinsics = decode_camera_params(c_front[0].cpu())

    #swin_config = get_config(opts)
    # E_psp = Encoder(swin_config, mlp_layer=opts.mlp_layer, stage_list=[0, 0, 0]).to(device)
    #E_psp = Encoder(512,512).to(device)
    Encoder = psp_Encoder(50, 'ir_se', opts).to(device)
    if opts.E_ckpt is not None:
        Encoder.load_state_dict(torch.load(opts.E_ckpt, map_location=device))

    triplane_encoder = tri_Encoder(opts).to(device)
    if opts.triplane_ckpt is not None:
        triplane_encoder.load_state_dict(torch.load(opts.triplane_ckpt, map_location=device))


    for param in Encoder.parameters():
        param.requires_grad_(True)

    Encoder.train()

    num_params = sum(param.numel() for param in Encoder.parameters())
    print("Encoder parmeters number is :    ", num_params)

    optimizer_encoder = torch.optim.Adam(Encoder.parameters(), lr=opts.lr)
    if opts.E_optim_ckpt is not None:
        optimizer_encoder.load_state_dict(torch.load(opts.E_optim_ckpt))

    optimizer_triplane = torch.optim.Adam(triplane_encoder.parameters(), lr=opts.lr)
    if opts.triplane_optim_ckpt is not None:
        optimizer_triplane.load_state_dict(torch.load(opts.triplane_optim_ckpt))


    # random_index = [index for index in range(70000)]
    camera_index = [index for index in range(95)]

    #total_loss = []

    gc.collect()
    torch.cuda.empty_cache()


    for i in tqdm.tqdm(range(opts.iteration)):

        with torch.no_grad():

            index = opts.id_index
            cam_index = np.random.choice(camera_index, batch_size)

            camera = torch.cat((cs[[0],:],cs[cam_index[:-1], :]))

            temp_pose,  temp_intrinsics = np.array(camera.cpu()[:, :16]).reshape(camera.shape[0], 4, 4), np.array(
                camera.cpu()[:, 16:]).reshape(camera.shape[0], 3, 3)
            flipped_pose = flip_yaw(temp_pose)
            mirror_camera = np.concatenate(
                [flipped_pose.reshape(camera.shape[0], -1), temp_intrinsics.reshape(camera.shape[0], -1)], axis=1)
            mirror_camera = torch.from_numpy(mirror_camera).cuda()


            rng = torch.Generator(device)
            rng.manual_seed(index)

            z = torch.randn((1, G._config.z_dim), device='cuda', generator=rng)

            w = G.mapping(z, camera[[0],:], truncation_psi=truncation_psi)

            #image = np.expand_dims(np.array(Image.open(f"./dataset/{int(index[0])}.jpg").convert('RGB')),axis=0)
            for j in range(1,batch_size):
                w = torch.cat((w,G.mapping(z, camera[[j],:], truncation_psi=truncation_psi)),dim=0)
            #image = torch.tensor(image).permute(0, 3, 1, 2).to(device)
            for c_batch, w_batch in zip(batchify_sliced(camera, batch_size=batch_size),
                                        batchify_sliced(w, batch_size=batch_size)):
                output_true = G.synthesis(w_batch, c_batch, sh_ref_cam=sh_ref_cam, return_masks=True, noise_mode='const',
                                     neural_rendering_resolution=resolution)

            test = Img.from_normalized_torch(output_true['image'][:, :3, ...]).to_torch().img
            test_img = test * 255
            test_image_resized = torch.tensor(resize(test_img)).to(device).to(torch.float32) / 127.5 - 1.
            test_image = torch.tensor(test_img).to(device).to(torch.float32) / 127.5 - 1.
            input_image = F1.normalize(test, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            input_image = F1.resize(input_image, [256, 256])



            for c_batch, w_batch in zip(batchify_sliced(mirror_camera, batch_size=batch_size),
                                        batchify_sliced(w, batch_size=batch_size)):
                output_true_mirror = G.synthesis(w_batch, c_batch, sh_ref_cam=sh_ref_cam, return_masks=True, noise_mode='const',
                                     neural_rendering_resolution=resolution)

            test_mirror = Img.from_normalized_torch(output_true_mirror['image'][:, :3, ...]).to_torch().img
            test_img_mirror = test_mirror * 255
            test_image_resized_mirror = torch.tensor(resize(test_img_mirror)).to(device).to(torch.float32) / 127.5 - 1.
            test_image_mirror = torch.tensor(test_img_mirror).to(device).to(torch.float32) / 127.5 - 1.
            input_image_mirror = F1.normalize(test_mirror, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            input_image_mirror = F1.resize(input_image_mirror, [256, 256])

        loss_dict = {}

        if i<opts.stop_encoder:
            optimizer_encoder.zero_grad()
            Encoder.requires_grad_(True)
            Encoder.train()
            #for _ in tqdm.tqdm(range(opts.per_iteration)):
            optimizer_encoder.zero_grad()
            #rec_ws, _ = E_psp(test_image[[0],...])
            rec_ws = Encoder(input_image)
            rec_ws = torch.mean(rec_ws,dim = 1)

            w_ = G.mapping(rec_ws, camera, truncation_psi=truncation_psi)
            #w_ = w_.repeat(batch_size, 1, 1)

            for c_batch, w_batch in zip(batchify_sliced(camera, batch_size=batch_size),
                                        batchify_sliced(w_, batch_size=batch_size)):
                output = G.synthesis(w_batch, c_batch, sh_ref_cam=sh_ref_cam, return_masks=True, noise_mode='const', #8 22 256 256
                                     neural_rendering_resolution=resolution)

            pre_image = output['image'][:, :3, ...]
            loss_psp, loss_dict, id_logs = calc_loss_psp(test_image.clone().detach(),pre_image, loss_dict,opts,Id_loss, lpips_loss)

            optimizer_encoder.zero_grad()
            loss_psp.backward()
            optimizer_encoder.step()
            optimizer_encoder.zero_grad()

            if i < opts.stop_encoder:
                print("psp_s loss:" + str(loss_psp.item()))

            shuffle_index=torch.randperm(batch_size)
            render_camera=camera[shuffle_index,...]
            test_image_shuffle=test_image[shuffle_index,...]
            render_camera_mirror=mirror_camera[shuffle_index,...]

            rec_ws = Encoder(input_image)

            rec_ws = torch.mean(rec_ws, dim=1)

            w_ = G.mapping(rec_ws, camera, truncation_psi=truncation_psi)
            # w_ = w_.repeat(batch_size, 1, 1)

            for c_batch, w_batch in zip(batchify_sliced(render_camera, batch_size=batch_size),
                                        batchify_sliced(w_, batch_size=batch_size)):
                output = G.synthesis(w_batch, c_batch, sh_ref_cam=sh_ref_cam, return_masks=True, noise_mode='const',
                                     # 8 22 256 256
                                     neural_rendering_resolution=resolution)

            pre_image = output['image'][:, :3, ...]
            loss_psp, loss_dict, id_logs = calc_loss_psp(test_image_shuffle.clone().detach(), pre_image, loss_dict, opts,
                                                         Id_loss, lpips_loss)

            optimizer_encoder.zero_grad()
            loss_psp.backward()
            optimizer_encoder.step()
            optimizer_encoder.zero_grad()
            Encoder.requires_grad_(False)
            Encoder.eval()

        gc.collect()
        torch.cuda.empty_cache()

        if i >= opts.triplane:
            optimizer_triplane.zero_grad()
            triplane_encoder.requires_grad_(True)
            triplane_encoder.train()
            rec_ws = Encoder(input_image)
            rec_ws = torch.mean(rec_ws, dim=1)

            w_ = G.mapping(rec_ws, camera, truncation_psi=truncation_psi)
            # w_ = w_.repeat(batch_size, 1, 1)
            with torch.no_grad():
                for c_batch, w_batch in zip(batchify_sliced(camera, batch_size=batch_size),
                                            batchify_sliced(w_, batch_size=batch_size)):
                    output = G.synthesis(w_batch, c_batch, sh_ref_cam=sh_ref_cam, return_masks=True, noise_mode='const',
                                         neural_rendering_resolution=resolution) # 8 22 256 256

                pre_image = output['image'][:, :3, ...]

                for c_batch, w_batch in zip(batchify_sliced(mirror_camera, batch_size=batch_size),
                                            batchify_sliced(w_, batch_size=batch_size)):
                    output_m = G.synthesis(w_batch, c_batch, sh_ref_cam=sh_ref_cam, return_masks=True, noise_mode='const',
                                         neural_rendering_resolution=resolution) # 8 22 256 256

                pre_image_mirror = output_m['image'][:, :3, ...]

                for c_batch, w_batch in zip(batchify_sliced(render_camera, batch_size=batch_size),
                                            batchify_sliced(w_, batch_size=batch_size)):
                    output_r = G.synthesis(w_batch, c_batch, sh_ref_cam=sh_ref_cam, return_masks=True, noise_mode='const',
                                         neural_rendering_resolution=resolution) # 8 22 256 256

                #pre_image_r = output_m['image'][:, :3, ...]


            x_clone = test_image
            x_clone_mirror = test_image_mirror
            y_hat_initial_clone =pre_image.clone().detach()
            y_hat_initial_clone_mirror = pre_image_mirror.clone().detach()

            x_input = torch.cat(
                [y_hat_initial_clone, x_clone - y_hat_initial_clone, x_clone_mirror - y_hat_initial_clone_mirror], dim=1)

            triplane_offsets = triplane_encoder(x_input, output["triplanes"].clone().detach())

            for c_batch, w_batch in zip(batchify_sliced(camera, batch_size=batch_size),
                                        batchify_sliced(w_, batch_size=batch_size)):
                outs = G.synthesis(w_batch, c_batch, sh_ref_cam=sh_ref_cam, return_masks=True, noise_mode='const',
                                     neural_rendering_resolution=resolution,triplane_offsets=triplane_offsets)

            y_hat = outs['image'][:, :3, ...]

            loss1, loss_dict1, id_logs1 = calc_loss_triplane(test_image.clone().detach(), y_hat, loss_dict,opts,Id_loss, lpips_loss)

            triplane_offsets_2 = triplane_encoder(x_input, output_r["triplanes"].clone().detach())

            for c_batch, w_batch in zip(batchify_sliced(render_camera, batch_size=batch_size),
                                        batchify_sliced(w_, batch_size=batch_size)):
                outs_ = G.synthesis(w_batch, c_batch, sh_ref_cam=sh_ref_cam, return_masks=True, noise_mode='const',
                                     neural_rendering_resolution=resolution,triplane_offsets=triplane_offsets_2)

            y_hat_ = outs_['image'][:, :3, ...]
            loss2, loss_dict2, id_logs2 = calc_loss_triplane(test_image_shuffle.clone().detach(), y_hat_, loss_dict, opts, Id_loss,
                                                          lpips_loss)

            optimizer_triplane.zero_grad()
            loss1.backward(retain_graph=True)
            loss2.backward()
            optimizer_triplane.step()
            optimizer_triplane.zero_grad()
            triplane_encoder.requires_grad_(False)
            triplane_encoder.eval()


        gc.collect()
        torch.cuda.empty_cache()

        if i < opts.stop_encoder:
            print("psp_c loss:" + str(loss_psp.item()))

        if i>=opts.triplane:
            print("triplane_c loss:" + str((loss1.item())))
            print("triplane_s loss:" + str((loss2.item())))

        if i != 0 and i% 1000==0:

            with torch.no_grad():
                # w_ = G.mapping(rec_ws, c_front, truncation_psi=truncation_psi)
                # w_ = w_.repeat(len(cs), 1, 1)

                all_frames = []

                for c_batch, w_batch in zip(batchify_sliced(camera, batch_size=batch_size),
                                            batchify_sliced(w_, batch_size=batch_size)):
                    output_test = G.synthesis(w_batch, c_batch, sh_ref_cam=sh_ref_cam, return_masks=True, noise_mode='const',
                                         neural_rendering_resolution=resolution,triplane_offsets=triplane_offsets)
                    frames = [Img.from_normalized_torch(image).to_numpy().img[..., :3] for image in output_test['image']]
                    all_frames.extend(frames)
                output_folder = f"./finetuning/test_result/"
                ensure_directory_exists(output_folder)
                mediapy.write_video(f"{output_folder}/step_{i}.mp4", all_frames, fps=24)

        if i!=0 and  i % 2000==0:
            torch.save(Encoder.state_dict(), f'inversion_E_params_{i}.pth')
            optimizer_state = optimizer_encoder.state_dict()
            torch.save(optimizer_state, f'inversion_optimizer_psp_{i}.pth')
            torch.save(triplane_encoder.state_dict(), f'inversion_T_params_{i}.pth')
            torch.save(optimizer_triplane.state_dict(), f'inversion_optimizer_triplane_{i}.pth')


parser = get_parser()
opts = parser.parse_args()
train(opts)