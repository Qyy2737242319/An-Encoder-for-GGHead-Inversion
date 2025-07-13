import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    ## path
    parser.add_argument('--cfg', type=str,  metavar="FILE", help='path to config file', default='./src/gghead/config/swinv2.yaml')
    parser.add_argument("--data", type=str, help='path to data directory', default='../example/real_person')
    parser.add_argument("--outdir", type=str, help='path to output directory', default='../output/')
    parser.add_argument("--cuda", type=str, help="specify used cuda idx ", default='0')

    ## model
    parser.add_argument("--mlp_layer", type=int, default=2)
    parser.add_argument("--start_from_latent_avg", type=bool, default=True)

    ## other
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--w_frames', type=int, default=240)
    parser.add_argument("--multi_view", action="store_true", default=False)
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--shape", action="store_true", default=False)
    parser.add_argument("--edit", action="store_true", default=False)

    ## edit
    parser.add_argument("--edit_attr", type=str, help="editing attribute direction", default="glass")
    parser.add_argument("--alpha", type=float, help="editing alpha", default=1.0)

    parser.add_argument("--run_name", type=str, default="GGHEAD-1")

    parser.add_argument("--iteration", type=int, default=2000000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--per_iteration", type=int, default=10)
    parser.add_argument("--n_styles", type=int, default=14)
    parser.add_argument("--use_pixelshuffle", action="store_true", default=False)
    parser.add_argument("--stop_encoder", type=int, default=2000000)
    parser.add_argument("--triplane", type=int, default=1000000)

    parser.add_argument('--id_lambda_triplane', default=0.1, type=float, help='ID loss multiplier factor for triplanenet')
    parser.add_argument('--l1_lambda_triplane', default=1.0, type=float, help='L1 smooth loss multiplier factor for triplanenet')
    parser.add_argument('--lpips_lambda_triplane', default=0.8, type=float, help='LPIPS loss multiplier factor for triplanenet')

    parser.add_argument('--l2_lambda_psp', default=1.0, type=float, help='L2 loss multiplier factor for psp')
    parser.add_argument('--id_lambda_psp', default=0.1, type=float, help='ID loss multiplier factor for psp')
    parser.add_argument('--lpips_lambda_psp', default=0.8, type=float, help='LPIPS loss multiplier factor for psp')

    parser.add_argument('--id_index', default=150, type=int, help='point the specific id ,only for single inversion use')
    parser.add_argument('--E_ckpt', type=str, help='path to encoder checkpoint')
    parser.add_argument('--triplane_ckpt', type=str, help='path to triplane checkpoint')
    parser.add_argument('--E_optim_ckpt', type=str, help='path to encoder optimizer checkpoint')
    parser.add_argument('--triplane_optim_ckpt', type=str, help='path to triplane optimizer checkpoint')

    parser.add_argument('--image_path', type=str, help='path to test image')
    return parser