from base.base_inference import VideoInference
from models import UNet
from models import DeepLabV3Plus
from subprocess import call
import torch
import ffmpeg
import argparse
import sys


parser = argparse.ArgumentParser(description="Arguments for the script")
parser.add_argument('--inp', type=str, help='Input video')
parser.add_argument('--out', type=str, help='Output video')
parser.add_argument('--model', type=str, help='Path to .pth pretrained model',
        default='./pretr/model.pth')
parser.add_argument('--frange', nargs=2, type=int, help='Frame range to process')


if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)


args = parser.parse_args()



if args.inp is None or args.out is None :
    print('No arguments specified. Exiting.')
    sys.exit(0)


# CHECKPOINT = "./pretr/DeepLabV3Plus_ResNet18.pth"
CHECKPOINT = args.model
BACKBONE   = "resnet18"


if not torch.cuda.is_available() : 
    print("GPU is not available. Abort.")
    sys.exit(0)


model = DeepLabV3Plus(backbone=BACKBONE, num_classes=2)
trained_dict = torch.load(CHECKPOINT, map_location="cpu")['state_dict']
model.load_state_dict(trained_dict, strict=False)
model.cuda()
model.eval()
print('Model loaded successfully.')


inference = VideoInference(
    model=model,
    video_path=args.inp,
    video_out_path='./.tmp.mp4',
    input_size=320,
    background_path = "./backgrounds/white.jpg",
    use_cuda=True,
    draw_mode='matting',
    frame_range=args.frange
)

print('Start processing frames...')
inference.run()
print('Done.')

print('Running ffmpeg to merge video channels...')
in1 = ffmpeg.input(args.inp)
in2 = ffmpeg.input('./.tmp.mp4')
out = ffmpeg.output(in1.audio, in2.video, args.out)
out.run(overwrite_output=True)
print('All done.')




