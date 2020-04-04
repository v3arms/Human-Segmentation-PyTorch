from base.base_inference import VideoInference
from models import UNet
from models import DeepLabV3Plus
from subprocess import call
import torch
import ffmpeg
import argparse
from sys import exit


parser = argparse.ArgumentParser(description="Arguments for the script")
parser.add_argument('--inp', type=str, help='Input video')
parser.add_argument('--out', type=str, help='Output video')
parser.add_argument('--model', type=str, help='Path to .pth pretrained model')
args = parser.parse_args()


# CHECKPOINT = "./pretr/DeepLabV3Plus_ResNet18.pth"
CHECKPOINT = args.model
BACKBONE   = "resnet18"


if not torch.cuda.is_available() : 
    print("GPU is not available. Abort.")
    exit(0)


model = DeepLabV3Plus(backbone=BACKBONE, num_classes=2)
trained_dict = torch.load(CHECKPOINT, map_location="cpu")['state_dict']
model.load_state_dict(trained_dict, strict=False)
model.cuda()
model.eval()
print('Model loaded successfully.')


inference = VideoInference(
    model=model,
    video_path=args.inp,
    video_out_path='./tmp',
    input_size=320,
    background_path = "./backgrounds/white.jpg",
    use_cuda=True,
    draw_mode='matting'
)
inference.run()


in1 = ffmpeg.input(args.inp)
in2 = ffmpeg.input('./tmp')
out = ffmpeg.output(in1.audio, in2.video, args.out)
out.run(overwrite_output=True)




