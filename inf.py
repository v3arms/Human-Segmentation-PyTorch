#!/usr/bin/env python
# coding: utf-8

# In[1]:


from base.base_inference import VideoInference
from models import UNet
from models import DeepLabV3Plus
from subprocess import call
import torch
import ffmpeg

# In[16]:


# CHECKPOINT = "./pretr/DeepLabV3Plus_ResNet18.pth"
CHECKPOINT = "pretr/model_best.pth"
BACKBONE   = "resnet18"
VIDEO_INP  = "./vd/5.mp4"
VIDEO_OUT  = "./vd/out.mp4"
VIDEO_TMP  = "./vd/.tmp.mp4"

# In[19]:


!nvidia-smi

# In[3]:


torch.cuda.set_device(1)
torch.cuda.current_device()

# In[4]:


model = DeepLabV3Plus(backbone=BACKBONE, num_classes=2)
trained_dict = torch.load(CHECKPOINT, map_location="cpu")['state_dict']
model.load_state_dict(trained_dict, strict=False)
model.cuda()
model.eval()
print('ok')

# In[17]:


call(["rm", VIDEO_TMP])
inference = VideoInference(
    model=model,
    video_path=VIDEO_INP,
    video_out_path=VIDEO_TMP,
    input_size=480,
    background_path = "./backgrounds/white.jpg",
    use_cuda=True,
    draw_mode='matting'
    # frame_range=(0, 1000)
)
inference.run()

# In[18]:


in1 = ffmpeg.input(VIDEO_INP)
in2 = ffmpeg.input(VIDEO_TMP)
out = ffmpeg.output(in1.audio, in2.video, VIDEO_OUT)
out.run(overwrite_output=True)

# In[ ]:



