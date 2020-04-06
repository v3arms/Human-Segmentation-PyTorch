#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import cv2, torch
import numpy as np
from time import time
from torch.nn import functional as F
from tqdm import tqdm

#------------------------------------------------------------------------------
#   BaseInference
#------------------------------------------------------------------------------
class BaseInference(object):
	def __init__(self, model, color_f=[255,0,0], color_b=[0,0,255], kernel_sz=25, sigma=0, background_path=None):
		self.model = model
		self.color_f = color_f
		self.color_b = color_b
		self.kernel_sz = kernel_sz
		self.sigma = sigma
		self.background_path = background_path
		if background_path is not None:
			self.background = cv2.imread(background_path)[...,::-1]
			self.background = self.background.astype(np.float32)


	def load_image(self):
		raise NotImplementedError


	def resize_image(self, image, expected_size, pad_value, ret_params=False, mode=cv2.INTER_LINEAR):
		"""
		image (ndarray) with either shape of [H,W,3] for RGB or [H,W] for grayscale.
		Padding is added so that the content of image is in the center.
		"""
		h, w = image.shape[:2]
		if w>h:
			w_new = int(expected_size)
			h_new = int(h * w_new / w)
			image = cv2.resize(image, (w_new, h_new), interpolation=mode)

			pad_up = (w_new - h_new) // 2
			pad_down = w_new - h_new - pad_up
			if len(image.shape)==3:
				pad_width = ((pad_up, pad_down), (0,0), (0,0))
				constant_values=((pad_value, pad_value), (0,0), (0,0))
			elif len(image.shape)==2:
				pad_width = ((pad_up, pad_down), (0,0))
				constant_values=((pad_value, pad_value), (0,0))

			image = np.pad(
				image,
				pad_width=pad_width,
				mode="constant",
				constant_values=constant_values,
			)
			if ret_params:
				return image, pad_up, 0, h_new, w_new
			else:
				return image

		elif w<h:
			h_new = int(expected_size)
			w_new = int(w * h_new / h)
			image = cv2.resize(image, (w_new, h_new), interpolation=mode)

			pad_left = (h_new - w_new) // 2
			pad_right = h_new - w_new - pad_left
			if len(image.shape)==3:
				pad_width = ((0,0), (pad_left, pad_right), (0,0))
				constant_values=((0,0), (pad_value, pad_value), (0,0))
			elif len(image.shape)==2:
				pad_width = ((0,0), (pad_left, pad_right))
				constant_values=((0,0), (pad_value, pad_value))

			image = np.pad(
				image,
				pad_width=pad_width,
				mode="constant",
				constant_values=constant_values,
			)
			if ret_params:
				return image, 0, pad_left, h_new, w_new
			else:
				return image

		else:
			image = cv2.resize(image, (expected_size, expected_size), interpolation=mode)
			if ret_params:
				return image, 0, 0, expected_size, expected_size
			else:
				return image



	def preprocess(self, image, *args):
		raise NotImplementedError


	def predict(self, X):
		raise NotImplementedError


	def draw_matting(self, image, mask):
		"""
		image (np.uint8) shape (H,W,3)
		mask  (np.float32) range from 0 to 1, shape (H,W)
		"""
		mask = 255*(1.0-mask)
		mask = np.expand_dims(mask, axis=2)
		mask = np.tile(mask, (1,1,3))
		mask = mask.astype(np.uint8)
		image_alpha = cv2.add(image, mask)
		return image_alpha


	def draw_transparency(self, image, mask):
		"""
		image (np.uint8) shape (H,W,3)
		mask  (np.float32) range from 0 to 1, shape (H,W)
		"""
		mask = mask.round()
		alpha = np.zeros_like(image, dtype=np.uint8)
		alpha[mask==1, :] = self.color_f
		alpha[mask==0, :] = self.color_b
		image_alpha = cv2.add(image, alpha)
		return image_alpha


	def draw_background(self, image, mask):
		"""
		image (np.uint8) shape (H,W,3)
		mask  (np.float32) range from 0 to 1, shape (H,W)
		"""
		image = image.astype(np.float32)
		mask_filtered = cv2.GaussianBlur(mask, (self.kernel_sz, self.kernel_sz), self.sigma)
		mask_filtered = np.expand_dims(mask_filtered, axis=2)
		mask_filtered = np.tile(mask_filtered, (1,1,3))

		image_alpha = image*mask_filtered + self.background*(1-mask_filtered)
		return image_alpha.astype(np.uint8)

	
	def draw_white(self, image, mask):
		image[mask == 0] = 255
		return image



#------------------------------------------------------------------------------
#   VideoInference
#------------------------------------------------------------------------------
class VideoInference(BaseInference):
	def __init__(self, model, video_path, video_out_path, input_size, use_cuda=True, draw_mode='white',
				color_f=[255,0,0], color_b=[0,0,255], kernel_sz=25, sigma=0, background_path=None, frame_range=None):

		# Initialize
		super(VideoInference, self).__init__(model, color_f, color_b, kernel_sz, sigma, background_path)
		self.input_size = input_size
		self.use_cuda = use_cuda
		self.draw_mode = draw_mode
		if draw_mode=='matting':
			self.draw_func = self.draw_matting
		elif draw_mode=='transparency':
			self.draw_func = self.draw_transparency
		elif draw_mode=='background':
			self.draw_func = self.draw_background
		elif draw_mode=='white':
			self.draw_func = self.draw_white
		else:
			raise NotImplementedError

		# Preprocess
		self.mean = np.array([0.485,0.456,0.406])[None,None,:]
		self.std = np.array([0.229,0.224,0.225])[None,None,:]

		# Read video
		self.video_path = video_path
		self.cap = cv2.VideoCapture(video_path)
		# _, frame = self.cap.read()
		self.H, self.W = int(self.cap.get(3)), int(self.cap.get(4))
		self.frame_range = frame_range
		self.cur_frame = 0
		self.fr_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

		self.video_out_path = video_out_path
		self.writer = cv2.VideoWriter(
							self.video_out_path, 
							cv2.VideoWriter_fourcc(*"mp4v"), 
							self.cap.get(cv2.CAP_PROP_FPS), 
							(self.H, self.W),	
						)


	def load_image(self):
		_, frame = self.cap.read()
		image = frame[...,::-1]
		return image


	def preprocess(self, image):
		image, pad_up, pad_left, h_new, w_new = self.resize_image(image, expected_size=self.input_size, pad_value=0, ret_params=True)
		self.pad_up = pad_up
		self.pad_left = pad_left
		self.h_new = h_new
		self.w_new = w_new

		# image = cv2.resize(image, (self.input_size,self.input_size), interpolation=cv2.INTER_LINEAR)
		image = image.astype(np.float32) / 255.0
		image = (image - self.mean) / self.std
		X = np.transpose(image, axes=(2, 0, 1))
		X = np.expand_dims(X, axis=0)
		X = torch.tensor(X, dtype=torch.float32)
		return X


	def predict(self, X):
		with torch.no_grad():
			if self.use_cuda:
				mask = self.model(X.cuda())
				mask = mask[..., self.pad_up: self.pad_up+self.h_new, self.pad_left: self.pad_left+self.w_new]
				mask = F.interpolate(mask, size=(self.W, self.H), mode='bilinear', align_corners=True)
				mask = F.softmax(mask, dim=1)
				mask = mask[0,1,...].cpu().numpy()
			else:
				mask = self.model(X)
				mask = mask[..., self.pad_up: self.pad_up+self.h_new, self.pad_left: self.pad_left+self.w_new]
				mask = F.interpolate(mask, size=(self.W, self.H), mode='bilinear', align_corners=True)
				mask = F.softmax(mask, dim=1)
				mask = mask[0,1,...].numpy()
			return mask


	def run(self, qt = False):
		if self.frame_range is None:
			self.frame_range = (0, self.fr_count - 10)

		for i in tqdm(range(self.frame_range[0], self.frame_range[1] - 10)):
			self.cur_frame = i

			image = self.load_image()
			X = self.preprocess(image)
			mask = self.predict(X)

			#if image.shape != mask.shape :
			#	raise ValueError(mask.shape, image.shape)

			image_alpha = self.draw_func(image, mask)
			self.writer.write(image_alpha[..., ::-1])

		self.writer.release()
		self.cap.release()
		cv2.destroyAllWindows() 

