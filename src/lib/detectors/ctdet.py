from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import pickle

from external.nms import soft_nms
from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector

class CtdetDetector(BaseDetector):
  def __init__(self, opt, hm_store_root):
    super(CtdetDetector, self).__init__(opt)
    # self.corner_store_root = '/home/ridhwan/storage/ridhwan/hm/'
    # self.corner_store_root = '/media/ridhwan/41b91e9e-9e35-4b55-9fd9-5c569c51d214/detection_datasets/hm/'
    self.hm_store_root = hm_store_root
    self.store_distance = 5

  def process(self, images, return_time=False):
    with torch.no_grad():
      output = self.model(images)[-1]

      # pickle_id = ((self.store_distance - 1) - (self.centernet_img_id%self.store_distance)) + self.centernet_img_id
      # corner_file = pickle.load(open( "{0}centernethm_{1}.p".format(self.corner_store_root, pickle_id), "rb" ) )
      # hm = corner_file['nms_hm'][self.centernet_img_id % 5].unsqueeze(0)
      # hm = corner_file['hm'][self.centernet_img_id % 5].unsqueeze(0).sigmoid_()

      hm = output['hm'].sigmoid_()
      wh = output['wh']
      reg = output['reg'] if self.opt.reg_offset else None
      if self.opt.flip_test:
        hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
        wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
        reg = reg[0:1] if reg is not None else None
      torch.cuda.synchronize()
      forward_time = time.time()
      dets = ctdet_decode(hm, wh, reg=reg, K=self.opt.K)

    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes)
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      dets[0][j][:, :4] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      if len(self.scales) > 1 or self.opt.nms:
         soft_nms(results[j], Nt=0.5, method=2)
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      pickle_id = ((self.store_distance - 1) - (self.centernet_img_id%self.store_distance)) + self.centernet_img_id
      corner_file = pickle.load(open( "{0}cornercenternethm_{1}.p".format(self.corner_store_root, pickle_id), "rb" ) )
      # img1 = pickle.load( open( "/home/ridhwan/cornercenternet/image.p", "rb" ) )
      # print(self.centernet_img_id)
      # print(type(corner_file['img']), len(corner_file['img']))
      if corner_file['img']:
        corner_img = corner_file['img'][self.centernet_img_id % 5]
        corner_img = corner_img.cpu().numpy().transpose(1,2,0)
        corner_img = ((corner_img * self.std + self.mean) * 255).astype(np.uint8)
      else:
        corner_img = img

      # corner_hm = corner_file['hm'][self.centernet_img_id].sigmoid_()
      # corner_hm = corner_hm.cpu().numpy().squeeze()
      corner_hm = corner_file['nms_hm'][self.centernet_img_id % 5].detach().cpu().numpy()
      # corner_hm = corner_hm.cpu().numpy().squeeze()

      from skimage.measure import compare_ssim as ssim
      # import numpy as np
      # print(img2.shape, corner_img.shape)
      # print(img2.max(), corner_img.max())
      # print("comparing the images: ", )
      assert ssim(img, corner_img, multichannel=True) >= 0.999


      # img = img.cpu().numpy().transpose(0,2,3,1).squeeze()


      corner_hm = debugger.gen_colormap(corner_hm, output_res=(corner_img.shape[0], corner_img.shape[1]))
      debugger.add_blend_img(corner_img, corner_hm, 'corner_hm_{:.1f}'.format(scale))
      # debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      # for k in range(len(dets[i])):
      #   if detection[i, k, 4] > self.opt.center_thresh:
      #     debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
      #                            detection[i, k, 4],
      #                            img_id='out_pred_{:.1f}'.format(scale))

  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='ctdet')
    for j in range(1, self.num_classes + 1):
      for bbox in results[j]:
        if bbox[4] > self.opt.vis_thresh:
          debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
    debugger.show_all_imgs(pause=self.pause)
