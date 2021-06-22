import multiprocessing
import traceback
from enum import IntEnum

import cv2
import numpy as np

from core import imagelib, pathex
from core.imagelib import sd
from core.cv2ex import *
from core.interact import interact as io
from core.joblib import Subprocessor, SubprocessGenerator, ThisThreadGenerator
from samplelib import SampleGeneratorBase


class SampleGenerator(SampleGeneratorBase):
    def __init__ (self, images_path, debug=False, batch_size=1, resolution=256, generators_count=4, **kwargs):
        super().__init__(debug, batch_size)
        self.initialized = False
                
        images_paths = pathex.get_image_paths(images_path)
        

        if self.debug:
            self.generators_count = 1
        else:
            self.generators_count = max(1, generators_count)

        if self.debug:
            self.generators = [ThisThreadGenerator ( self.batch_func, (images_paths, resolution) )]
        else:
            self.generators = [SubprocessGenerator ( self.batch_func, (images_paths, resolution), start_now=False ) \
                               for i in range(self.generators_count) ]

            #SubprocessGenerator.start_in_parallel( self.generators )

        self.generator_counter = -1

        self.initialized = True

    #overridable
    def is_initialized(self):
        return self.initialized

    def __iter__(self):
        return self

    def __next__(self):
        self.generator_counter += 1
        generator = self.generators[self.generator_counter % len(self.generators) ]
        return next(generator)

    def batch_func(self, param ):
        images_paths, resolution = param
        images_paths_len = len(images_paths)
        
        sample_idxs = np.arange(0, images_paths_len).tolist()
        
        
        shuffle_idxs = []

        rotation_range=[-45,45]
        scale_range=[-0.5, 0.5]
        tx_range=[-0.25, 0.25]
        ty_range=[-0.25, 0.25]
        random_bilinear_resize_chance, random_bilinear_resize_max_size_per = 25,75
        motion_blur_chance, motion_blur_mb_max_size = 25, 5
        gaussian_blur_chance, gaussian_blur_kernel_max_size = 25, 5

        bs = self.batch_size
        
        img_mean = np.array([0.406, 0.456, 0.485], np.float32)
        img_std  = np.array([0.225, 0.224, 0.229], np.float32)
                    
        while True:
            batches = [ [], [] ]

            n_batch = 0
            while n_batch < bs:
                try:
                    if len(shuffle_idxs) == 0:
                        shuffle_idxs = sample_idxs.copy()
                        np.random.shuffle(shuffle_idxs)
                        
                    path = images_paths[shuffle_idxs.pop()]
                    img = cv2_imread(path)
                    img = img.astype(np.float32) / 255.0
                    img = cv2.resize(img, (resolution,resolution), interpolation=cv2.INTER_CUBIC)
                    
                    if np.random.randint(2) == 0:
                        img = imagelib.apply_random_hsv_shift(img, mask=sd.random_circle_faded ([resolution,resolution]))
                    else:
                        img = imagelib.apply_random_rgb_levels(img, mask=sd.random_circle_faded ([resolution,resolution]))

                    img = imagelib.apply_random_motion_blur( img, motion_blur_chance, motion_blur_mb_max_size, mask=sd.random_circle_faded ([resolution,resolution]))
                    img = imagelib.apply_random_gaussian_blur( img, gaussian_blur_chance, gaussian_blur_kernel_max_size, mask=sd.random_circle_faded ([resolution,resolution]))
                    img = imagelib.apply_random_bilinear_resize( img, random_bilinear_resize_chance, random_bilinear_resize_max_size_per, mask=sd.random_circle_faded ([resolution,resolution]))
                    
                    warp_params = imagelib.gen_warp_params(resolution, False, rotation_range=rotation_range, scale_range=scale_range, tx_range=tx_range, ty_range=ty_range )
                    img   = imagelib.warp_by_params (warp_params, img,  can_warp=True, can_transform=True, can_flip=False, border_replicate=True)
                    #img   = imagelib.warp_by_params (warp_params, img,  can_warp=False, can_transform=True, can_flip=False, border_replicate=True)

                    mat = warp_params['umat']           

                    img = np.clip(img, 0, 1)
                    
                    img = (img-img_mean) / img_std

                    batches[0].append ( img.transpose((2,0,1)) )
                    batches[1].append ( mat.reshape( (6,) ).astype(np.float32) )
                    

                    n_batch += 1
                except:
                    io.log_err ( traceback.format_exc() )
                    
            yield [ np.array(batch) for batch in batches]
          