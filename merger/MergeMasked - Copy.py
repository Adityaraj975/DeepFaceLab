import traceback

import cv2
import numpy as np

from core import imagelib
from facelib import FaceType, LandmarksProcessor
from core.interact import interact as io
from core.cv2ex import *

xseg_input_size = 256

from numpy import linalg as npla
from skimage.transform import rescale
def mls_rigid_deformation_inv(image, p, q, alpha=1.0, density=1.0):
    ''' Rigid inverse deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width*density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height*density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]

    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha                     # [ctrls, grow, gcol]
    w[w == np.inf] = 2**31 - 1
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    qhat = reshaped_q - qstar                                                           # [ctrls, 2, grow, gcol]
    reshaped_phat1 = phat.reshape(ctrls, 1, 2, grow, gcol)                              # [ctrls, 1, 2, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 2, 1, grow, gcol)                              # [ctrls, 2, 1, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol)                               # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]

    mu = np.sum(np.matmul(reshaped_w.transpose(0, 3, 4, 1, 2) *
                          reshaped_phat1.transpose(0, 3, 4, 1, 2),
                          reshaped_phat2.transpose(0, 3, 4, 1, 2)), axis=0)             # [grow, gcol, 1, 1]
    reshaped_mu = mu.reshape(1, grow, gcol)                                             # [1, grow, gcol]
    neg_phat_verti = phat[:, [1, 0],...]                                                # [ctrls, 2, grow, gcol]
    neg_phat_verti[:, 1,...] = -neg_phat_verti[:, 1,...]
    reshaped_neg_phat_verti = neg_phat_verti.reshape(ctrls, 1, 2, grow, gcol)           # [ctrls, 1, 2, grow, gcol]
    mul_right = np.concatenate((reshaped_phat1, reshaped_neg_phat_verti), axis=1)       # [ctrls, 2, 2, grow, gcol]
    mul_left = reshaped_qhat * reshaped_w                                               # [ctrls, 1, 2, grow, gcol]
    Delta = np.sum(np.matmul(mul_left.transpose(0, 3, 4, 1, 2),
                             mul_right.transpose(0, 3, 4, 1, 2)),
                   axis=0).transpose(0, 1, 3, 2)                                        # [grow, gcol, 2, 1]
    Delta_verti = Delta[...,[1, 0],:]                                                   # [grow, gcol, 2, 1]
    Delta_verti[...,0,:] = -Delta_verti[...,0,:]
    B = np.concatenate((Delta, Delta_verti), axis=3)                                    # [grow, gcol, 2, 2]
    try:
        inv_B = np.linalg.inv(B)                                                        # [grow, gcol, 2, 2]
        flag = False
    except np.linalg.linalg.LinAlgError:
        flag = True
        det = np.linalg.det(B)                                                          # [grow, gcol]
        det[det < 1e-8] = np.inf
        reshaped_det = det.reshape(grow, gcol, 1, 1)                                    # [grow, gcol, 1, 1]
        adjoint = B[:,:,[[1, 0], [1, 0]], [[1, 1], [0, 0]]]                             # [grow, gcol, 2, 2]
        adjoint[:,:,[0, 1], [1, 0]] = -adjoint[:,:,[0, 1], [1, 0]]                      # [grow, gcol, 2, 2]
        inv_B = (adjoint / reshaped_det).transpose(2, 3, 0, 1)                          # [2, 2, grow, gcol]

    vqstar = reshaped_v - qstar                                                         # [2, grow, gcol]
    reshaped_vqstar = vqstar.reshape(1, 2, grow, gcol)                                  # [1, 2, grow, gcol]

    # Get final image transfomer -- 3-D array
    temp = np.matmul(reshaped_vqstar.transpose(2, 3, 0, 1),
                     inv_B).reshape(grow, gcol, 2).transpose(2, 0, 1)                   # [2, grow, gcol]
    norm_temp = np.linalg.norm(temp, axis=0, keepdims=True)                             # [1, grow, gcol]
    norm_vqstar = np.linalg.norm(vqstar, axis=0, keepdims=True)                         # [1, grow, gcol]
    transformers = temp / norm_temp * norm_vqstar + pstar                               # [2, grow, gcol]

    # Correct the points where pTwp is singular
    if flag:
        blidx = det == np.inf    # bool index
        transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image
    transformed_image = image[tuple(transformers.astype(np.int16))]    # [grow, gcol]

    # Rescale image
    transformed_image = rescale(transformed_image, scale=1.0 / density, mode='reflect')

    return transformed_image
    
def MergeMaskedFace (predictor_func, predictor_input_shape,
                     face_enhancer_func,
                     xseg_256_extract_func,
                     cfg, frame_info, img_bgr_uint8, img_bgr, img_face_landmarks):
    img_size = img_bgr.shape[1], img_bgr.shape[0]
    img_face_mask_a = LandmarksProcessor.get_image_hull_mask (img_bgr.shape, img_face_landmarks)


    out_img = img_bgr.copy()
    out_merging_mask_a = None

    input_size = predictor_input_shape[0]
    mask_subres_size = input_size*4
    output_size = input_size
    if cfg.super_resolution_power != 0:
        output_size *= 4

    face_mat        = LandmarksProcessor.get_transform_mat (img_face_landmarks, output_size, face_type=cfg.face_type)
    face_output_mat = LandmarksProcessor.get_transform_mat (img_face_landmarks, output_size, face_type=cfg.face_type, scale= 1.0 + 0.01*cfg.output_face_scale)

    if mask_subres_size == output_size:
        face_mask_output_mat = face_output_mat
    else:
        face_mask_output_mat = LandmarksProcessor.get_transform_mat (img_face_landmarks, mask_subres_size, face_type=cfg.face_type, scale= 1.0 + 0.01*cfg.output_face_scale)

    dst_face_bgr      = cv2.warpAffine( img_bgr        , face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC )
    dst_face_bgr      = np.clip(dst_face_bgr, 0, 1)

    dst_face_mask_a_0 = cv2.warpAffine( img_face_mask_a, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC )
    dst_face_mask_a_0 = np.clip(dst_face_mask_a_0, 0, 1)

    predictor_input_bgr      = cv2.resize (dst_face_bgr, (input_size,input_size) )

    predicted = predictor_func (predictor_input_bgr)
    prd_face_bgr          = np.clip (predicted[0], 0, 1.0)
    prd_face_mask_a_0     = np.clip (predicted[1], 0, 1.0)
    prd_face_dst_mask_a_0 = np.clip (predicted[2], 0, 1.0)

    if cfg.super_resolution_power != 0:
        prd_face_bgr_enhanced = face_enhancer_func(prd_face_bgr, is_tanh=True, preserve_size=False)
        mod = cfg.super_resolution_power / 100.0
        prd_face_bgr = cv2.resize(prd_face_bgr, (output_size,output_size))*(1.0-mod) + prd_face_bgr_enhanced*mod
        prd_face_bgr = np.clip(prd_face_bgr, 0, 1)

    if cfg.super_resolution_power != 0:
        prd_face_mask_a_0     = cv2.resize (prd_face_mask_a_0,      (output_size, output_size), cv2.INTER_CUBIC)
        prd_face_dst_mask_a_0 = cv2.resize (prd_face_dst_mask_a_0,  (output_size, output_size), cv2.INTER_CUBIC)

    if cfg.mask_mode == 1: #dst
        wrk_face_mask_a_0 = cv2.resize (dst_face_mask_a_0, (output_size,output_size), cv2.INTER_CUBIC)
    elif cfg.mask_mode == 2: #learned-prd
        wrk_face_mask_a_0 = prd_face_mask_a_0
    elif cfg.mask_mode == 3: #learned-dst
        wrk_face_mask_a_0 = prd_face_dst_mask_a_0
    elif cfg.mask_mode == 4: #learned-prd*learned-dst
        wrk_face_mask_a_0 = prd_face_mask_a_0*prd_face_dst_mask_a_0
    elif cfg.mask_mode == 5: #learned-prd+learned-dst
        wrk_face_mask_a_0 = np.clip( prd_face_mask_a_0+prd_face_dst_mask_a_0, 0, 1)
    elif cfg.mask_mode >= 6 and cfg.mask_mode <= 9:  #XSeg modes
        if cfg.mask_mode == 6 or cfg.mask_mode == 8 or cfg.mask_mode == 9:
            # obtain XSeg-prd
            prd_face_xseg_bgr = cv2.resize (prd_face_bgr, (xseg_input_size,)*2, cv2.INTER_CUBIC)
            prd_face_xseg_mask = xseg_256_extract_func(prd_face_xseg_bgr)
            X_prd_face_mask_a_0 = cv2.resize ( prd_face_xseg_mask, (output_size, output_size), cv2.INTER_CUBIC)

        if cfg.mask_mode >= 7 and cfg.mask_mode <= 9:
            # obtain XSeg-dst
            xseg_mat            = LandmarksProcessor.get_transform_mat (img_face_landmarks, xseg_input_size, face_type=cfg.face_type)
            dst_face_xseg_bgr   = cv2.warpAffine(img_bgr, xseg_mat, (xseg_input_size,)*2, flags=cv2.INTER_CUBIC )
            dst_face_xseg_mask  = xseg_256_extract_func(dst_face_xseg_bgr)
            X_dst_face_mask_a_0 = cv2.resize (dst_face_xseg_mask, (output_size,output_size), cv2.INTER_CUBIC)

        if cfg.mask_mode == 6:   #'XSeg-prd'
            wrk_face_mask_a_0 = X_prd_face_mask_a_0
        elif cfg.mask_mode == 7: #'XSeg-dst'
            wrk_face_mask_a_0 = X_dst_face_mask_a_0
        elif cfg.mask_mode == 8: #'XSeg-prd*XSeg-dst'
            wrk_face_mask_a_0 = X_prd_face_mask_a_0 * X_dst_face_mask_a_0
        elif cfg.mask_mode == 9: #learned-prd*learned-dst*XSeg-prd*XSeg-dst
            wrk_face_mask_a_0 = prd_face_mask_a_0 * prd_face_dst_mask_a_0 * X_prd_face_mask_a_0 * X_dst_face_mask_a_0

    wrk_face_mask_a_0[ wrk_face_mask_a_0 < (1.0/255.0) ] = 0.0 # get rid of noise

    # resize to mask_subres_size
    if wrk_face_mask_a_0.shape[0] != mask_subres_size:
        wrk_face_mask_a_0 = cv2.resize (wrk_face_mask_a_0, (mask_subres_size, mask_subres_size), cv2.INTER_CUBIC)

    # process mask in local predicted space
    if 'raw' not in cfg.mode:
        # add zero pad
        wrk_face_mask_a_0 = np.pad (wrk_face_mask_a_0, input_size)

        ero  = cfg.erode_mask_modifier
        blur = cfg.blur_mask_modifier

        if ero > 0:
            wrk_face_mask_a_0 = cv2.erode(wrk_face_mask_a_0, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ero,ero)), iterations = 1 )
        elif ero < 0:
            wrk_face_mask_a_0 = cv2.dilate(wrk_face_mask_a_0, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(-ero,-ero)), iterations = 1 )

        # clip eroded/dilated mask in actual predict area
        # pad with half blur size in order to accuratelly fade to zero at the boundary
        clip_size = input_size + blur // 2

        wrk_face_mask_a_0[:clip_size,:] = 0
        wrk_face_mask_a_0[-clip_size:,:] = 0
        wrk_face_mask_a_0[:,:clip_size] = 0
        wrk_face_mask_a_0[:,-clip_size:] = 0

        if blur > 0:
            blur = blur + (1-blur % 2)
            wrk_face_mask_a_0 = cv2.GaussianBlur(wrk_face_mask_a_0, (blur, blur) , 0)

        wrk_face_mask_a_0 = wrk_face_mask_a_0[input_size:-input_size,input_size:-input_size]

        wrk_face_mask_a_0 = np.clip(wrk_face_mask_a_0, 0, 1)

    img_face_mask_a = cv2.warpAffine( wrk_face_mask_a_0, face_mask_output_mat, img_size, np.zeros(img_bgr.shape[0:2], dtype=np.float32), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC )[...,None]
    img_face_mask_a = np.clip (img_face_mask_a, 0.0, 1.0)

    img_face_mask_a [ img_face_mask_a < (1.0/255.0) ] = 0.0 # get rid of noise

    if wrk_face_mask_a_0.shape[0] != output_size:
        wrk_face_mask_a_0 = cv2.resize (wrk_face_mask_a_0, (output_size,output_size), cv2.INTER_CUBIC)

    wrk_face_mask_a = wrk_face_mask_a_0[...,None]
    wrk_face_mask_area_a = wrk_face_mask_a.copy()
    wrk_face_mask_area_a[wrk_face_mask_area_a>0] = 1.0

    if cfg.mode == 'original':
        return img_bgr, img_face_mask_a

    elif 'raw' in cfg.mode:
        if cfg.mode == 'raw-rgb':
            out_img = cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, out_img, cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT )
            out_merging_mask_a = img_face_mask_a

        elif cfg.mode == 'raw-predict':
            out_img = prd_face_bgr
            out_merging_mask_a = wrk_face_mask_a

        out_img = np.clip (out_img, 0.0, 1.0 )
    else:
        #averaging [lenx, leny, maskx, masky] by grayscale gradients of upscaled mask
        ar = []
        for i in range(1, 10):
            maxregion = np.argwhere( img_face_mask_a > i / 10.0 )
            if maxregion.size != 0:
                miny,minx = maxregion.min(axis=0)[:2]
                maxy,maxx = maxregion.max(axis=0)[:2]
                lenx = maxx - minx
                leny = maxy - miny
                if min(lenx,leny) >= 4:
                    ar += [ [ lenx, leny]  ]

        if len(ar) > 0:

            if 'seamless' not in cfg.mode and cfg.color_transfer_mode != 0:
                if cfg.color_transfer_mode == 1: #rct
                    prd_face_bgr = imagelib.reinhard_color_transfer ( np.clip( prd_face_bgr*wrk_face_mask_area_a*255, 0, 255).astype(np.uint8),
                                                                      np.clip( dst_face_bgr*wrk_face_mask_area_a*255, 0, 255).astype(np.uint8), )

                    prd_face_bgr = np.clip( prd_face_bgr.astype(np.float32) / 255.0, 0.0, 1.0)
                elif cfg.color_transfer_mode == 2: #lct
                    prd_face_bgr = imagelib.linear_color_transfer (prd_face_bgr, dst_face_bgr)
                elif cfg.color_transfer_mode == 3: #mkl
                    prd_face_bgr = imagelib.color_transfer_mkl (prd_face_bgr, dst_face_bgr)
                elif cfg.color_transfer_mode == 4: #mkl-m
                    prd_face_bgr = imagelib.color_transfer_mkl (prd_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a)
                elif cfg.color_transfer_mode == 5: #idt
                    prd_face_bgr = imagelib.color_transfer_idt (prd_face_bgr, dst_face_bgr)
                elif cfg.color_transfer_mode == 6: #idt-m
                    prd_face_bgr = imagelib.color_transfer_idt (prd_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a)
                elif cfg.color_transfer_mode == 7: #sot-m
                    prd_face_bgr = imagelib.color_transfer_sot (prd_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a, steps=10, batch_size=30)
                    prd_face_bgr = np.clip (prd_face_bgr, 0.0, 1.0)
                elif cfg.color_transfer_mode == 8: #mix-m
                    prd_face_bgr = imagelib.color_transfer_mix (prd_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a)

            if cfg.mode == 'hist-match':
                hist_mask_a = np.ones ( prd_face_bgr.shape[:2] + (1,) , dtype=np.float32)

                if cfg.masked_hist_match:
                    hist_mask_a *= wrk_face_mask_area_a

                white =  (1.0-hist_mask_a)* np.ones ( prd_face_bgr.shape[:2] + (1,) , dtype=np.float32)

                hist_match_1 = prd_face_bgr*hist_mask_a + white
                hist_match_1[ hist_match_1 > 1.0 ] = 1.0

                hist_match_2 = dst_face_bgr*hist_mask_a + white
                hist_match_2[ hist_match_1 > 1.0 ] = 1.0

                prd_face_bgr = imagelib.color_hist_match(hist_match_1, hist_match_2, cfg.hist_match_threshold ).astype(dtype=np.float32)

            if 'seamless' in cfg.mode:
                #mask used for cv2.seamlessClone
                img_face_seamless_mask_a = None
                for i in range(1,10):
                    a = img_face_mask_a > i / 10.0
                    if len(np.argwhere(a)) == 0:
                        continue
                    img_face_seamless_mask_a = img_face_mask_a.copy()
                    img_face_seamless_mask_a[a] = 1.0
                    img_face_seamless_mask_a[img_face_seamless_mask_a <= i / 10.0] = 0.0
                    break

            out_img = cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, out_img, cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT )
            out_img = np.clip(out_img, 0.0, 1.0)

            if 'seamless' in cfg.mode:
                try:
                    #calc same bounding rect and center point as in cv2.seamlessClone to prevent jittering (not flickering)
                    l,t,w,h = cv2.boundingRect( (img_face_seamless_mask_a*255).astype(np.uint8) )
                    s_maskx, s_masky = int(l+w/2), int(t+h/2)
                    out_img = cv2.seamlessClone( (out_img*255).astype(np.uint8), img_bgr_uint8, (img_face_seamless_mask_a*255).astype(np.uint8), (s_maskx,s_masky) , cv2.NORMAL_CLONE )
                    out_img = out_img.astype(dtype=np.float32) / 255.0
                except Exception as e:
                    #seamlessClone may fail in some cases
                    e_str = traceback.format_exc()

                    if 'MemoryError' in e_str:
                        raise Exception("Seamless fail: " + e_str) #reraise MemoryError in order to reprocess this data by other processes
                    else:
                        print ("Seamless fail: " + e_str)

            cfg_mp = 0.3#todo cfg.motion_blur_power / 100.0
            
            ###
            shrink_res = output_size #512
            
            shrink_prd_face_dst_mask_a_0 = cv2.resize (prd_face_dst_mask_a_0,  (shrink_res, shrink_res), cv2.INTER_CUBIC)     
            
            shrink_blur_size = (shrink_res // 32)+1            
            shrink_blur_size += (1-shrink_blur_size % 2)
              
            # Feather the mask
            shrink_prd_face_dst_mask_a_0 = cv2.GaussianBlur(shrink_prd_face_dst_mask_a_0, (shrink_blur_size,shrink_blur_size) , 0)
            shrink_prd_face_dst_mask_a_0[shrink_prd_face_dst_mask_a_0 < 0.5] = 0.0
            shrink_prd_face_dst_mask_a_0[shrink_prd_face_dst_mask_a_0 >= 0.5] = 1.0
          
            
            cnts = cv2.findContours( shrink_prd_face_dst_mask_a_0.astype(np.uint8), cv2.RETR_LIST , cv2.CHAIN_APPROX_TC89_KCOS  )
            # Get the largest found contour
            cnt = sorted(cnts[0], key = cv2.contourArea, reverse = True)[0].squeeze()
            l,t = cnt.min(0)
            r,b = cnt.max(0)
            min_dist_to_edge = min(l,t,r,b)
    
            center = np.mean(cnt,0)
            cnt2 = cnt.copy().astype(np.float32)
            cnt2_c = center - cnt2    
            cnt2_len = npla.norm(cnt2_c, axis=1, keepdims=True)
            cnt2_vec = cnt2_c / cnt2_len
            # Anchor perimeter
            pts_count = shrink_res // 2
            
            h=shrink_res
            w=shrink_res
            perim_pts = np.concatenate (
                           (np.concatenate ( [ np.arange(0,w+w/pts_count, w/pts_count)[...,None], np.array ( [[0]]*(pts_count+1) ) ], axis=-1 ),
                            np.concatenate ( [ np.arange(0,w+w/pts_count, w/pts_count)[...,None], np.array ( [[h]]*(pts_count+1) ) ], axis=-1 ),
                            np.concatenate ( [ np.array ( [[0]]*(pts_count+1) ), np.arange(0,h+h/pts_count, h/pts_count)[...,None] ], axis=-1 ),
                            np.concatenate ( [ np.array ( [[w]]*(pts_count+1) ), np.arange(0,h+h/pts_count, h/pts_count)[...,None] ], axis=-1 ) ), 0 ).astype(np.int32)


            cnt2 += cnt2_vec *  cnt2_len * cfg_mp #todo
            
            cnt2 = cnt2.astype(np.int32)
            cnt2 = np.concatenate ( (cnt2, perim_pts), 0 )
            cnt = np.concatenate ( (cnt, perim_pts), 0 )
    
            shrink_face_mat     = LandmarksProcessor.get_transform_mat (img_face_landmarks, shrink_res, face_type=cfg.face_type)#todo check face type
            shrink_dst_face_bgr = cv2.warpAffine( img_bgr, shrink_face_mat, (shrink_res, shrink_res), flags=cv2.INTER_CUBIC )
   
            shrinked_dst_face_bgr = mls_rigid_deformation_inv( shrink_dst_face_bgr, cnt, cnt2 )
   
            new_img_bgr = cv2.warpAffine( shrinked_dst_face_bgr, shrink_face_mat, img_size, img_bgr.copy(), cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT )
            
            shrink_ero_size = int(min_dist_to_edge*0.9) #todo
            shrink_ero_size += (1-shrink_ero_size % 2)
            shrink_blur_size = int(shrink_ero_size * 1.5)
            shrink_blur_size += (1-shrink_blur_size % 2)
            
            if shrink_ero_size != 0:
                shrink_prd_face_dst_mask_a_0_before = shrink_prd_face_dst_mask_a_0.copy()
                shrink_prd_face_dst_mask_a_0 = cv2.dilate(shrink_prd_face_dst_mask_a_0, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(shrink_ero_size,shrink_ero_size)), iterations = 1 )            
                shrink_prd_face_dst_mask_a_0 = cv2.GaussianBlur(shrink_prd_face_dst_mask_a_0, ( shrink_blur_size,shrink_blur_size) , 0)
            
                #while True:
                #    cv2.imshow("", (shrink_prd_face_dst_mask_a_0_before*255).astype(np.uint8) )
                #    cv2.waitKey(0)
                #    cv2.imshow("", (shrink_prd_face_dst_mask_a_0*255).astype(np.uint8) )
                #    cv2.waitKey(0)
                
            shrink_img_mask = cv2.warpAffine( shrink_prd_face_dst_mask_a_0, shrink_face_mat, img_size, img_bgr.copy(), cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT )
            shrink_img_mask_a = shrink_img_mask[...,None]
            
            new_img_bgr = img_bgr*(1-shrink_img_mask_a) + (new_img_bgr*shrink_img_mask_a)
            
            #cv2.imshow("", (shrink_dst_face_bgr*255).astype(np.uint8) )
            #cv2.waitKey(0)
            #cv2.imshow("", (shrinked_dst_face_bgr*255).astype(np.uint8) )
            #cv2.waitKey(0)
            
            while True:
                cv2.imshow("", (img_bgr*255).astype(np.uint8) )
                cv2.waitKey(0)
                cv2.imshow("", (new_img_bgr*255).astype(np.uint8) )
                cv2.waitKey(0)
                #cv2.imshow("", (shrink_img_mask*255).astype(np.uint8) )
                #cv2.waitKey(0)
                
            
                
            ###

            out_img = img_bgr*(1-img_face_mask_a) + (out_img*img_face_mask_a)

            

            if ('seamless' in cfg.mode and cfg.color_transfer_mode != 0) or \
               cfg.mode == 'seamless-hist-match' or \
               cfg_mp != 0 or \
               cfg.blursharpen_amount != 0 or \
               cfg.image_denoise_power != 0 or \
               cfg.bicubic_degrade_power != 0:
                
                out_face_bgr = cv2.warpAffine( out_img, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC )

                if 'seamless' in cfg.mode and cfg.color_transfer_mode != 0:
                    if cfg.color_transfer_mode == 1:
                        out_face_bgr = imagelib.reinhard_color_transfer ( np.clip(out_face_bgr*wrk_face_mask_area_a*255, 0, 255).astype(np.uint8),
                                                                        np.clip(dst_face_bgr*wrk_face_mask_area_a*255, 0, 255).astype(np.uint8) )
                        out_face_bgr = np.clip( out_face_bgr.astype(np.float32) / 255.0, 0.0, 1.0)
                    elif cfg.color_transfer_mode == 2: #lct
                        out_face_bgr = imagelib.linear_color_transfer (out_face_bgr, dst_face_bgr)
                    elif cfg.color_transfer_mode == 3: #mkl
                        out_face_bgr = imagelib.color_transfer_mkl (out_face_bgr, dst_face_bgr)
                    elif cfg.color_transfer_mode == 4: #mkl-m
                        out_face_bgr = imagelib.color_transfer_mkl (out_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a)
                    elif cfg.color_transfer_mode == 5: #idt
                        out_face_bgr = imagelib.color_transfer_idt (out_face_bgr, dst_face_bgr)
                    elif cfg.color_transfer_mode == 6: #idt-m
                        out_face_bgr = imagelib.color_transfer_idt (out_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a)
                    elif cfg.color_transfer_mode == 7: #sot-m
                        out_face_bgr = imagelib.color_transfer_sot (out_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a, steps=10, batch_size=30)
                        out_face_bgr = np.clip (out_face_bgr, 0.0, 1.0)
                    elif cfg.color_transfer_mode == 8: #mix-m
                        out_face_bgr = imagelib.color_transfer_mix (out_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a)

                if cfg.mode == 'seamless-hist-match':
                    out_face_bgr = imagelib.color_hist_match(out_face_bgr, dst_face_bgr, cfg.hist_match_threshold)

                if cfg_mp != 0:
                    k_size = int(frame_info.motion_power*cfg_mp)
                    if k_size >= 1:
                        k_size = np.clip (k_size+1, 2, 50)
                        if cfg.super_resolution_power != 0:
                            k_size *= 2
                        out_face_bgr = imagelib.LinearMotionBlur (out_face_bgr, k_size , frame_info.motion_deg)

                if cfg.blursharpen_amount != 0:
                    out_face_bgr = imagelib.blursharpen ( out_face_bgr, cfg.sharpen_mode, 3, cfg.blursharpen_amount)

                if cfg.image_denoise_power != 0:
                    n = cfg.image_denoise_power
                    while n > 0:
                        img_bgr_denoised = cv2.medianBlur(img_bgr, 5)
                        if int(n / 100) != 0:
                            img_bgr = img_bgr_denoised
                        else:
                            pass_power = (n % 100) / 100.0
                            img_bgr = img_bgr*(1.0-pass_power)+img_bgr_denoised*pass_power
                        n = max(n-10,0)

                if cfg.bicubic_degrade_power != 0:
                    p = 1.0 - cfg.bicubic_degrade_power / 101.0
                    img_bgr_downscaled = cv2.resize (img_bgr, ( int(img_size[0]*p), int(img_size[1]*p ) ), cv2.INTER_CUBIC)
                    img_bgr = cv2.resize (img_bgr_downscaled, img_size, cv2.INTER_CUBIC)

                new_out = cv2.warpAffine( out_face_bgr, face_mat, img_size, img_bgr.copy(), cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT )
                out_img =  np.clip( img_bgr*(1-img_face_mask_a) + (new_out*img_face_mask_a) , 0, 1.0 )


            if cfg.color_degrade_power != 0:
                out_img_reduced = imagelib.reduce_colors(out_img, 256)
                if cfg.color_degrade_power == 100:
                    out_img = out_img_reduced
                else:
                    alpha = cfg.color_degrade_power / 100.0
                    out_img = (out_img*(1.0-alpha) + out_img_reduced*alpha)

        out_merging_mask_a = img_face_mask_a

    return out_img, out_merging_mask_a


def MergeMasked (predictor_func,
                 predictor_input_shape,
                 face_enhancer_func,
                 xseg_256_extract_func,
                 cfg,
                 frame_info):
    img_bgr_uint8 = cv2_imread(frame_info.filepath)
    img_bgr_uint8 = imagelib.normalize_channels (img_bgr_uint8, 3)
    img_bgr = img_bgr_uint8.astype(np.float32) / 255.0

    outs = []
    for face_num, img_landmarks in enumerate( frame_info.landmarks_list ):
        out_img, out_img_merging_mask = MergeMaskedFace (predictor_func, predictor_input_shape, face_enhancer_func, xseg_256_extract_func, cfg, frame_info, img_bgr_uint8, img_bgr, img_landmarks)
        outs += [ (out_img, out_img_merging_mask) ]

    #Combining multiple face outputs
    final_img = None
    final_mask = None
    for img, merging_mask in outs:
        h,w,c = img.shape

        if final_img is None:
            final_img = img
            final_mask = merging_mask
        else:
            final_img = final_img*(1-merging_mask) + img*merging_mask
            final_mask = np.clip (final_mask + merging_mask, 0, 1 )

    final_img = np.concatenate ( [final_img, final_mask], -1)

    return (final_img*255).astype(np.uint8)