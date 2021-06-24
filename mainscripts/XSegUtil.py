import json
import shutil
import traceback
from pathlib import Path
from PIL import Image
import dlib
import cv2
import numpy as np
import time 
from core import pathex
from core.cv2ex import *
from core.interact import interact as io
from core.leras import nn
from DFLIMG import *
from facelib import XSegNet, LandmarksProcessor, FaceType
import pickle

def apply_xseg(input_path, model_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')

    if not model_path.exists():
        raise ValueError(f'{model_path} not found. Please ensure it exists.')
    
    face_type = None
    
    model_dat = model_path / 'XSeg_data.dat'
    if model_dat.exists():
        dat = pickle.loads( model_dat.read_bytes() )
        dat_options = dat.get('options', None)
        if dat_options is not None:
            face_type = dat_options.get('face_type', None)
        
        
        
    if face_type is None:
        face_type = io.input_str ("XSeg model face type", 'same', ['h','mf','f','wf','head','same'], help_message="Specify face type of trained XSeg model. For example if XSeg model trained as WF, but faceset is HEAD, specify WF to apply xseg only on WF part of HEAD. Default is 'same'").lower()
        if face_type == 'same':
            face_type = None
    
    if face_type is not None:
        face_type = {'h'  : FaceType.HALF,
                     'mf' : FaceType.MID_FULL,
                     'f'  : FaceType.FULL,
                     'wf' : FaceType.WHOLE_FACE,
                     'head' : FaceType.HEAD}[face_type]
                     
    io.log_info(f'Applying trained XSeg model to {input_path.name}/ folder.')

    device_config = nn.DeviceConfig.ask_choose_device(choose_only_one=True)
    nn.initialize(device_config)
        
    
    
    #xseg = XSegNet(name='XSeg', 
    #                load_weights=True,
    #                weights_file_root=model_path,
    #                data_format=nn.data_format,
    #                raise_on_no_model_files=True)
    xseg_res = 256
              
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    face_detector = dlib.get_frontal_face_detector()
    #http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    file_path = "DeepFaceLab/data/Xseg/shape_predictor_81_face_landmarks.dat"
    #file_path = "E:\DeepFaceLab_DirectX12 testing 1006\_internal\DeepFaceLab\data\Xseg\shape_predictor_81_face_landmarks.dat"
    landmark_detector = dlib.shape_predictor(file_path)
    
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        #t0 = time.time()          

        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} is not a DFLIMG')
            continue
        imgg = cv2_imread(filepath)
        img = cv2_imread(filepath).astype(np.float32) / 255.0
        h,w,c = img.shape
    
        img_face_type = FaceType.fromString( dflimg.get_face_type() )
        if face_type is not None and img_face_type != face_type:
            lmrks = dflimg.get_source_landmarks()
            
            fmat = LandmarksProcessor.get_transform_mat(lmrks, w, face_type)
            imat = LandmarksProcessor.get_transform_mat(lmrks, w, img_face_type)
            
            g_p = LandmarksProcessor.transform_points (np.float32([(0,0),(w,0),(0,w) ]), fmat, True)
            g_p2 = LandmarksProcessor.transform_points (g_p, imat)
            
            mat = cv2.getAffineTransform( g_p2, np.float32([(0,0),(w,0),(0,w) ]) )
            
            img = cv2.warpAffine(img, mat, (w, w), cv2.INTER_LANCZOS4)
            img = cv2.resize(img, (xseg_res, xseg_res), interpolation=cv2.INTER_LANCZOS4)
        else:
            if w != xseg_res:
                img = cv2.resize( img, (xseg_res,xseg_res), interpolation=cv2.INTER_LANCZOS4 )    
                    
        if len(img.shape) == 2:
            img = img[...,None]     

         
        #imgg = dlib.load_rgb_image(filepath)
        imgg = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)

        faces = face_detector(imgg, 1)
        landmark_tuple = []
        for k, d in enumerate(faces):
           landmarks = landmark_detector(imgg, d)
           for n in range(0, 81):
              x = landmarks.part(n).x
              y = landmarks.part(n).y
              landmark_tuple.append((x, y))
              #cv2.circle(img, (x, y), 2, (255, 255, 0), -1)
        routes = []
        if(len(landmark_tuple)==0):
            continue
        for i in range(15, -1, -1):
            from_coordinate = landmark_tuple[i+1]
            to_coordinate = landmark_tuple[i]
            routes.append(from_coordinate)

        from_coordinate = landmark_tuple[0]
        to_coordinate = landmark_tuple[17]
        routes.append(from_coordinate)
        from_coordinate = landmark_tuple[75]
        routes.append(from_coordinate)
        from_coordinate = landmark_tuple[76]
        routes.append(from_coordinate)
        from_coordinate = landmark_tuple[68]
        routes.append(from_coordinate)
        from_coordinate = landmark_tuple[69]
        routes.append(from_coordinate)
        from_coordinate = landmark_tuple[72]
        routes.append(from_coordinate)
        from_coordinate = landmark_tuple[73]
        routes.append(from_coordinate)
        from_coordinate = landmark_tuple[79]
        routes.append(from_coordinate)
        from_coordinate = landmark_tuple[74]
        routes.append(from_coordinate)
        #from_coordinate = landmark_tuple[78]
        #routes.append(from_coordinate)
        to_coordinate = landmark_tuple[16]
        routes.append(to_coordinate)
        for i in range(0, len(routes)-1):
            from_coordinate = routes[i]
            to_coordinate = routes[i+1]
            #img = cv2.line(img, from_coordinate, to_coordinate, (255, 255, 0), 1)
        mask = np.zeros((img.shape[0], img.shape[1]))
        mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
        mask = mask.astype('float32')
        #t1 = time.time() - t0
        #print("Time elapsed : ", t1)

        if face_type is not None and img_face_type != face_type:
            mask = cv2.resize(mask, (w, w), interpolation=cv2.INTER_LANCZOS4)
            mask = cv2.warpAffine( mask, mat, (w,w), np.zeros( (h,w,c), dtype=np.float), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4)
            mask = cv2.resize(mask, (xseg_res, xseg_res), interpolation=cv2.INTER_LANCZOS4)
        else :
            mask = cv2.resize(mask, (xseg_res, xseg_res), interpolation=cv2.INTER_LANCZOS4)
        
        mask[mask < 0.5]=0
        mask[mask >= 0.5]=1    
        dflimg.set_xseg_mask(mask)
        dflimg.save()


        
def fetch_xseg(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')
    
    output_path = input_path.parent / (input_path.name + '_xseg')
    output_path.mkdir(exist_ok=True, parents=True)
    
    io.log_info(f'Copying faces containing XSeg polygons to {output_path.name}/ folder.')
    
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    
    files_copied = []
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} is not a DFLIMG')
            continue
        
        ie_polys = dflimg.get_seg_ie_polys()

        if ie_polys.has_polys():
            files_copied.append(filepath)
            shutil.copy ( str(filepath), str(output_path / filepath.name) )
    
    io.log_info(f'Files copied: {len(files_copied)}')
    
    is_delete = io.input_bool (f"\r\nDelete original files?", True)
    if is_delete:
        for filepath in files_copied:
            Path(filepath).unlink()
            
    
def remove_xseg(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')
    
    io.log_info(f'Processing folder {input_path}')
    io.log_info('!!! WARNING : APPLIED XSEG MASKS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : APPLIED XSEG MASKS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : APPLIED XSEG MASKS WILL BE REMOVED FROM THE FRAMES !!!')
    io.input_str('Press enter to continue.')
                               
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    files_processed = 0
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} is not a DFLIMG')
            continue
        
        if dflimg.has_xseg_mask():
            dflimg.set_xseg_mask(None)
            dflimg.save()
            files_processed += 1
    io.log_info(f'Files processed: {files_processed}')
    
def remove_xseg_labels(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')
    
    io.log_info(f'Processing folder {input_path}')
    io.log_info('!!! WARNING : LABELED XSEG POLYGONS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : LABELED XSEG POLYGONS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : LABELED XSEG POLYGONS WILL BE REMOVED FROM THE FRAMES !!!')
    io.input_str('Press enter to continue.')
    
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    files_processed = 0
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} is not a DFLIMG')
            continue

        if dflimg.has_seg_ie_polys():
            dflimg.set_seg_ie_polys(None)
            dflimg.save()            
            files_processed += 1
            
    io.log_info(f'Files processed: {files_processed}')
