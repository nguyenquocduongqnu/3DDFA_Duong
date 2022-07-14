# coding: utf-8

__author__ = 'cleardusk'

import logging
import sys
import argparse
import time
import io
import pickle
import numpy as np
import yaml
from PIL import Image
import json
from types import SimpleNamespace

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from post_process import post_process, export
import pickle
from utils.render import render
# from utils.render_ctypes import render  # faster
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose
from utils.serialization import ser_to_ply, ser_to_obj, get_colors
from utils.functions import draw_landmarks, get_suffix
from utils.app_api import *
from utils.tddfa_util import str2bool
import trimesh

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
						level=logging.DEBUG, filename='logs/output_logs.txt', filemode='w')
logging.getLogger("urllib3").propagate = False
def main(args):
    #Get data from app
    data = get_data(args.api.url,args.api.token, args.api.last_id)
##    while data is None:
##        for s in range(5):
##            print(f"Connection refused...Try again in {s}", flush=True, end="\r")
##            time.sleep(1)
##            s -= 1
##        data = get_data(args.api.url,args.api.token, args.api.last_id)
    if len(data) == 0:
        return

    # Given a still image path and load to BGR channel
    for image in data:
        files = []
        print(image)
        img = url2img(image['image'])
        # Detect faces, get 3DMM params and roi boxes
        boxes = face_boxes(img)
        n = len(boxes)
        if n == 0:
            print(f'No face detected, exit')
            args.api.last_id = image['id']
            json.dump(args,open('config.json','w'), default=lambda x: x.__dict__)
            files.append(('error', 'No face detected'))
            resp = send_data(args.api.url, args.api.token, image['id'], files)
            continue
            # sys.exit(-1)
        print(f'Detect {n} faces')

        param_lst, roi_box_lst = tddfa(img, boxes)

        # Visualization and serialization
        dense_flag = args.output_option in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
        img_name = image['image'].split("/")[-1]
        old_suffix = img_name.split(".")[-1]
        new_suffix = f'.{args.output_option}' if args.output_option in ('ply', 'obj') else '.jpg'
        obj_path = f'examples/results/{img_name.replace(f".{old_suffix}", f"{new_suffix}")}'
        scene_path =  f'examples/results/{img_name.replace(f".{old_suffix}", ".png")}'

        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)

        vertices, vertex_colors, faces  = ser_to_obj(img, ver_lst, tddfa.tri, img.shape[0])

        obj_lst = post_process(vertices, vertex_colors, faces)

        obj_lst = export(obj_lst, obj_path, scene_path)

        # face_obj = trimesh.Trimesh(vertices=face_obj_o3d.vertices, vertex_colors=face_obj_o3d.vertex_colors, faces=face_obj_o3d.triangles)

        # face_obj.export(obj_path)
        # scene = face_obj.scene().save_image(visible=False)
        # scene_img = Image.open(trimesh.util.wrap_as_stream(scene))
        # scene_img.save(scene_path)
        files_mng = OpenFiles()

        for obj in obj_lst:
            files.append(('objs[]',files_mng.open(obj['obj_path'],'rb')))
            files.append(('stls[]',''))
            files.append(('thumbnails[]',files_mng.open(obj['img_path'],'rb')))
        resp = send_data(args.api.url,args.api.token,image['id'],files)
        files_mng.close()
        if not resp:
            for s in range(5):
                print(f"Connection refused...Try again in {s}", flush=True, end="\r")
                time.sleep(1)
                s -= 1
        elif resp['result']:
            args.api.last_id = image['id']
            json.dump(args,open('config.json','w'), default=lambda x: x.__dict__)
            for obj in obj_lst:
                os.remove(obj['obj_path'])
                os.remove(obj['img_path'])
        else:
            print(resp['message'])

if __name__ == '__main__':
    args = json.load(open("config.json", "r"), object_hook=lambda d: SimpleNamespace(**d))
    cfg = yaml.load(open(args.model_config), Loader=yaml.SafeLoader)
    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os

        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()
    while True:
        args = json.load(open("config.json", "r"), object_hook=lambda d: SimpleNamespace(**d))
        try:
            main(args)
            print('Waiting for input...', flush=True, end="\r")
            time.sleep(2)
        except Exception as ex:
            logging.error('FAILED', exc_info=True)
            break
