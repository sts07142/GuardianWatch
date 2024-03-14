import cv2
import json
from collections import defaultdict
import torch

import warnings
warnings.filterwarnings("ignore")

from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)

from torchvision.transforms import Compose, Lambda
import os
import datetime
import argparse
from tqdm import tqdm
import numpy as np

from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slow_r50_detection

'''
    최종 저장물
    txt, mp4
'''
def make_parser():
    parser = argparse.ArgumentParser("behavior")

    parser.add_argument(
        "--path_info", nargs='+', help="Output_path_information"
    )

    return parser 

# # This method takes in an image and generates the bounding boxes for people in the image.
def get_person_bboxes(inp_img, predictor):
    predictions = predictor(inp_img.cpu().detach().numpy())['instances'].to('cpu')
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = np.array(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None)
    predicted_boxes = boxes[np.logical_and(classes==0, scores>0.75 )].tensor.cpu() # only person
    return predicted_boxes

def ava_inference_transform(
    clip, 
    boxes,
    num_frames = 4, #if using slowfast_r50_detection, change this to 32
    crop_size = 256, 
    data_mean = [0.45, 0.45, 0.45], 
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = None, #if using slowfast_r50_detection, change this to 4
):
    boxes = np.array(boxes)
    ori_boxes = boxes.copy()

    # Image [0, 255] -> [0, 1].

    # print(f"clip.shape : {clip.shape}") # clip.shape : torch.Size([3, 30, 1080, 1920])

    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0
    
    height, width = clip.shape[2], clip.shape[3]
    # print(f"height : {height} / width : {width}") # height : 1080 / width : 1920

    # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
    # range of [0, width] for x and [0,height] for y
    boxes = clip_boxes_to_image(boxes, height, width)

    # Resize short side to crop_size. Non-local and STRG uses 256.
    clip, boxes = short_side_scale_with_boxes(
        clip,
        size=crop_size,
        boxes=boxes,
    )
    
    # Normalize images by mean and std.
    clip = normalize(
        clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),
    )
    
    boxes = clip_boxes_to_image(
        boxes, clip.shape[2],  clip.shape[3]
    )
    
    # Incase of slowfast, generate both pathways
    if slow_fast_alpha is not None:
        fast_pathway = clip
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            clip,
            1,
            torch.linspace(
                0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha
            ).long(),
        )
        clip = [slow_pathway, fast_pathway]
    
    return clip, torch.from_numpy(boxes), ori_boxes

def main(args) :
    '''
        MOT_output_path : MOT 최종본 경로 (txt, mp4의 경로) /home2/korengc/Output/MOT/ch02/2023/10/14
        Action_output_path : Action 저장 경로 /home2/korengc/Output/Action/ch02/2023/10/14
        file_name : 파일명 (타임스탬프) 20231014233229
    '''
    print("behavior_prediction.py")

    path_list = args.path_info

    MOT_output_path = path_list[0]
    Action_output_path = path_list[1]
    file_name = path_list[9]

    device = 'cuda' # or 'cpu'
    model = slow_r50_detection(True) # Another option is slowfast_r50_detection
    model = model.eval().to(device)

    print("Finish Load Model")

    # 행동 예측 영상 Config
    video_path = MOT_output_path+"/"+file_name + ".mp4" # MOT 파일
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # action_output_video의 경로 -> /home2/korengc/Output/Action/ch02/2023/10/14/20231014233229.mp4
    output_path = Action_output_path+"/"+file_name + ".mp4"
    # print(output_path)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # MOT txt 파일 열기
    # /home2/korengc/Output/MOT/ch02/2023/10/14/20231014233229.txt

    with open(MOT_output_path+"/"+file_name+".txt", 'r') as f:
        bbox_info = f.readlines()

    # Convert bbox info to a useful format (a dictionary with frame_id as keys)
    bbox_dict = defaultdict(list)
    for line in bbox_info:
        values = line.strip().split(',')
        frame_id = int(values[0])
        obj_id = int(values[1])
        x, y, w, h = float(values[2]), float(values[3]), float(values[4]), float(values[5])
        current_time_str = values[7]
        bbox_dict[frame_id].append((obj_id, x, y, w, h, current_time_str))  

    label_map, _ = AvaLabeledVideoFramePaths.read_label_map('ava_action_list.pbtxt')
    
    clip_duration = 0.5  # 클립 길이 (0.5초)
    clip_frame_count = int(clip_duration * fps)  # 클립에 포함될 프레임 수

    frame_id = 0
    frame_list = []
    inp_imgs = torch.empty(3, 2, 4, 5)

    results=[]
    # result_label = "Unknown"
    result_label_list = ['Unknown'] * 256
    result_label_list_clip = ['Unknown'] * 256

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        if frame_id % 500 == 0:
            print(f"{frame_id}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        # print(f"frame_tensor.shape : {frame_tensor.shape}") # torch.Size([1, 3, 1080, 1920])
        frame_list.append(frame_tensor)

        if len(frame_list) == clip_frame_count:
            result_label_list_clip = ['Unknown'] * 256

            # print(f"frame : {frame_id}")
            inp_imgs = torch.cat(frame_list, dim=0)  # dim=1로 설정하여 프레임을 누적
            # print(inp_imgs.shape) # inp_imgs.shape : torch.Size([3, 30, 1080, 1920])
            
            inp_imgs = inp_imgs.permute(1, 0, 2, 3)
            # print(f"inp_imgs.shape : {inp_imgs.shape}") # inp_imgs.shape : torch.Size([3, 30, 1080, 1920])

            inp_image = torch.from_numpy(frame)
            # print(f"inp_image.shape : {inp_image.shape}") # inp_image.shape : torch.Size([1080, 1920, 3])
            
            bboxes = bbox_dict.get(frame_id, [])

            predicted_boxes = torch.tensor([[w, y, x + w, y + h] for _, x, y, w, h, _ in bboxes])
            frame_obj_id = torch.tensor([[obj_id] for obj_id, _, _, _, _, _ in bboxes])
            # print(frame_obj_id)
            # print(f"predicted_boxes.shape : {predicted_boxes.shape}") # predicted_boxes.shape : torch.Size([5, 4])

            inputs, inp_boxes, ori_boxes = ava_inference_transform(clip=inp_imgs, boxes=predicted_boxes.numpy())
            # print(f"inputs.shape : {inputs.shape}") # inputs.shape : torch.Size([3, 4, 256, 455])
            # print(f"inp_boxes.shape : {inp_boxes.shape}") # inp_boxes.shape : torch.Size([5, 4])

            inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
            if isinstance(inputs, list):
                inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
            else:
                inputs = inputs.unsqueeze(0).to(device)

            preds = model(inputs, inp_boxes.to(device))
            preds= preds.to('cpu')
            preds = torch.cat([torch.zeros(preds.shape[0],1), preds], dim=1)
            # print(f"preds.shape : {preds.shape}")

            top_scores, top_classes = [], []
            for pred in preds:
                mask = pred >= 0.3
                top_scores.append(pred[mask].tolist())
                top_class = torch.squeeze(torch.nonzero(mask), dim=-1).tolist()
                top_classes.append(top_class)

            # print(f"top_classes : {top_classes}")
            print(f"top_classes : {top_classes} / top_scores : {top_scores}")

            n_instances  = len(preds)
            for i in range(n_instances) : 
                labels = [label_map.get(c, "n/a") for c in top_classes[i]]
                if len(labels) == 0 :
                    labels.append("Unknown")
                result_label = "_".join(labels)
                # result_label = labels[0]
                # print(result_label)
                # print(f"result_label : {result_label}")

                for j in frame_obj_id : 
                    # print(j)
                    if result_label_list[j] == "Unknown" : 
                        result_label_list[j] = result_label
                        break

                    elif result_label_list[j] != "Unknown" :
                        continue
            
                # print(f"result_label_list : {result_label_list}")
                result_label_list_clip = result_label_list.copy()

            result_label_list = ['Unknown'] * 256
            frame_list = []  # frame_list 초기화
            

        bboxes = bbox_dict.get(frame_id, [])
        for obj_id, x, y, w, h, current_time_str in bboxes:
            results.append(f"{frame_id},{int(obj_id)},{result_label_list_clip[obj_id]},{current_time_str}\n")
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.putText(frame, result_label_list_clip[obj_id], (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        out.write(frame)
        
        frame_id += 1

    # 예측 결과 txt 저장 -> /home2/korengc/Output/Action/ch02/2023/10/14/20231014233229.txt
    with open(Action_output_path+"/"+file_name + ".txt", "w") as f:
        f.writelines(results)

    cap.release()
    out.release()

    motToTxt_command = f'python3 motToTxt.py --path_info {path_list[0]} {path_list[1]} {path_list[2]} {path_list[3]} {path_list[4]} {path_list[5]} {path_list[6]} {path_list[7]} {path_list[8]} {path_list[9]}'
    os.system(motToTxt_command)

if __name__ == "__main__" :
    args = make_parser().parse_args()
    main(args)