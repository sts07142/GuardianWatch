import cv2
import json
from collections import defaultdict
import torch
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    ShortSideScale,
    UniformTemporalSubsample
)
from torchvision.transforms import Compose, Lambda
import os
import datetime
import argparse
from tqdm import tqdm

from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slow_r50_detection

'''
    최종 저장물
txt, mp4

'''
def make_parser():
    parser = argparse.ArgumentParser("behavior")

    parser.add_argument(
            "--MOT_output_path", default=None, help="MOT_output_path"
        )

    parser.add_argument(
            "--Action_output_path", default=None, help="Action_output_path"
        )

    parser.add_argument(
            "--file_name", default=None, help="file_name"
        )

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
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0

    height, width = clip.shape[2], clip.shape[3]
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
    file_name = path_list[8]

    # 행동 예측 모델 Load
    # model = torch.hub.load('SLOW_8x8_R50.pyth', 'slow_r50', weights=True)
    # model = torch.load('')
    # model = model.to('cuda')
    # model.eval()

    # path = "/home2/korengc/.cache/torch/hub/checkpoints/SLOWFAST_8x8_R50.pyth"
    # path = "/home2/korengc/.cache/torch/hub/checkpoints/AVAv2.2_SLOW_4x16_R50_DETECTION.pyth"
    # model = create_slowfast(model_num_class=400)
    # model.load_state_dict(torch.load(path, map_location=torch.device('cuda'))['model_state'])
    # slowfast_model.eval()

    # model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50_detection', pretrained=True)
    # model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50_detection', pretrained=True)

    # model.eval()

    device = 'cuda' # or 'cpu'
    model = slow_r50_detection(True) # Another option is slowfast_r50_detection
    model = model.eval().to(device)

    print("Finish Load Model")


    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55  # set threshold for this model
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    # predictor = DefaultPredictor(cfg)

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

    # Load class names
    with open("kinetics_classnames.json", "r") as f:
        kinetics_classnames_json = json.load(f)

    kinetics_classnames = {}
    for k, v in kinetics_classnames_json.items():
        kinetics_classnames[v] = str(k).replace('"', "")        


    label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map('ava_action_list.pbtxt')

    # To store past frames and bounding boxes for each object
    stored_frames = defaultdict(list)
    stored_bboxes = defaultdict(list)

    results=[]

    frame_id = 0
    while cap.isOpened():

        if frame_id % 1000 == 0:
            print(frame_id)

        ret, frame = cap.read()
        if not ret:
            break
        
        bboxes = bbox_dict.get(frame_id, [])

        for obj_id, x, y, w, h, current_time_str in bboxes:
            stored_frames[obj_id].append(frame)
            stored_bboxes[obj_id].append((x, y, w, h))
            # print(stored_bboxes)
            if len(stored_frames[obj_id]) == 8:
                clips = []
                for idx, f in enumerate(stored_frames[obj_id]):
                    x, y, w, h = stored_bboxes[obj_id][idx]
                    clip = f[int(y):int(y + h), int(x):int(x + w)]
                    if clip.shape[0] < 7 or clip.shape[1] < 7:
                        continue
                    if clip.shape[2] == 1:
                        clip = cv2.cvtColor(clip, cv2.COLOR_GRAY2RGB)
                    clip_resized = cv2.resize(clip, (224, 224))
                    clips.append(clip_resized)

                if len(clips) == 8:
                    tensor_clips = [torch.from_numpy(clip).permute(2, 0, 1).float() / 255.0 for clip in clips]
                    tensor = torch.stack(tensor_clips, dim=1)
                    tensor = tensor.unsqueeze(0)

                    with torch.no_grad():
                        # print(tensor.shape)
                        preds = model(tensor)
                    
                        post_act = torch.nn.Softmax(dim=1)
                        preds = post_act(preds)
                        pred_classes = preds.topk(k=2).indices

                    action_label = [kinetics_classnames[int(i)] for i in pred_classes[0]]
                    action_label = action_label[0]

                    results.append(f"{frame_id},{int(obj_id)},{action_label},{current_time_str}\n")
                    x, y, w, h = stored_bboxes[obj_id][-1]
                    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                    cv2.putText(frame, action_label, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    stored_frames[obj_id] = []
                    stored_bboxes[obj_id] = []

        out.write(frame)
        frame_id += 1

    # 예측 결과 txt 저장 -> /home2/korengc/Output/Action/ch02/2023/10/14/20231014233229.txt
    with open(Action_output_path+"/"+file_name + ".txt", "w") as f:
        f.writelines(results)

    cap.release()
    out.release()

    motToTxt_command = f'python3 motToTxt.py --MOT_output_path {path_list[0]} --Trans_output_path {path_list[2]} --file_name {path_list[8]} --path_info {path_list[0]} {path_list[1]} {path_list[2]} {path_list[3]} {path_list[4]} {path_list[5]} {path_list[6]} {path_list[7]} {path_list[8]}'
    os.system(motToTxt_command)

if __name__ == "__main__" :
    args = make_parser().parse_args()
    main(args)