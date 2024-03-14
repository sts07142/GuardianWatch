## BEV
# 흰 배경에 바뀐 좌표
import cv2
from collections import defaultdict
import numpy as np
from collections import deque
import argparse
import json
import os

def make_parser():
    parser = argparse.ArgumentParser("BEV")

    parser.add_argument(
        "--path_info", nargs='+', help="Output_path_information"
    )

    return parser

# 1 = 인재, 2 = 지우, 3 = 정곤, 4 = 지성, 5 = 태경
# 빈 딕셔너리 생성
def main(args) :
    '''
        original_video : 원본 영상 /home2/korengc/KOREN/CCTV/ch02/2023/10/14/{timestamp}.mp4
        MOT_output_path : MOT 저장 경로 /home2/korengc/Output/MOT/ch02/2023/10/14
        Action_output_path : Action 저장 경로 /home2/korengc/Output/Action/ch02/2023/10/14
        Mapping_output_path : Mapping 저장 경로 /home2/korengc/Output/Mapping/ch02/2023/10/14
        BEV_output_path : Trans 저장 경로 /home2/korengc/Output/BEV/ch02/2023/10/14
        file_name : 파일명 (타임스탬프) 20231014233229
    '''
    print("BEV.py")

    path_list = args.path_info

    original_video = path_list[8]
    MOT_output_path = path_list[0]
    Action_output_path = path_list[1]
    Mapping_output_path = path_list[5]
    BEV_output_path = path_list[6]
    file_name = path_list[9]

    print(original_video)
    print(MOT_output_path)
    print(Action_output_path)
    print(Mapping_output_path)
    print(BEV_output_path)
    print(file_name)


    mapping = {}
    # with open('your_info.json', 'r', encoding='utf-8') as file:
    #     nameDic = json.load(file)

    nameDic = {1:"In Jae",2:"Jee woo",3:"Jung Gon",4:"Ji Sung",5:"Tae Gyeong"}

    with open(Mapping_output_path + "/" + file_name +"_mapping" +".txt", 'r') as txt_file:
        for line in txt_file:
            # 줄 바꿈 문자 제거 후 콜론(:)을 기준으로 문자열 분리
            key, value = line.strip().split(',')
            # 딕셔너리에 추가
            mapping[key] = int(value)
            print(mapping)

    # 바운딩 박스 정보를 포함하는 파일을 엽니다.
    with open(MOT_output_path+"/"+file_name+".txt", 'r') as f:
        bbox_info = f.readlines()

    # 프레임 아이디를 키로 사용하여 모든 바운딩 박스 정보를 사전으로 구성합니다.
    bbox_dict = defaultdict(list)
    for line in bbox_info:
        # print(f'line : {line}')
        values = line.strip().split(',')
        frame_id, obj_id = int(values[0]), int(values[1])
        coords = list(map(float, values[2:6]))
        timestamp = values[7]
        bbox_dict[frame_id].append((obj_id, *coords, timestamp))

    actions = {}
    with open(Action_output_path+"/"+file_name + ".txt", "r") as f:
        for line in f.readlines():
            # print(line)
            # print(line.strip().split(", "))
            fid, oid, action, _ = line.strip().split(",")
            actions[(int(fid), int(oid))] = action

    pending_frames = deque()

    # 투시 변환 설정
    src_points = np.array([
        [240,560],
        [1100, 260],
        [565, 1070],
        [1560, 450]
    ], dtype=np.float32)

    # src_points = np.array([
    #     [72,603],
    #     [1056, 241],
    #     [1622, 1080],
    #     [1866, 570]
    # ], dtype=np.float32)

    cap = cv2.VideoCapture(original_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 165,0
    # 1810,750

    # dst_points = np.array([
    #     [0, 0],
    #     [width, 0],
    #     [0, height],
    #     [width, height]
    # ], dtype=np.float32)

    dst_points = np.array([
        [165, 0],
        [1810, 0],
        [165, 750],
        [1810, 750]
    ], dtype=np.float32)

    # dst_points = np.array([
    #     [440, 0],
    #     [1480, 0],
    #     [713, 1080],
    #     [1480, 1080]
    # ], dtype=np.float32)
    

    # 투시 변환 매트릭스 계산
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(BEV_output_path + "/" + file_name +"_video_bev" +".mp4", fourcc, cap.get(cv2.CAP_PROP_FPS),
                        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # 흰색으로 초기화된 배경 이미지를 생성합니다.
    # blank_image = np.ones((height, width, 3), dtype=np.uint8) * 255

    blank_image = cv2.imread('/home2/korengc/code/ByteTrack/back.png')
    # blank_image = cv2.imread('/home2/korengc/code/ByteTrack/back2.png')


    # 변환된 좌표를 저장할 파일 객체 생성
    transformed_file = open(BEV_output_path + "/" + file_name +"_video_bev" +".txt", 'w')

    frame_id = 0
    while cap.isOpened():

        if frame_id % 1000 == 0:
            print(frame_id)

        ret, frame = cap.read()
        if not ret:
            break

        # 출력 프레임을 흰색으로 초기화합니다.
        output_frame = blank_image.copy()

        # 현재 프레임의 모든 바운딩 박스를 가져옵니다.
        bboxes = bbox_dict[frame_id]

        for obj_id, x, y, w, h, timestamp in bboxes:
            action = actions.get((frame_id, obj_id), None)

            obj_id=str(obj_id)
            mapped = bool
            name = ""
            #매핑
            if obj_id in mapping.keys():
                obj_id = mapping[obj_id]
                mapped = True
                name = nameDic.get(obj_id)
                # print(name)
            
            obj_id_check = int(obj_id)
            if 1<=obj_id_check<=5:
                # 중심점 좌표 변환
                transformed_points = cv2.perspectiveTransform(np.array([[[x+0.5*w, y+h]]]), matrix)[0][0]
                trans_x, trans_y = transformed_points

                if mapped == True:
                    # label_text = f"ID: {int(obj_id)}, Action: {action}, Name: {name}"
                    label_text = f"ID: {int(obj_id)}"
                else:
                    label_text = f"ID: {int(obj_id)}"

                # 중심점 좌표 변환
                transformed_points = cv2.perspectiveTransform(np.array([[[x+0.5*w, y+h]]]), matrix)[0][0]
                trans_x, trans_y = transformed_points

                # 변환된 좌표를 파일에 저장
                
                transformed_file.write(f"{frame_id},{int(obj_id)},{trans_x:.2f},{trans_y:.2f},{action},{timestamp}\n")

                # 객체의 ID와 행동 시간을 그립니다.
                cv2.putText(output_frame, label_text, (int(trans_x), int(trans_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # 변환된 좌표로 원과 텍스트를 그립니다.
                if obj_id==1:
                    cv2.circle(output_frame, (int(trans_x), int(trans_y)), 8, (1,0,0),-1)
                else:
                    cv2.circle(output_frame, (int(trans_x), int(trans_y)), 8, (0,0,255),-1)

                cv2.putText(output_frame, "({},{})".format(int(trans_x), int(trans_y)), (int(trans_x), int(trans_y+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 변환된 프레임을 새로운 동영상 파일에 씁니다.
        out.write(output_frame)

        frame_id += 1

    cap.release()
    out.release()
    transformed_file.close()  # 파일을 닫아줍니다.

    # heatmap_command = f'python3 heatmap.py --path_info {path_list[0]} {path_list[1]} {path_list[2]} {path_list[3]} {path_list[4]} {path_list[5]} {path_list[6]} {path_list[7]} {path_list[8]} {path_list[9]}'
    # os.system(heatmap_command)

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)