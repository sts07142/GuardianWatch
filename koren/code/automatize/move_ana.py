import csv
import math
import argparse
import os
import json
import requests

def make_parser():
    parser = argparse.ArgumentParser("BEV")

    parser.add_argument(
        "--path_info", nargs='+', help="Output_path_information"
    )

    return parser

def get_distance(x1, y1, x2, y2):
    x1 = float(x1)
    y1 = float(y1)
    x2 = float(x2)
    y2 = float(y2)
    
    return math.sqrt(math.pow((x1 - x2),2) + math.pow((y1 - y2),2))

def get_real_distance(distance):
    # 630 cm = 1755 px
    return round(float(distance) * 630 / 1755)

def get_kcal(distance,frameNum):

    #걷기 kcal
    #walking kcal = weight * distance(cm) * 6 / (10**6)
    #성인 기준 67kg / 5살 기준 20kg
    #+기초대사량 1500kcal 기준 / 1초당 kcal = 1500 / 24 / 60 / 60

    walking_kcal = float(distance) * 0.000006 * 67
    idle_kcal = int(frameNum) / 30 * 1500 / 24 / 60 / 60
    return  round(walking_kcal + idle_kcal)

def transfer_data(url, folder_path) : 
    files = {}

    for foldername, subfolders, filenames in os.walk(folder_path):
        # print(f'foldername : {foldername}')
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            # print(f'file_path : {file_path}')
            relative_path = os.path.relpath(file_path, folder_path)

            with open(file_path, 'rb') as file:
                files = {'file': (relative_path, file)}
                # print(f'files : {files}')
                # print(f'url : {url}')
                response = requests.post(url, files=files)

                # 서버 응답 확인
                if response.status_code == 200:
                    print(f'File uploaded successfully: {file_path}')
                else:
                    print(f'File upload failed: {file_path}')


def main(args) :

    print("move_ana.py")

    path_list = args.path_info
    Ana_output_path = path_list[7]
    file_name = path_list[9]

    moveDistance_output_filePath = Ana_output_path + "/" + file_name +"_moveDistance.csv"
    useKcal_output_filePath = Ana_output_path + "/" + file_name +"_useKcal.csv"
    bev_Interpolation_filePath = Ana_output_path + "/" + file_name+"_video_bev_Interpolation.csv"

    moveDistanceF=open(moveDistance_output_filePath,'w',newline='')
    useKcalF=open(useKcal_output_filePath,'w',newline='')
    
    mD_wr=csv.writer(moveDistanceF)
    uK_wr=csv.writer(useKcalF)
    
    with open('/home2/korengc/code/automatize/your_info.json', 'r') as json_file:
        user_json_data = json.load(json_file)
    
    for i in range(1, len(user_json_data) + 1):
        globals()[f'total_distance{i}'] = 0
        globals()[f'x{i}'] = 0
        globals()[f'y{i}'] = 0
        globals()[f'bx{i}'] = 0
        globals()[f'by{i}'] = 0
        globals()[f'real_distance{i}'] = 0
        globals()[f'use_kcal{i}'] = 0

    with open(bev_Interpolation_filePath, "r") as f:
        reader = csv.reader(f)
        next(reader, None)

        for row in reader:
            frameNum = row[0]
            values = [float(value) for value in row[1:]]
            # print(f'values : {values}')

            for j, value in enumerate(values):
                if j % 2 == 0:
                    x_index = j // 2 + 1  # 인덱스를 1부터 시작하도록 수정
                    globals()[f'x{x_index}'] = value
                else:
                    y_index = (j - 1) // 2 + 1  # 인덱스를 1부터 시작하도록 수정
                    globals()[f'y{y_index}'] = value      

            if float(frameNum) == 0:
                for i in range(1, len(user_json_data) + 1):
                    globals()[f'bx{i}'] = globals()[f'x{i}']
                    globals()[f'by{i}'] = globals()[f'y{i}']
                continue
            else:
                for i in range(1, len(user_json_data) + 1):
                    globals()[f'total_distance{i}'] += get_distance(globals()[f'x{i}'], globals()[f'y{i}'], globals()[f'bx{i}'], globals()[f'by{i}'])
                    globals()[f'bx{i}'] = globals()[f'x{i}']
                    globals()[f'by{i}'] = globals()[f'y{i}']

        real_distances = []
        use_kcals = []

        for i in range(1, len(user_json_data) + 1):
                globals()[f'real_distance{i}'] = get_real_distance(globals()[f'total_distance{i}'])
                globals()[f'use_kcal{i}'] = get_kcal(globals()[f'real_distance{i}'], frameNum)
                real_distances.append(globals()[f'real_distance{i}'])
                use_kcals.append(globals()[f'use_kcal{i}'])

        mD_wr.writerow(real_distances)
        uK_wr.writerow(use_kcals)

        # print("Moving cm")
        # print(real_distances)

        # print("Use kcal")
        # print(use_kcals)
        print("Finish move analysis")

    moveDistanceF.close()
    useKcalF.close()

    url_list = ["http://210.102.178.157:8000/upload/folder/Analysis", "http://210.102.178.157:8000/upload/folder/BEV"]
    folder_path_list =['/home2/korengc/Output/Analysis', '/home2/korengc/Output/BEV']

    print("Transfer file to other server")
    for url, folder_path in zip(url_list, folder_path_list) : 
        # print(f'URL : {url}')
        # print(f'folder_path : {folder_path}')
        transfer_data(url, folder_path)

    print("End Process")



if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)