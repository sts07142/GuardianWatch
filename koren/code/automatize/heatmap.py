import pandas as pd
import numpy as np
import argparse
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as img
import warnings
import os 

map_img = '/home2/korengc/code/ByteTrack/back2.png'
sns_color = 'Reds'

def make_parser():
    parser = argparse.ArgumentParser("Heatmap")

    parser.add_argument(
        "--path_info", nargs='+', help="Output_path_information"
    )

    return parser

def make_grid():
    # Todo : 추후 맵 스케일링 진행시 해당 부분 수정 요망 (추가 파라미터 작업 필요)
    x_ticks = list(range(0, 1951, 50))
    y_ticks = list(range(0, 1101, 50))

    # 1920 * 1080으로 마지막 크기를 맞추기 위해서 마지막 원소에 대해 1920으로 맞춤
    x_ticks.pop(-1)
    x_ticks.append(1920)

    y_ticks.pop(-1)
    y_ticks.append(1080)
    return x_ticks, y_ticks

def draw_path_map(csv_file, kid_num, Ana_output_path, file_name):

    ori_data = pd.read_csv(csv_file)
    path_file_dir = Ana_output_path + "/" + file_name + "/" + "path/"
    os.makedirs(path_file_dir, exist_ok=True)

    for kid_number in range(1, kid_num + 1) : 
        # 아이의 번호로 data에서 x, y 값만 가져오기
        x, y = str(kid_number) + '_X', str(kid_number) + '_Y'
        data = ori_data[[x, y]]
        # 이미지 오버레이를 위한 이미지 load
        img_test = img.imread(map_img)

        fig, ax = plt.subplots(figsize=(12, 8))

        # scatter 생성
        ax = sns.scatterplot(data, x=x, y=y,
                            zorder=2, alpha=0.2)
        ax.imshow(img_test,
                aspect=ax.get_aspect(),
                extent=(0, 1920.0) + (1080.0, 0),
                zorder=1)
        sns.set_style("ticks", {'axes.grid': True})

        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)

        plt.xticks(rotation=90)
        plt.savefig(path_file_dir + 'path_kid' + str(kid_number) + '.jpg', bbox_inches='tight')
        plt.clf()

def draw_heat_map(csv_file, kid_num, Ana_output_path, file_name) :
    ori_data = pd.read_csv(csv_file)
    path_file_dir = Ana_output_path + "/" + file_name + "/" + "heatmap/"
    os.makedirs(path_file_dir, exist_ok=True)

    for kid_number in range(1, kid_num + 1) : 
        x, y = str(kid_number) + '_X', str(kid_number) + '_Y'
        data = ori_data[[x, y]]

        # 그리드 생성
        x_ticks, y_ticks = make_grid()

        # heatmap counting을 위해서 pd.cut으로 동일한 간격으로 나누어줌
        x_label = list(range(len(x_ticks) - 1))
        x_bins = pd.cut(data[x], x_ticks, labels=x_label)
        y_label = list(range(len(y_ticks) - 1))
        y_bins = pd.cut(data[y], y_ticks, labels=y_label)

        # 새로운 count 매트릭스를 위한 공간 생성
        bins_data = pd.DataFrame(columns=['count', 'x', 'y'])
        bins_data['x'] = x_bins
        bins_data['y'] = y_bins
        bins_data['count'] = 1

        # count 값 계산
        bins_data = bins_data.groupby(['x', 'y'])['count'].count().reset_index()

        # pd.cut을 사용하게 되면 return 값은 Categories로 생성되게 된다.
        # 즉 범위 값으로 되게 되는 데 이를 pd.factorize를 통해서 분리 가능

        bins_data['x'] = pd.factorize(bins_data['x'])[0]
        bins_data['y'] = pd.factorize(bins_data['y'])[0]

        # 추후 X, Y 축 범례 표시를 위해서 한 칸의 px만큼 다시 곱하여 원래의 좌표처럼 표현
        bins_data['x'] = bins_data['x'] * 50
        bins_data['y'] = bins_data['y'] * 50

        # 원래 데이터프레임 형태로는 heatmap을 그리기 적절하지 않음
        # pivot 함수를 통해서 x, y, count를 명확하게 표현
        bins_data = bins_data.pivot(index='x', columns='y', values='count')

        # 가로를 X축으로 두기 위해서 transpose를 진행
        bins_data = bins_data.T

        # heatmap 생성
        img_test = img.imread(map_img)
        fig, ax = plt.subplots(figsize=(12, 8))

        ax = sns.heatmap(bins_data, vmax=100, vmin=0,
                        alpha=0.3,
                        cmap=sns.color_palette(sns_color, 9),
                        zorder=2, cbar=False,
                        linewidths=0.7, linecolor='gray')
        
        ax.imshow(img_test,
                aspect=ax.get_aspect(),
                extent=ax.get_xlim() + ax.get_ylim(),
                zorder=1)

        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
        plt.xticks(rotation=90)
        plt.savefig(path_file_dir + 'HeatMap_kid' + str(kid_number) + '.jpg', bbox_inches='tight')
        plt.clf()

def main(args) :
    print("Heatmap.py")

    path_list = args.path_info
    BEV_output_path = path_list[6]
    Ana_output_path = path_list[7]
    file_name = path_list[9]

    input_file_csv = BEV_output_path + "/" + file_name+"_video_bev.txt"
    output_file_csv = Ana_output_path + "/" + file_name+"_video_bev.csv"

    # 오브젝트 번호와 좌표 정보를 저장할 딕셔너리 초기화
    data_dict = {}
    max_obj_num = 0

    # 텍스트 파일을 읽어서 데이터 딕셔너리에 저장
    with open(input_file_csv, "r") as txt_file:
        for line in txt_file:
            # 쉼표로 구분된 데이터 추출
            parts = line.strip().split(',')
            
            # 프레임 번호 가져오기
            frame_num = parts[0]
            
            # 오브젝트 번호 가져오기
            obj_num = parts[1]
            max_obj_num = max(max_obj_num, int(obj_num))

            # 좌표 정보 가져오기
            x = parts[2]
            y = parts[3]
            
            # 데이터 딕셔너리에 저장
            if frame_num not in data_dict:
                data_dict[frame_num] = {}
            data_dict[frame_num][f"{obj_num}_X"] = x
            data_dict[frame_num][f"{obj_num}_Y"] = y

    # 데이터를 CSV 파일로 쓰기
    with open(output_file_csv, "w", newline="") as csv_file:
        # CSV 파일 작성기 생성
        csv_writer = csv.writer(csv_file)
        
        # CSV 헤더 행 작성
        header = ["frameNum"]
        for obj_num in range(1, max_obj_num + 1):
            header.extend([f"{obj_num}_X", f"{obj_num}_Y"])
        csv_writer.writerow(header)
        
        # 데이터 딕셔너리에서 데이터 추출하여 CSV 파일에 쓰기
        for frame_num, data in data_dict.items():
            row = [frame_num]
            for obj_num in range(1, max_obj_num + 1):
                row.extend([data.get(f"{obj_num}_X", ""), data.get(f"{obj_num}_Y", "")])
            csv_writer.writerow(row)

    print("End Convert txt to csv.")

    # 입력 CSV 파일 경로
    input_file_csv_Interpolation = Ana_output_path + "/" + file_name+"_video_bev.csv"

    # 출력 CSV 파일 경로
    output_file_csv_Interpolation = Ana_output_path + "/" + file_name+"_video_bev_Interpolation.csv"

    # 데이터를 저장할 리스트 초기화
    data = []

    # 입력 CSV 파일 읽기
    with open(input_file_csv_Interpolation, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = [row for row in csv_reader]

    # 데이터를 선형적으로 채우기
    for row_num in range(1, len(data)):
        for col_num in range(1, len(data[row_num])):
            if data[row_num][col_num] == '':
                prev_value = None
                next_value = None

                # 이전 값 찾기
                for i in range(row_num, 0, -1):
                    if data[i][col_num] != '':
                        prev_value = float(data[i][col_num])
                        break

                # 다음 값 찾기
                for i in range(row_num, len(data)):
                    if data[i][col_num] != '':
                        next_value = float(data[i][col_num])
                        break

                if prev_value is not None and next_value is not None:
                    data[row_num][col_num] = (prev_value + next_value) / 2

    # 수정된 데이터를 출력 CSV 파일에 쓰기
    with open(output_file_csv_Interpolation, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(data)

    print("End Interpolation")

    print("Create Pathmap")
    draw_path_map(output_file_csv_Interpolation, max_obj_num, Ana_output_path, file_name)

    print("Create Heatmap")
    draw_heat_map(output_file_csv_Interpolation, max_obj_num, Ana_output_path, file_name)

    heatmap_command = f'python3 move_ana.py --path_info {path_list[0]} {path_list[1]} {path_list[2]} {path_list[3]} {path_list[4]} {path_list[5]} {path_list[6]} {path_list[7]} {path_list[8]} {path_list[9]}'
    os.system(heatmap_command)

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)