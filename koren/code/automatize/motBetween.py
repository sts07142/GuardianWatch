## MOTbetween.py
# 입력 파일 이름과 출력 파일 이름 설정
import argparse
import json
import os

def make_parser():
    parser = argparse.ArgumentParser("motBetween")

    parser.add_argument(
        "--path_info", nargs='+', help="Output_path_information"
    )

    return parser

def main(args) :
    '''
        Trans_output_path : Trans 저장 경로 /home2/korengc/Output/Trans/ch02/2023/10/14
        filename : 파일명 (타임스탬프) 20231014233229
    '''
    print("motBetween.py")

    path_list = args.path_info

    Trans_output_path = path_list[2]
    file_name = path_list[9]

    input_file = Trans_output_path + "/" + file_name +"_transMOT" +".txt"
    output_file = Trans_output_path + "/" + file_name +"_MOTbetween" +".txt"

    # input_file = r"/home2/korengc/code/ByteTrack/transMOT.txt"
    # output_file = r"/home2/korengc/code/ByteTrack/MOTbetween.txt"

    # 입력 파일 열기
    with open(input_file, "r") as f:
        lines = f.readlines()

    # 출력 파일 열기
    with open(output_file, "w") as f:
        # 헤더 쓰기
        f.write("objectId,time,value\n")

        # 각 줄을 파싱하고 변환하여 출력 파일에 쓰기
        for line in lines[1:]:  # 첫 번째 줄은 헤더이므로 건너뜁니다.
            parts = line.strip().split(',')
            objectId = parts[0]
            first_time = parts[1]
            last_time = parts[2]

            # objectId, 시간 및 value를 출력 파일에 쓰기
            f.write(f"{objectId},{first_time},first\n")
            f.write(f"{objectId},{last_time},last\n")


    # 입력 파일 열기
    with open(output_file, "r") as f:
        lines = f.readlines()

    # 헤더를 제외한 데이터 정렬
    sorted_lines = sorted(lines[1:], key=lambda line: line.split(',')[1])

    # 출력 파일 열기
    with open(output_file, "w") as f:
        # 헤더 쓰기
        f.write(lines[0])

        # 정렬된 데이터 쓰기
        for line in sorted_lines:
            f.write(line)

    transBT_modify_command = f'python3 transBT_modify.py --path_info {path_list[0]} {path_list[1]} {path_list[2]} {path_list[3]} {path_list[4]} {path_list[5]} {path_list[6]} {path_list[7]} {path_list[8]} {path_list[9]}'
    os.system(transBT_modify_command)

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)