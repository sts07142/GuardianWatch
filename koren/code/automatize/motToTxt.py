##newMOTtoTXT.py
import pandas as pd
import argparse
import json
import os

'''
    최종 저장물
    용도 모름
'''

def make_parser():
    parser = argparse.ArgumentParser("motTotxt")

    parser.add_argument(
        "--path_info", nargs='+', help="Output_path_information"
    )

    return parser


def main(args) :
    '''
        MOT_output_path : MOT 최종본 경로 (txt, mp4의 경로) /home2/korengc/Output/MOT/ch02/2023/10/14
        Trans_output_path : Trans 저장 경로 /home2/korengc/Output/Trans/ch02/2023/10/14
        file_name : 파일명 (타임스탬프) 20231014233229
    '''
    print("motToTxt.py")
    path_list = args.path_info

    MOT_output_path = path_list[0]
    Trans_output_path = path_list[2]
    file_name = path_list[9]

    # 텍스트 파일 읽기
    data = []
    with open(MOT_output_path+"/"+file_name+".txt", 'r') as file:
        for line in file:
            line = line.strip().split(',')
            data.append(line)

    # 데이터를 DataFrame으로 변환
    df = pd.DataFrame(data, columns=["frameId", "objectId", "bboxX", "bboxY", "bboxW", "bboxH", "score", "currentTime", "-", "-"])

    # objectId를 숫자로 변환
    df['objectId'] = df['objectId'].astype(int)

    # currentTime 열을 datetime 형식으로 변환
    df['currentTime'] = pd.to_datetime(df['currentTime'])

    # 각 objectId에 대한 가장 처음 currentTime과 가장 마지막 currentTime 출력
    first_last_times = df.groupby('objectId')['currentTime'].agg(['first', 'last']).reset_index()

    # last - first가 5초 이상인 레코드만 선택
    first_last_times['duration'] = (first_last_times['last'] - first_last_times['first']).dt.total_seconds()
    first_last_times = first_last_times[first_last_times['duration'] >= 5]

    # last가 빠르고 first가 느린 순으로 정렬
    first_last_times = first_last_times.sort_values(by=['objectId'], ascending=[True])

    # duration 열을 삭제
    first_last_times = first_last_times.drop(columns=['duration'])

    # 정렬된 데이터를 'first_last_times.csv' 파일에 저장
    # first_last_times.to_csv('/home2/korengc/code/ByteTrack/transMOT.txt', index=False)
    first_last_times.to_csv(Trans_output_path + "/" + file_name +"_transMOT" +".txt", index=False)
    
    motBetween_command = f'python3 motBetween.py --path_info {path_list[0]} {path_list[1]} {path_list[2]} {path_list[3]} {path_list[4]} {path_list[5]} {path_list[6]} {path_list[7]} {path_list[8]} {path_list[9]}'
    os.system(motBetween_command)

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)