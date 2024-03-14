import pandas as pd
import argparse
import json
import os

def make_parser():
    parser = argparse.ArgumentParser("transBT_modify")

    parser.add_argument(
        "--path_info", nargs='+', help="Output_path_information"
    )

    return parser

def main(args) :
    '''
        Bluetooth_input_path : Bluetooth 입력 경로 /home2/korengc/KOREN/Bluetooth/2023/10/14
        Bluetooth_output_path : Bluetooth 저장 경로 /home2/korengc/Output/Bluetooth/2023/10/14
        filename : 파일명 (타임스탬프) 20231014233229
    '''
    print("transBT_modify.py")

    path_list = args.path_info

    Bluetooth_input_path = path_list[3]
    Bluetooth_output_path = path_list[4]
    file_name = path_list[9]

    input_file = Bluetooth_input_path + "/transBT.txt"
    output_file = Bluetooth_output_path + "/" + file_name +"_transBT_modified" +".txt"

    # input_file =r'/home2/korengc/code/ByteTrack/transBT.txt'
    # output_file=r"/home2/korengc/code/ByteTrack/transBT_modified.txt"
    
    # 파일을 읽어 DataFrame으로 변환
    data = pd.read_csv(input_file, header=None, names=['time', 'deviceName', 'mac', 'rssi'], skiprows=1)

    # rssi 값을 기반으로 value 열 추가
    def classify_rssi(rssi):
        rssi=int(rssi)
        if rssi >= -90:
            return 'in'
        elif rssi <= -100:
            return 'out'
        else:
            return None

    data['value'] = data['rssi'].apply(classify_rssi)

    # 이전 시간과 rssi 값을 기반으로 was_in과 was_out 열 추가
    data['prev_value'] = data.groupby('deviceName')['value'].shift(1)
    data['was_in'] = (data['value'] == 'in') & (data['prev_value'] == 'in')
    data['was_out'] = (data['value'] == 'out') & (data['prev_value'] == 'out')

    # was_in 값이 True이면 value 값을 'was_in'으로 저장
    data.loc[data['was_in'], 'value'] = 'was_in'

    # was_out 값이 True이면 value 값을 'was_out'으로 저장
    data.loc[data['was_out'], 'value'] = 'was_out'


    # 결과를 새로운 파일에 저장
    data.to_csv(output_file, index=False, header=None, columns=['time', 'deviceName', 'mac', 'rssi', 'value'])

    mapping_command = f'python3 mapping.py --path_info {path_list[0]} {path_list[1]} {path_list[2]} {path_list[3]} {path_list[4]} {path_list[5]} {path_list[6]} {path_list[7]} {path_list[8]} {path_list[9]}'
    os.system(mapping_command)

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)