##mapping.py
import csv
import argparse
import json
import os


def make_parser():
    parser = argparse.ArgumentParser("mapping")

    parser.add_argument(
        "--path_info", nargs='+', help="Output_path_information"
    )

    return parser

def main(args) :
    '''
        Trans_output_path : Trans 저장 경로 /home2/korengc/Output/Trans/ch02/2023/10/14
        Bluetooth_output_path : Bluetooth 저장 경로 /home2/korengc/Output/Bluetooth/2023/10/14
        Mapping_output_path : Mapping 저장 경로 /home2/korengc/Output/Mapping/ch02/2023/10/14
        filename : 파일명 (타임스탬프) 20231014233229
    '''
    path_list = args.path_info

    Trans_output_path = path_list[2]
    Bluetooth_output_path = path_list[4]
    Mapping_output_path = path_list[5]
    file_name = path_list[9]

    MOTbetween = Trans_output_path + "/" + file_name +"_MOTbetween" +".txt"
    transBT_modified = Bluetooth_output_path + "/" + file_name +"_transBT_modified" +".txt"

    # 데이터를 저장할 딕셔너리 초기화
    mapping = {}
    max_last_times = {}

    # MOTbetween.txt 파일 읽기
    with open(MOTbetween, mode='r') as mot_file:
        mot_reader = csv.DictReader(mot_file, delimiter=',')
        for row in mot_reader:
            object_id = int(row['objectId'])
            time = row['time']
            value = row['value']

            # value가 'first'인 경우 매핑
            if value == 'first':
                # 이미 매핑된 objectId가 아니라면
                if object_id not in mapping:
                    mapping[object_id] = {'first': time, 'deviceName': None}
            else:
                mapping[object_id]['last'] = time

    # print(mapping)

    # transBT_modified.txt 파일 읽기
    with open(transBT_modified, mode='r') as trans_file:
        for line in trans_file:
            fields = line.strip().split(',')
            # print(fields)
            time = fields[0]
            device_name = fields[1]
            mac = fields[2]
            rssi = fields[3]
            value = fields[4]

            # value가 'in'이고, 해당 시간의 objectId가 매핑되지 않았다면
            if value == 'in' and time in [mapping[obj]['first'] for obj in mapping if mapping[obj]['deviceName'] is None]:
                for object_id in mapping:
                    if mapping[object_id]['first'] == time and mapping[object_id]['deviceName'] is None:
                        mapping[object_id]['deviceName'] = device_name
                        max_last_times[device_name] = mapping[object_id]['last']
                        break

    # print(max_last_times)

    # Iterate through the mapping dictionary again and update deviceName for objects with None
    for obj, data in mapping.items():
        device_name = data['deviceName']
        first_time = data['first']
        last_time = data['last']

        # Check if deviceName is None, and the object's first time is smaller than the maximum last time
        if device_name is None:
            filter_max = [key for key, value in max_last_times.items() if value <= first_time]
            # Update deviceName with the deviceName of the object with the largest last time
            data['deviceName'] = max(filter_max)
            max_last_times[data['deviceName']] =  last_time = data['last']

    final_mapping = {}
    # 결과 출력
    for obj in mapping:
        # print(f"ObjectId {obj}: DeviceName {mapping[obj]['deviceName']}")
        final_mapping[obj] = mapping[obj]['deviceName']

    # 딕셔너리를 텍스트 파일에 저장
    with open(Mapping_output_path + "/" + file_name +"_mapping" +".txt", 'w') as txt_file:
        for key, value in final_mapping.items():
            txt_file.write(f"{key},{value}\n")

    BEV_command = f'python3 BEV.py --path_info {path_list[0]} {path_list[1]} {path_list[2]} {path_list[3]} {path_list[4]} {path_list[5]} {path_list[6]} {path_list[7]} {path_list[8]} {path_list[9]}'
    os.system(BEV_command)

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
