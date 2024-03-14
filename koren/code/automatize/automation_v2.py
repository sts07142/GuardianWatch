import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import subprocess
import json

# import behavior_prediction
# import motToTxt
# import motBetween
# import transBT_modify
# import mapping
# import BEV

import warnings
warnings.filterwarnings(action='ignore')

Bluetooth_base_dir = "/home2/korengc/KOREN/Bluetooth"

MOT_save_base_dir = "/home2/korengc/Output/MOT"
Action_save_base_dir = "/home2/korengc/Output/Action"
Trans_save_base_dir = "/home2/korengc/Output/Trans"
Bluetooth_save_base_dir = "/home2/korengc/Output/Bluetooth"
Mapping_save_base_dir = "/home2/korengc/Output/Mapping"
BEV_save_base_dir = "/home2/korengc/Output/BEV"
Analysis_save_base_dir = "/home2/korengc/Output/Analysis"


class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        # .mp4 확장명을 갖는 파일이 생성됐을 때 실행할  작업
        time.sleep(2 * 60)

        if not event.is_directory and event.src_path.endswith(".mp4") :
                    
            # /home2/korengc/KOREN/CCTV/ch02/2023/10/14/{timestamp}.mp4
            path_parts = event.src_path.split('/')            # ['', 'home2', 'korengc', 'KOREN', 'CCTV', 'ch02', '2023', '10', '14', '{timestamp}.mp4']
            file_name_with_extension = path_parts[-1]
            file_name, extension = os.path.splitext(file_name_with_extension)

            date = '/'.join(path_parts[6:9]) # 2023/10/14
            ch_date = '/'.join(path_parts[5:9]) # ch02/2023/10/14

            # Bluetooth Input
            Bluetooth_input_path = os.path.join(Bluetooth_base_dir, date) # /home2/korengc/KOREN/Bluetooth/2023/10/14

            # Output dir
            MOT_output_path = os.path.join(MOT_save_base_dir, ch_date) # /home2/korengc/Output/MOT/ch02/2023/10/14
            Action_output_path = os.path.join(Action_save_base_dir, ch_date) # /home2/korengc/Output/Action/ch02/2023/10/14
            Trans_output_path = os.path.join(Trans_save_base_dir, ch_date) # /home2/korengc/Output/Action/ch02/2023/10/14
            Bluetooth_output_path = os.path.join(Bluetooth_save_base_dir, date) # /home2/korengc/Output/Bluetooth/2023/10/14
            Mapping_output_path = os.path.join(Mapping_save_base_dir, ch_date) # /home2/korengc/Output/Mapping/ch02/2023/10/14
            BEV_output_path = os.path.join(BEV_save_base_dir, ch_date) # /home2/korengc/Output/BEV/ch02/2023/10/14
            Analysis_output_path = os.path.join(Analysis_save_base_dir, ch_date) # /home2/korengc/Output/Analysis/ch02/2023/10/14

            # 디렉토리가 없을 경우 output_path 디렉토리 생성
            os.makedirs(MOT_output_path, exist_ok=True)
            os.makedirs(Action_output_path, exist_ok=True)
            os.makedirs(Trans_output_path, exist_ok=True)
            os.makedirs(Bluetooth_output_path, exist_ok=True)
            os.makedirs(Mapping_output_path, exist_ok=True)
            os.makedirs(BEV_output_path, exist_ok=True)
            os.makedirs(Analysis_output_path, exist_ok=True)

            time.sleep(5)

            # MOT 실행
            MOT_command = f'python ByteTrack/tools/demo_track.py video -f ByteTrack/exps/example/mot/yolox_x_mix_det.py -c ByteTrack/pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --path {event.src_path} --output_path {MOT_output_path} --path_info {MOT_output_path} {Action_output_path} {Trans_output_path} {Bluetooth_input_path} {Bluetooth_output_path} {Mapping_output_path} {BEV_output_path} {Analysis_output_path} {event.src_path} {file_name} --save_result &> {MOT_output_path}/log.txt'            
            os.system(MOT_command)

    def on_modified(self, event):
        return super().on_modified(event)
    
if __name__ == "__main__":
    Monitoring_dir = "/home2/korengc/KOREN/CCTV/"  # 모니터링할 경로를 지정
    print(f'Watching {Monitoring_dir}')
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, Monitoring_dir, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
