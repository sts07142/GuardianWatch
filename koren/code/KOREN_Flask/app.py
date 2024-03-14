from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import json
import mysql.connector
import os
import base64

app = Flask(__name__)

# DB config
with open('db_info.json', 'r') as f:
    db_config = json.load(f)

connection = mysql.connector.connect(
    host=db_config['Database']['host'],
    user=db_config['Database']['user'],
    password=db_config['Database']['password'],
    database=db_config['Database']['database'],
    auth_plugin='mysql_native_password',
)

cursor = connection.cursor()

UPLOAD_FOLDER = 'profile_image'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# signup
@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        id = data['id']
        password = data['password']

        insert_query = "INSERT INTO users (id, password) VALUES (%s, %s)"
        cursor.execute(insert_query, (id, password))
        connection.commit()

        response = {'message': 'signup OK'}
        return jsonify(response), 201

    except Exception as e:
        response = {'error': str(e)}
        return jsonify(response), 500


# login
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        id = data['id']
        password = data['password']

        select_query = "SELECT * FROM users WHERE id = %s AND password = %s"
        cursor.execute(select_query, (id, password))
        user = cursor.fetchone()

        if user:
            response = {'message': 'Login Successful!'}
            return jsonify(response), 200
        else:
            response = {'message': 'Login failed!'}
            return jsonify(response), 401

    except Exception as e:
        response = {'error': str(e)}
        return jsonify(response), 500


@app.route('/add_kid', methods=['POST'])
def add_kid():
    try:
        # 이미지 선택 처리
        id = request.form['id']
        name = request.form['name']
        gender = request.form['gender']
        year = request.form['year']
        month = request.form['month']
        day = request.form['day']
        place = request.form['place']

        image = request.files['image']

        if image:
            filename = secure_filename(image.filename)
            upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], id, name)
            os.makedirs(upload_dir, exist_ok=True)
            upload_path = os.path.join(upload_dir, filename)
            image.save(upload_path)
        else:
            upload_path = None

        # 아이 정보와 이미지 파일명을 데이터베이스에 저장
        insert_child_query = "INSERT INTO kids (id, name, gender, year, month, day, place, image) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        cursor.execute(insert_child_query, (id, name, gender, year, month, day, place, upload_path))
        connection.commit()

        response = {'message': 'Registering Child Information!'}
        return jsonify(response), 201

    except Exception as e:
        response = {'error': str(e)}
        return jsonify(response), 500


@app.route('/get_kids/<string:id>', methods=['GET'])
def get_kids_by_id(id):
    try:
        # 데이터베이스에서 특정 ID 값을 가진 아이들의 정보 가져오기
        select_kids_query = "SELECT name, place, year, month, day, image FROM kids WHERE id = %s"
        cursor.execute(select_kids_query, (id,))
        kids_data = cursor.fetchall()

        if not kids_data:
            response = {'message': 'No child information found.'}
            return jsonify(response), 404

        kids_list = []
        for kid_info in kids_data:
            if kid_info[5] is not None:
                # 이미지 데이터를 Base64로 인코딩
                image_path = kid_info[5]
                with open(image_path, 'rb') as img_file:
                    image_data = base64.b64encode(img_file.read()).decode('utf-8')
                    kid_data = {
                        'name': kid_info[0],
                        'place': kid_info[1],
                        'year': kid_info[2],
                        'month': kid_info[3],
                        'day': kid_info[4],
                        'image': f"data:image/jpg;base64,{image_data}"  # Base64 인코딩된 이미지 데이터
                    }
            else:
                kid_data = {
                    'name': kid_info[0],
                    'place': kid_info[1],
                    'year': kid_info[2],
                    'month': kid_info[3],
                    'day': kid_info[4],
                    'image': None  # Base64 인코딩된 이미지 데이터
                }
            kids_list.append(kid_data)

        response = jsonify(kids_list)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        response.status_code = 200

        return response

    except Exception as e:
        response = {'error': str(e)}
        return jsonify(response), 500


@app.route('/get_kid/<string:id>/<string:name>', methods=['GET'])
def get_kid_by_id_name(id, name):
    try:
        select_kids_query = "SELECT place, year, month, day, image FROM kids WHERE id = %s AND name = %s"
        cursor.execute(select_kids_query, (id, name))
        kid_info = cursor.fetchone()
        if kid_info is not None:
            if kid_info[4] is not None:
                image_path = kid_info[4]
                with open(image_path, 'rb') as img_file:
                    image_data = base64.b64encode(img_file.read()).decode('utf-8')
                    kid_data = {
                        'name': name,
                        'place': kid_info[0],
                        'year': kid_info[1],
                        'month': kid_info[2],
                        'day': kid_info[3],
                        'image': f"data:image/jpg;base64,{image_data}"  # Base64 인코딩된 이미지 데이터
                    }
            else:
                kid_data = {
                    'name': name,
                    'place': kid_info[0],
                    'year': kid_info[1],
                    'month': kid_info[2],
                    'day': kid_info[3],
                    'image': None  # Base64 인코딩된 이미지 데이터
                }

            response = jsonify(kid_data)
            response.headers['Content-Type'] = 'application/json; charset=utf-8'
            response.status_code = 200

            return response

    except Exception as e:
        response = {'error': str(e)}
        return jsonify(response), 500


def retrieve_existing_image(id, name):
    try:
        # Query the database to get the existing image path based on the ID and name
        query = "SELECT image FROM kids WHERE id = %s AND name = %s"
        cursor.execute(query, (id, name))
        result = cursor.fetchone()

        if result and result[0]:
            return result[0]  # Return the existing image path
        else:
            return None  # No existing image found

    except Exception as e:
        print(f"Error retrieving existing image: {str(e)}")
        return None


@app.route('/change_kid/<string:id>/<string:name>', methods=['PUT'])
def change_kid_by_id_name(id, name):
    try:
        # 이미지 선택 처리
        new_id = request.form['id']
        new_name = request.form['name']
        new_gender = request.form['gender']
        new_year = request.form['year']
        new_month = request.form['month']
        new_day = request.form['day']
        new_place = request.form['place']

        new_image = request.files['image']

        if new_image:
            filename = secure_filename(new_image.filename)
            upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], id, name)
            os.makedirs(upload_dir, exist_ok=True)
            upload_path = os.path.join(upload_dir, filename)

            existing_image_path = retrieve_existing_image(id, name)
            if existing_image_path is not None:
                if existing_image_path != upload_path:
                    new_image.save(upload_path)
            else:
                new_image.save(upload_path)

        else:
            upload_path = None

        # 새로운 아이 정보와 이미지 파일명을 데이터베이스에 저장
        update_child_query = """
                UPDATE kids
                SET id=%s, name=%s, gender=%s, year=%s, month=%s, day=%s, place=%s, image=%s
                WHERE id=%s AND name=%s
                """
        cursor.execute(update_child_query,
                       (new_id, new_name, new_gender, new_year, new_month, new_day, new_place, upload_path, id, name))
        connection.commit()

        response = {'message': 'Change Child Information!'}
        return jsonify(response), 200

    except Exception as e:
        response = {'error': str(e)}
        return jsonify(response), 500


def get_map_from_dir(device_num, year, month, day, mode):
    base_directory = '/home2/korengc/Output/Analysis'
    all_subdirectories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if
                          os.path.isdir(os.path.join(base_directory, d))]
    result = []
    subdirectories = []

    for subdirectory in all_subdirectories:
        subdirectory_path = os.path.join(subdirectory, f'{year}/{month}/{day}')
        if os.path.exists(subdirectory_path):
            subdirectories.extend(
                [os.path.join(subdirectory_path, d) for d in os.listdir(subdirectory_path) if
                 os.path.isdir(os.path.join(subdirectory_path, d))])

    for subdirectory in subdirectories:
        for root, dirs, files in os.walk(subdirectory):
            for file in files:
                if file.startswith(f'{mode}_kid{device_num}.jpg'):
                    file_path = os.path.join(root, file)
                    result.append(file_path)
    return result


@app.route('/pathmap/<string:id>/<string:name>/<string:year>/<string:month>/<string:day>', methods=['GET'])
def get_pathmap(id, name, year, month, day):
    try:
        query = "SELECT deviceNum FROM kids WHERE id = %s AND name = %s"
        cursor.execute(query, (id, name))
        result = cursor.fetchone()

        if not result:
            response = {'message': 'No device information found.'}
            return jsonify(response), 404
        else:
            pathmap_list = []
            pathmap_path_list = get_map_from_dir(result[0], year, month, day, "path")
            for pathmap in pathmap_path_list:
                directories = pathmap.split('/')
                timestamp = directories[9]

                with open(pathmap, 'rb') as img_file:
                    image_data = base64.b64encode(img_file.read()).decode('utf-8')
                    data = {
                        'ch': directories[5],
                        'hour': timestamp[8:10],
                        'minute': timestamp[10:12],
                        'pathmap': f"data:image/jpg;base64,{image_data}"
                    }
                pathmap_list.append(data)

            response = jsonify(pathmap_list)
            response.headers['Content-Type'] = 'application/json; charset=utf-8'
            response.status_code = 200

            return response
    except Exception as e:
        response = {'error': str(e)}
        return jsonify(response), 500


@app.route('/heatmap/<string:id>/<string:name>/<string:year>/<string:month>/<string:day>', methods=['GET'])
def get_heatmap(id, name, year, month, day):
    try:
        query = "SELECT deviceNum FROM kids WHERE id = %s AND name = %s"
        cursor.execute(query, (id, name))
        result = cursor.fetchone()

        if not result:
            response = {'message': 'No device information found.'}
            return jsonify(response), 404
        else:
            heatmapmap_list = []
            heatmap_path_list = get_map_from_dir(result[0], year, month, day, "HeatMap")
            for heatmap in heatmap_path_list:
                directories = heatmap.split('/')
                timestamp = directories[9]

                with open(heatmap, 'rb') as img_file:
                    image_data = base64.b64encode(img_file.read()).decode('utf-8')
                    data = {
                        'ch': directories[5],
                        'hour': timestamp[8:10],
                        'minute': timestamp[10:12],
                        'pathmap': f"data:image/jpg;base64,{image_data}"
                    }
                heatmapmap_list.append(data)

            response = jsonify(heatmapmap_list)
            response.headers['Content-Type'] = 'application/json; charset=utf-8'
            response.status_code = 200

            return response

    except Exception as e:
        response = {'error': str(e)}
        return jsonify(response), 500


def get_bev_from_dir(year, month, day):
    base_directory = '/home2/korengc/Output/BEV'
    all_subdirectories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if
                          os.path.isdir(os.path.join(base_directory, d))]
    BEV = []

    for subdirectory in all_subdirectories:
        subdirectory_path = os.path.join(subdirectory, f'{year}/{month}/{day}')

        if os.path.exists(subdirectory_path):
            # .mp4 파일을 검색하고 BEV 리스트에 추가
            mp4_files = [os.path.join(root, file) for root, dirs, files in os.walk(subdirectory_path) for file in files
                         if file.endswith('.mp4')]
            BEV.extend(mp4_files)

    return BEV


@app.route('/BEV/<string:id>/<string:name>/<string:year>/<string:month>/<string:day>/<string:ch>', methods=['GET'])
def get_bev(id, name, year, month, day, ch):
    try:
        bev_path_list = get_bev_from_dir(year, month, day)
        print(f'bev_path_list : {bev_path_list}', flush=True)

        for bev in bev_path_list:
            directories = bev.split('/')

            if directories[5] == ch:
                def generate():
                    with open(bev, 'rb') as video_file:
                        while True:
                            video_chunk = video_file.read(1024)
                            if not video_chunk:
                                break
                            yield (b'--frame\r\n'
                                   b'Content-Type: video/mp4\r\n\r\n' + video_chunk + b'\r\n')

        response = app.response_class(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
        return response

    except Exception as e:
        response = {'error': str(e)}
        return jsonify(response), 500


@app.route('/upload_cctv/<string:ch>', methods=['POST'])
def upload_cctv(ch):
    base_upload_dir = "/home2/korengc/KOREN/CCTV/"
    save_path = os.path.join(base_upload_dir, ch)
    uploaded_file = request.files['file']

    if uploaded_file:
        uploaded_file.save(save_path + uploaded_file.filename)
        return 'File uploaded successfully.'

    return 'File upload failed.'


@app.route("/")
def hello():
    return "Hello World!"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=11000)
    # flask run --host=0.0.0.0 --port=11000
