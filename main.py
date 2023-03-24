import os
import torch
import time
import glob

from model import Trainer
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor

from cut_template import find_axis
from create_font_img import create_template_data
from create_my_font import create_my_font_img
from model import checkEpoch

app = Flask(__name__)
executor = ThreadPoolExecutor(5)
lang = 'en'
model_name = ''

# Local인지 외부 서버인지 확인
isLocal = input("is Local? (T/F)>> ")
if isLocal == 't' or isLocal == 'T':
    addr = 'http://localhost:5000'
else:
    addr = 'http://35.239.13.181:5000'

print(addr)


# 메인 페이지
@app.route('/')
@app.route('/page_en')
def main_en():
    global lang
    is_learning = False
    lang = 'en'
    return render_template('mainpage_en.html', addr=addr)


@app.route('/page_ko')
def main_ko():
    global lang
    lang = 'ko'
    return render_template('mainpage_ko.html', addr=addr)


is_learning = False
pre_progress = 1
curr_epoch = 0


# 파일 업로드가 되면 학습중 페이지 표시
@app.route('/uploader', methods=['GET', 'POST'])
def uploader_file():
    global progress, model_name, is_learning, curr_epoch
    if request.method == 'POST':
        # epoch이 0인지 확인
        curr_epoch = checkEpoch()
        if curr_epoch != 0:
            return render_template('learning.html', addr=addr, progress=progress, lang=lang)

        # modelName값 받아오기
        model_name = request.form['modelName']

        if model_name in os.listdir('model/'):
            return render_template('mainpage_' + lang + '.html', model_name=model_name)

        # chooseFile값 받아서 template.png로 저장
        is_learning = False
        f = request.files['chooseFile']
        f.filename = 'template.png'
        f.save('static/images/' + secure_filename(f.filename))

    # 업로드한 템플릿 train data만들기
    executor.submit(make_all_syll)
    return render_template('learning.html', progress=0, lang=lang)


# 학습 중
@app.route('/learning')
def learning():
    global is_learning, progress, pre_progress, lang

    # 학습이 시작 안되었으면 학습 시작
    if not is_learning:
        time.sleep(15)
        progress = 1
        executor.submit(learn_template)
        is_learning = True

        return render_template('learning.html', addr=addr, progress=progress, lang=lang)

    from model import curr_epoch
    progress = 1 + int((curr_epoch / 300) * 99)

    # 학습 완료 시
    if progress == 100:
        progress = 0
        pre_progress = 0
        is_learning = False

        return render_template('endPage_' + lang + '.html', addr=addr)

    if pre_progress != progress:
        pre_progress = progress
    return render_template('learning.html', addr=addr, progress=progress)


# 음절 자르고 만들기
def make_all_syll():
    print('make_all_syll_model_name: ', model_name)
    find_axis(model_name=model_name)
    print(model_name)
    create_template_data(model_name)


# 템플릿 학습
def learn_template():
    global is_learning
    GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = 'data/template_syll/' + model_name + '/'
    embed_dir = 'model/pretrain/'
    fonts_num = 31
    batch_size = 15
    img_size = 128

    max_epoch = 300
    schedule = 30
    os.mkdir('model/' + model_name + '/')

    save_model_path = 'model/' + model_name + '/'
    from_model_path = 'model/pretrain/'
    restore = ['180-0922-22_07-Encoder.pkl', '180-0922-22_07-Decoder.pkl', '180-0922-22_07-Discriminator.pkl']

    myTrainer = Trainer(GPU, data_dir, embed_dir, fonts_num, batch_size, img_size)
    myTrainer.train(max_epoch=max_epoch, schedule=schedule, save_model_path=save_model_path, lr=0.001,
                    log_step=10, fine_tune=True, restore=restore, from_model_path=from_model_path)


# 문자열 입력받고 처리
@app.route('/result', methods=["GET", "POST"])
def register():
    if request.method == 'POST':
        # textArea값 가져오기
        sentence = request.form['textArea']

        # 줄바꿈 처리
        tmp = []
        for i in sentence.split("\r\n"):
            tmp.extend(i)
            if sentence.split("\r\n")[-1] != i:
                tmp.append('\n')

        # 이전에 있던 파일들 삭제
        fileList = glob.glob('static/images/result*.png', recursive=True)
        for file in fileList:
            os.remove(file)

        fileList = glob.glob('static/images/my*.png', recursive=True)
        for file in fileList:
            os.remove(file)

        # 이미지 만들고 이름 반환
        f_name = create_my_font_img(tmp, model_name)

        if lang == 'en':
            return render_template('endPage_' + lang + '.html', sentence=sentence, image_file='images/' + f_name,
                                   addr=addr)
        else:
            return render_template('endPage_' + lang + '.html', sentence=sentence, image_file='images/' + f_name,
                                   addr=addr)
    else:
        return 'false'


# LoadModel
@app.route('/loadModel', methods=["GET", "POST"])
def loadModel():
    global model_name
    if request.method == 'POST':
        # loadModelName값 가져오기
        model_name = request.form['loadModelName']

        folders = os.listdir("model/")

        # 모델이 있으면 마지막으로, 없으면 그대로
        if model_name in folders:
            return render_template('endPage_' + lang + '.html', addr=addr)
        else:
            return render_template('mainpage_' + lang + '.html', addr=addr, load_model_name=model_name,
                                   isModelExist=False)


if __name__ == '__main__':
    # learn_template()
    app.run('0.0.0.0')
#   app.run(debug=False)