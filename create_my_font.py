import cv2
import time
import torch
import numpy as np

from torchvision.utils import save_image
from model import Encoder, Decoder, Generator
from function import sentence_to_src, get_src_pickle, PickledImageProvider, denorm_image, batch_all

# 입력받은 문장을 학습결과를 통해 생성
def create_my_font_img(sentence:str, modelName:str):
    # 문장->단어 리스트로 받기
    src_list = sentence_to_src(sentence)

    # pkl파일 생성, 객체 생성
    get_src_pickle(src_list, 'data/batang', 'data/my_string.obj')
    my_pkl = PickledImageProvider('data/my_string.obj')

    sentence_len = 0
    for c in src_list:
        if c != 'blank' and c != 'new_line':
            sentence_len += 1

    my_pkl_data = my_pkl.data[:]
    batch_iter = batch_all(my_pkl_data)

    for batch in batch_iter:
        ids, images = batch
        # images = images.cuda()

        torch.save(images, 'model/' + modelName + '/my_sentence.pkl')

    GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if GPU != 'cpu':
    #     En = Encoder().cuda()
    #     De = Decoder().cuda()
    # else:
    En = Encoder()
    De = Decoder()

    En.load_state_dict(torch.load('model/'+modelName+'/' + 'Encoder.pkl', map_location=torch.device('cpu')))
    De.load_state_dict(torch.load('model/'+modelName+'/' + 'Decoder.pkl', map_location=torch.device('cpu')))

    my_source = torch.load('model/'+modelName+'/my_sentence.pkl',  map_location=torch.device('cpu'))
    embeddings = torch.load('model/' + 'EMBEDDINGS.pkl',  map_location=torch.device('cpu'))

    L = [30]*sentence_len

    my_sentence_images = Generator(my_source, En, De, embeddings, L, GPU=None)[0]
    for i in range(sentence_len):
        save_image(denorm_image(my_sentence_images.data[i]),  'static/images/my'+str(i)+'.png')

    f_name = create_sentence_img(src_list)
    return f_name

# 손글씨 이미지 붙이기
def create_sentence_img(sentence:list):
    bg = np.full((128, 10), 255, np.uint8)  # 시작
    tap = np.full((128, 60), 255, np.uint8) # 띄어쓰기

    count = 0
    tmp = []
    for k in range(len(sentence)):
        # 공백이면 tap 추가
        if sentence[k] == 'blank':
            count += 1
            bg = np.hstack((bg, tap))

        # 줄바꿈이면 지금까지 bg 저장 후 새로운 라인 만들기
        elif sentence[k] == 'new_line':
            count += 1
            tmp.append(bg)
            bg = np.full((128, 10), 255, np.uint8)

        # 글씨면 bg에 붙이기
        else:
            img = cv2.imread('static/images/my'+str(k-count)+'.png', cv2.IMREAD_GRAYSCALE)[:, 20:108]
            bg = np.hstack((bg, img))

    # 마자막 처리 bg 넣기
    tmp.append(bg)

    w = 0
    h = 128*len(tmp)     # tmp에 저장된 개수
    
    # 최대 가로 길이 구하기
    for k in tmp:
        print(len(k[1]))
        if w < len(k[1]):
            w = len(k[1])

    # 최대 가로/세로 길이만큼 흰 배경 만들기
    bg = np.full((h, w), 255, np.uint8)
    # 배경에 이미지 넣기
    for i in range(len(tmp)):
        bg[128*i:128*(i+1), :len(tmp[i][1])] = tmp[i]
        bg[128*(i+1)-10:128*(i+1)-7, :] = 0

    # 상/하 공백 넣기
    pad_tb = np.full((128, np.shape(bg)[1]), 255, np.uint8)
    bg = np.vstack((pad_tb, bg))
    bg = np.vstack((bg, pad_tb))

    # 좌/우 공백 넣기
    pad_lr = np.full((np.shape(bg)[0], 100), 255, np.uint8)
    bg = np.hstack((pad_lr, bg))
    bg = np.hstack((bg, pad_lr))


    resize_img = cv2.resize(bg, (int(bg.shape[1]*0.35), int(bg.shape[0]*0.35)), interpolation=cv2.INTER_AREA)

    fname = 'result' + str(time.time()) + '.png'
    cv2.imwrite('static/images/'+fname, resize_img)

    return fname