import os
import cv2
import glob
import pickle
import shutil
import numpy as np

from PIL import ImageDraw, ImageFont, Image

# 128x128 흰 배경 만들기
def create_white_bg():
    fname = "white_bg.png"
    white_bg = np.full((128, 128), 255, np.uint8)

    cv2.imwrite(fname, white_bg)

# 2350한글 txt읽고 list 반환
def read_2350_syllables():
    syl = []
    with open("han2350.txt", "r", encoding="utf-8") as f:
        for char in f:
            syl.append(char.strip())
    return syl

# 11172자 한글 만들기
def create_11172syllables():
    # AC00 ~ D7A3
    n1 = 'A B C D'
    n2 = '0 1 2 3 4 5 6 7 8 9 A B C D E F'
    start = 'AC00'
    end = 'D7A3'

    n1 = n1.split(" ")
    n2 = n2.split(" ")

    # A000 ~ DFFF
    syllable = [a + b + c + d
                for a in n1
                for b in n2
                for c in n2
                for d in n2]

    syllable = np.array(syllable)

    # 시작, 끝 인덱스
    s = np.where(start == syllable)
    e = np.where(end == syllable)

    syllable = syllable[s[0][0]:e[0][0]+1]

    return syllable

# 폰트 이미지 만들기
def create_syllables_images(font_path, save_path, all=False):
    # 11172 or 2350
    if all:
        syllables = create_11172syllables()
    else:
        syllables = read_2350_syllables()
    fonts = os.listdir(font_path)

    font_size = 100
    text_pos = (0, 0)

    for font in fonts:
        font_name = font.split('.')[0]
        if font_name not in os.listdir(save_path):
            os.mkdir(save_path + font.split('.')[0])
        else:
            continue

        img_font = ImageFont.truetype(font_path + font, font_size)
        for idx, syllable in enumerate(syllables):
            bg = Image.open('white_bg.png')
            draw = ImageDraw.Draw(bg)
            if all:
                draw.text(text_pos, chr(int(syllable, 16)), font = img_font)
            else:
                draw.text(text_pos, syllable, font=img_font)
            bg.save(save_path + font_name + "/" + font_name + str(idx) + ".png")

            if idx % 500 == 0:
              print(save_path + font_name + str(idx) + ".png")

# 이미지 자르기
def cut_image(img):
    row, col = img.shape
    row1, row2, col1, col2 = (0, 0, 0, 0)

    # 가장 위 점 찾기
    for i in range(row):
        for j in range(col):
            if img[i, j] < 255:
                row1 = i
                break
        if row1 != 0:
            break
    # 가장 아래 점 찾기
    for i in range(row - 1, 0, -1):
        for j in range(col):
            if img[i, j] < 255:
                row2 = i
                break
        if row2 != 0:
            break
    # 가장 왼쪽 점 찾기
    for i in range(col):
        for j in range(row):
            if img[j, i] < 255:
                col1 = i
                break
        if col1 != 0:
            break
    # 가장 오른쪽 점 찾기
    for i in range(col - 1, 0, -1):
        for j in range(row):
            if img[j, i] < 255:
                col2 = i
                break
        if col2 != 0:
            break

    cut_img = img[row1:row2 + 1, col1:col2 + 1]

    return cut_img

# 자른 이미지 생성
def create_cut_img(font_path, save_path):
    font_names = os.listdir(font_path)
    
    for font_name in font_names:
        if font_name in os.listdir(save_path):
            continue
        else:
            os.mkdir(save_path + font_name)
        
        # 자른 이미지 저장
        for i in range(len(os.listdir(font_path + font_name))):
            print(font_path + font_name + "/" + font_name + str(i)  + ".png")
            img = cv2.imread(font_path + font_name + "/" + font_name + str(i) + ".png", cv2.IMREAD_GRAYSCALE)
            img = cut_image(img)
            cv2.imwrite(save_path + font_name + '/' + font_name + str(i) + ".png", img)

            if i % 100 == 0:
                print(save_path + font_name + '/' + font_name + str(i) + ".png")

# 이미지 크기 변환
def resize_img(img):
    h, w = img.shape

    # w/h 축소
    if w > 100 and h > 100:
        if w >= h:
            h = h * 100 / w
            w = 100
        elif w < h:
            w = w * 100 / h
            h = 100

    # 작은 w/h를 100고정
    elif w < 100 or h < 100:
        if w >= h:
            h *= 100 / w
            w = 100
        elif w < h:
            w *= 100 / h
            h = 100

    # 큰 w/h를 100고정
    elif w > 100 or h > 100:
        if w >= h:
            h *= 100 / w
            w = 100
        elif w < h:
            w *= 100 / h
            h = 100

    w = int(w)
    h = int(h)

    img = cv2.resize(img, (w, h), None, interpolation=cv2.INTER_LINEAR)
    return img

# 중앙정렬 이미지 생성
def create_norm_image(cut_image_path, save_path):
    font_names = os.listdir(cut_image_path)
    white_bg = cv2.imread('white_bg.png', cv2.IMREAD_GRAYSCALE)

    for font_name in font_names:
        os.mkdir(save_path + font_name)
        for i in range(len(os.listdir(cut_image_path + font_name))):
            img = cv2.imread(cut_image_path + font_name + '/' + font_name + str(i) + ".png", cv2.IMREAD_GRAYSCALE)
            norm_img = white_bg.copy()

            resized_img = resize_img(img)
            h_padding = int((128 - resized_img.shape[0]) / 2)
            w_padding = int((128 - resized_img.shape[1]) / 2)

            norm_img[h_padding:h_padding + resized_img.shape[0], w_padding:w_padding + resized_img.shape[1]] = resized_img

            cv2.imwrite(save_path + font_name + '/' + font_name + str(i) + ".png", norm_img)
            if i % 500 == 0:
                print(save_path + font_name + '/' + font_name + str(i) + ".png")

# 왼쪽:폰트체/ 오른쪽:바탕체 쌍 이미지 생성
def create_couple_image(norm_img_path, save_path):
    font_names = os.listdir(norm_img_path)

    for font_name in font_names:
        if font_name == 'batang':
            continue
        elif font_name not in font_names:
            os.mkdir(save_path + font_name + '+batang')

        for i in range(len(os.listdir(norm_img_path + font_name))):
            img1 = cv2.imread(norm_img_path + font_name + '/' + font_name + str(i) + '.png', cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(norm_img_path + 'batang/batang' + str(i) + '.png', cv2.IMREAD_GRAYSCALE)
            img3 = np.hstack((img1, img2))

            cv2.imwrite(save_path + font_name + '+batang/' + str(i) + '.png', img3)
            if i % 500 == 0:
                print(save_path + font_name + 'batang/' + str(i) + '.png')


# 생성한 폰트 이미지들 train 데이터로 뽑기
def select_train_data(coupled_img_path, save_path, size):
    fonts = os.listdir(coupled_img_path)

    l = 0
    for font in fonts:
        k = 0
        total_size = len(os.listdir(coupled_img_path + font))
        a = np.random.choice(total_size, size, replace=False)

        for i in range(size):
            shutil.copyfile(coupled_img_path + font + '/' + str(a[i]) + ".png",
                            save_path + "%d_%04d" % (l, k) + ".png")
            k += 1

            if i % 500 == 0:
                print(save_path + str(i) + '.png')
        l += 1

# train pkl파일 생성
def create_train_pickle_obj(train_img_path, train_obj_path):
    imgs_path = glob.glob(train_img_path, "*.png")
    print(imgs_path)

    with open(train_obj_path, 'wb') as pkl:
        print('train data num:', len(imgs_path))
        count = 0

        for p in imgs_path:
            label = int(os.path.basename(p).split("_")[0])
            charid = int(os.path.basename(p).split("_")[1].split(".")[0])
            with open(p, 'rb') as f:
                img_bytes = f.read()
                data = (label, charid, img_bytes)
                pickle.dump(data, pkl)
                count += 1
                if count % 100 == 0:
                    print("%d imgs saved in train.obj" % count)

# train data, pkl 생성
def create_train_data():
    train_data_dir = 'data/train_data/'
    if 'train_data' not in os.listdir('data/'):
       os.mkdir(train_data_dir)
       os.mkdir(train_data_dir + 'ori_img')
       os.mkdir(train_data_dir + 'cut_img')
       os.mkdir(train_data_dir + 'norm_img')
       os.mkdir(train_data_dir + 'coupled_img')
       os.mkdir(train_data_dir + 'all_coupled_img')

    create_white_bg()
    create_syllables_images('data/fonts/', train_data_dir + 'ori_img/')
    create_cut_img(train_data_dir + 'ori_img/', train_data_dir + 'cut_img/')
    create_norm_image(train_data_dir + 'cut_img/', train_data_dir + 'norm_img/')
    create_couple_image(train_data_dir + 'norm_img/', train_data_dir + 'coupled_img/')
    select_train_data(train_data_dir + 'coupled_img/', train_data_dir + 'all_coupled_img/', 2350)
    create_train_pickle_obj(train_data_dir + 'all_coupled_img/', train_data_dir + 'train.obj')


# 템플릿 파일 저장 폴더 생성
def create_dir(syll_path):
    list = os.listdir(syll_path)
    if 'cut_syll' in list:
        return
    else:
        os.mkdir(syll_path + '/cut_syll')
        os.mkdir(syll_path + '/norm_syll')
        os.mkdir(syll_path + '/coupled_syll')

# 템플릿 이미지 자르기
def create_my_cut_syllables(my_data_path, save_path):
    my_data_path = my_data_path
    save_path = save_path

    for i in range(len(os.listdir(my_data_path))):
        img = cv2.imread(my_data_path + '/' + str(i) + ".png", cv2.IMREAD_GRAYSCALE)
        if type(img) == type(None):
            return

        img = cut_image(img)
        cv2.imwrite(save_path+ '/' + str(i) + ".png", img)

        if i % 10 == 0:
            print(save_path + '/' + str(i) + ".png")

    print(len(os.listdir(my_data_path)))

# 템플릿 이미지 크기 변환
def resize_my_img(img):
    h, w = img.shape

    if w < 71 and w > 60:
        w = w * 1.2
    elif w > 50:
        w = w * 1.3
    elif w > 40:
        w = w * 1.4
    else:
        w = w * 1.5

    if h > 75:
        h = h * 1.1
    elif h > 65:
        h = h * 1.2
    elif h > 55:
        h = h * 1.3
    elif h > 45:
        h = h * 1.4
    else:
        h = h * 1.5

    w = int(w)
    h = int(h)

    img = cv2.resize(img, (w, h), None, interpolation=cv2.INTER_LINEAR)

    return img

# 템플릿 이미지 중앙 정렬
def create_my_norm_img(my_cut_img_path, save_path):
    my_cut_img_path = my_cut_img_path
    save_path = save_path
    white_bg = cv2.imread('data/white_bg.png', cv2.IMREAD_GRAYSCALE)

    for i in range(len(os.listdir(my_cut_img_path))):
        img = cv2.imread(my_cut_img_path + '/' + str(i) + ".png", cv2.IMREAD_GRAYSCALE)
        norm_img = white_bg.copy()

        resized_img = resize_my_img(img)
        h_padding = int((128 - resized_img.shape[0]) / 2)
        w_padding = int((128 - resized_img.shape[1]) / 2)

        norm_img[h_padding:h_padding + resized_img.shape[0], w_padding:w_padding + resized_img.shape[1]] = resized_img

        cv2.imwrite(save_path + '/' + str(i) + ".png", norm_img)
        if i % 10 == 0:
            print(save_path + '/' + str(i) + ".png")

# 템플릿 바탕체 이미지 만들기
def create_template_batang_syllables_img(font_path, txt_path, save_path):
    if 'batang' not in os.listdir('data/'):
        os.mkdir(save_path)
        os.mkdir(save_path + 'norm_img')
        os.mkdir(save_path + 'img')
        os.mkdir(save_path + 'cut_img')
    else:
        return

    with open(txt_path, 'r', encoding='UTF-8') as f:
        syllables = f.readlines()
        for idx, syllable in enumerate(syllables):
            syllables[idx] = syllable.strip()
        print(syllables)

    font_size = 100
    text_pos = (0, 0)
    imgfont = ImageFont.truetype(font_path, font_size)
    for idx, syllable in enumerate(syllables):
        bg = Image.open('data/white_bg.png')
        draw = ImageDraw.Draw(bg)
        draw.text(text_pos, syllable, font=imgfont)
        bg.save(save_path + '/img/' + str(idx) + ".png")

    img_names = os.listdir(save_path + '/img')
    for img_name in img_names:
        img = cv2.imread(save_path + '/img/' + img_name, cv2.IMREAD_GRAYSCALE)
        img = cut_image(img)

        cv2.imwrite(save_path + '/cut_img/' + img_name, img)

    cut_images = os.listdir(save_path + '/cut_img')
    white_bg = cv2.imread('data/white_bg.png', cv2.IMREAD_GRAYSCALE)
    for img_name in cut_images:
        img = cv2.imread(save_path + '/cut_img/' + img_name, cv2.IMREAD_GRAYSCALE)
        norm_img = white_bg.copy()

        resized_img = resize_img(img)
        h_padding = int((128 - resized_img.shape[0]) / 2)
        w_padding = int((128 - resized_img.shape[1]) / 2)

        norm_img[h_padding:h_padding + resized_img.shape[0],
        w_padding:w_padding + resized_img.shape[1]] = resized_img

        cv2.imwrite(save_path + '/norm_img/' + img_name, norm_img)

# 학습할 손글씨 이미지 쌍 만들기
def create_my_coupled_img(norm_img_path, my_norm_img_path, save_path):
    img_names = os.listdir(norm_img_path)

    for img_name in img_names:
        img1 = cv2.imread(my_norm_img_path + img_name, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(norm_img_path + img_name, cv2.IMREAD_GRAYSCALE)
        img3 = np.hstack((img1, img2))

        cv2.imwrite(save_path + img_name, img3)

# 학습을 위한 피클 파일 만들기
def create_template_pickle_obj(train_img_path, train_obj_path):
    imgs_path = glob.glob(train_img_path, "*.png")

    with open(train_obj_path, 'wb') as pkl:
        print('train data num:', len(imgs_path))
        count = 0

        for p in imgs_path:
            label = 30
            charid = int(os.path.basename(p).split(".")[0])
            with open(p, 'rb') as f:
                img_bytes = f.read()
                example = (label, charid, img_bytes)
                pickle.dump(example, pkl)
                count += 1
                if count % 100 == 0:
                    print("%d imgs saved in train.obj" % count)

# 손글씨 전체 이미지 생성
def create_template_data(model_name:str):
    syll_path = 'data/template_syll/' + model_name + '/'
    print(syll_path)
    create_dir(syll_path)
    create_my_cut_syllables(syll_path + 'ori/', syll_path + 'cut_syll/')
    create_my_norm_img(syll_path + 'cut_syll/', syll_path + 'norm_syll/')
    # create_template_batang_syllables_img('data/batang.ttc', 'data/rand210.txt', 'data/batang/')
    create_my_coupled_img('data/batang/norm_img/', syll_path + 'norm_syll/', syll_path + 'coupled_syll/')
    create_template_pickle_obj(syll_path + 'coupled_syll/', syll_path + 'train.obj')

