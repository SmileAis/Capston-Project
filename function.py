import cv2
import torch
import random
import pickle
import numpy as np
import torch.nn as nn

from io import BytesIO
from imageio import imread

# 2진 파일 읽기
def bytes_to_file(bytes_img):
    return BytesIO(bytes_img)

# pkl 데이터 파일 읽기
class PickledImageProvider(object):
    def __init__(self, obj_path):
        self.obj_path = obj_path
        self.data = self.load_pickled_data()

    def load_pickled_data(self):
        with open(self.obj_path, "rb") as f:
            data = list()
            while True:
                try:
                    e = pickle.load(f)
                    data.append(e)
                except EOFError:
                    break
                except Exception:
                    pass
            return data

# 이미지 src/trg 나누기
def read_split_image(img):
    img_ST = imread(img).astype(np.float64)
    mid = int(img_ST.shape[1] / 2)

    img_A = img_ST[:, :mid]  # target
    img_B = img_ST[:, mid:]  # source

    return img_A, img_B

# 이미지확대/이동
def shift_and_resize_image(img, shift_x, shift_y, nw, nh):
    w, h = img.shape
    new_img = cv2.resize(img, (nw, nh))
    return new_img[shift_x:shift_x + w, shift_y:shift_y + h]

def normalize_image(img):
    normalized = (img / 127.5) - 1.
    return normalized

def denorm_image(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# 임베딩 초기화
def init_embedding(embedding_num, embedding_dim, stddev=0.01):
    embedding = torch.randn(embedding_num, embedding_dim) * stddev
    embedding = embedding.reshape((embedding_num, 1, 1, embedding_dim))
    return embedding

# id값에 따른 임베딩값 찾기
def embedding_lookup(embeddings, embedding_ids, GPU=False):
    batch_size = len(embedding_ids)
    embedding_dim = embeddings.shape[3]
    local_embeddings = []
    for id_ in embedding_ids:
        if GPU:
            local_embeddings.append(embeddings[id_].cpu().numpy())
        else:
            local_embeddings.append(embeddings[id_].data.numpy())
    local_embeddings = torch.from_numpy(np.array(local_embeddings))
    if GPU:
        local_embeddings = local_embeddings.cuda()
    local_embeddings = local_embeddings.reshape(batch_size, embedding_dim, 1, 1)

    return local_embeddings

# Convolution
def conv2d(c_in, c_out, k_size=3, stride=2, pad=1, bn=True, lrelu=True, leak=0.2):
    layers = []
    if lrelu:
        layers.append(nn.LeakyReLU(leak))
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

# DeConvolution
def deconv2d(c_in, c_out, k_size=3, stride=1, pad=1, bn=True, dropout=False, p=0.5):
    layers = []
    layers.append(nn.LeakyReLU(0.2))
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    if dropout:
        layers.append(nn.Dropout(p))
    return nn.Sequential(*layers)

# FullyConnected
def fc(input_size, output_size):
    return nn.Linear(input_size, output_size)

# 학습결과 확인 시 사용
def process(binary_img):
    img = bytes_to_file(binary_img)
    img = imread(img).astype(np.float64)
    img = normalize_image(img)
    return [img]

def batch_all(data):
    id = [e[0] for e in data]
    image = [process(e[1]) for e in data]
    image = np.array(image).astype(np.float32)
    image = torch.from_numpy(image)
    yield [id, image]


# 문장 -> 단어 리스트
def sentence_to_list(sentence:str):
    sentence_list = []
    for c in sentence:
        sentence_list.append(c)
    return sentence_list

# 단어 리스트 -> 유니코드 리스트
def char_to_unicode(s_list:list):
    uni_list = []
    for c in s_list:
        uni_list.append(hex(ord(c)))
    print(uni_list)
    return uni_list


# 공백: 0x20  \n: 0xa
def uni_to_num(uni_list:list):
    num_list = []
    for uni in uni_list:
        uni = int(uni, 16)
        if uni == 0x20:
            num_list.append(-1)
        elif uni == 0xa:
            num_list.append(-2)
        else:
            num_list.append(uni - 0xac00)
    return num_list

# 숫자 -> 파일명
def get_src_list(num_list:list):
    src_list = []
    for num in num_list:
        if num == -1:
            src_list.append('blank')
        elif num == -2:
            src_list.append('new_line')
        else:
            src_list.append('batang'+str(num))
    return src_list

# sentence -> 파일명
def sentence_to_src(sentence:str):
    sentence_list = sentence_to_list(sentence)
    uni_list = char_to_unicode(sentence_list)
    num_list = uni_to_num(uni_list)
    src_list = get_src_list(num_list)
    return src_list

# src_list -> pkl
def get_src_pickle(src_list:list, dir_path, save_path):
    with open(save_path, 'wb') as pkl:
        count = 0
        for src in src_list:
            if src == 'blank' or src == 'new_line':
                continue
            else:
                id = src.split('batang')[1]

            path = dir_path + '/' +str(src) + '.png'
            with open(path, 'rb') as f:
                img_bytes = f.read()
                example = (id, img_bytes)

                pickle.dump(example, pkl)
                count += 1
    return

# 학습데이터 읽기
class TrainDataProvider(object):
    def __init__(self, data_dir, data_name="train.obj"):
        self.data_dir = data_dir
        self.obj_path = self.data_dir + data_name
        self.obj = PickledImageProvider(self.obj_path)

        print("%s data -> %d" % (data_name, len(self.obj.data)))

    def get_iter(self, batch_size, shuffle=True, augment=True):
        obj_data = self.obj.data[:]
        if shuffle:
            np.random.shuffle(obj_data)

        return get_batch_iter(obj_data, batch_size, augment=augment)

    def get_total_batch_num(self, batch_size):
        return int(np.ceil(len(self.obj.data) / float(batch_size)))

# 이미지 배치 단위로 읽기
def get_batch_iter(examples, batch_size, augment):
    def process(img):
        img = bytes_to_file(img)
        img_A, img_B = read_split_image(img)

        # 이미지 augmentation
        if augment:
            w, h = img_A.shape
            multi = random.uniform(1.00, 1.15)
            new_w = int(multi * w) + 1
            new_h = int(multi * h) + 1
            shift_x = int(np.ceil(np.random.uniform(0.01, new_w - w)))
            shift_y = int(np.ceil(np.random.uniform(0.01, new_h - h)))
            img_A = shift_and_resize_image(img_A, shift_x, shift_y, new_w, new_h)
            img_B = shift_and_resize_image(img_B, shift_x, shift_y, new_w, new_h)
        img_A = normalize_image(img_A)
        img_A = img_A.reshape(1, len(img_A), len(img_A[0]))
        img_B = normalize_image(img_B)
        img_B = img_B.reshape(1, len(img_B), len(img_B[0]))

        return np.concatenate([img_A, img_B], axis=0)


    def batch_iter():
        for i in range(0, len(examples), batch_size):
            batch = examples[i: i + batch_size]
            labels = [e[0] for e in batch]
            charid = [e[1] for e in batch]
            image = [process(e[2]) for e in batch]
            image = np.array(image).astype(np.float32)
            image = torch.from_numpy(image)

            yield [labels, charid, image]

    return batch_iter()