import os
import cv2

# 템플릿 불러오기
def load_template():
    template = cv2.imread("static/images/template.png", cv2.IMREAD_GRAYSCALE)

    return template

# 템플릿 각 좌표 찾기
def find_axis(model_name:str):
    print('find_axis_model_name: ', model_name)
    template = load_template()

    _, img_binary = cv2.threshold(template, 150, 255, cv2.THRESH_BINARY)
    ret_img = img_binary.copy()

    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    # contours -> [1][2][3][4]
    # [1] = contour 번호
    # [2] = n번째 컨투어의 꼭지점 (0:좌상, 1:좌하, 2:우하, 3:우상)
    # [3] = 0
    # [4] = 좌표(0:x, 1:y)

    # 100보다 작은 컨투어는 제거
    area_set = set()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area / 100 < 2:
            continue
        area_set.add(area)


    area_list = list(area_set)    # 중복 제거한 전체 area list
    area_list.sort(reverse=True)    # 큰 순서 정렬

    # 가장 큰 area 2개 삭제
    area_set.remove(area_list[0])
    area_set.remove(area_list[1])
    area_list = area_list[2:]

    # 최대/최소 area 지정
    max_size = area_list[0] + area_list[0]*0.10
    min_size = area_list[0] - area_list[0]*0.10

    # 필요없는 작은 area 제거
    for i in area_list:
        if i < min_size or i > max_size:
            area_set.remove(i)

    # 크기에 맞는 컨투어 번호 저장
    idx = 0
    idx_list = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area in area_set:
            idx_list.append(idx)
        idx += 1

    # 컨투어 번호 정렬
    k=0
    coord_list = []
    for i in idx_list:
        # 각 컨투어의 좌상단, 우하단 좌표 찾기
        x_min, x_max = contours[i][0][0][0], contours[i][0][0][0]
        y_min, y_max = contours[i][0][0][1], contours[i][0][0][1]
        for j in range(1, len(contours[i])):
            if contours[i][j][0][0] > x_max:
                x_max = contours[i][j][0][0]
            if contours[i][j][0][0] < x_min:
                x_min = contours[i][j][0][0]
            if contours[i][j][0][1] > y_max:
                y_max = contours[i][j][0][1]
            if contours[i][j][0][1] < y_min:
                y_min = contours[i][j][0][1]

        coord_list.append([[x_min, y_min], [x_max, y_max]])
        k += 1

    # 좌상단 좌표 리스트
    min_list = []
    for a in coord_list:
        min_list.append(a[0])

    # y좌표 기준 정렬
    min_list.sort(key=lambda x: (x[1], x[0]))
    print(len(min_list))
    
    last = []
    tmp_list = []
    size = 210
    n_row = 15

    # y축값에 따라 15개의 리스트로 last에 저장
    for i in range(size):
        tmp_list.append(min_list[i])
        if i != 0 and (i+1) % n_row == 0:
            last.append(tmp_list.copy())
            tmp_list.clear()

    for i in range(len(last)):
        last[i].sort()
    print(last)

    # 정렬된 순서대로 last_list에 좌표값 넣기
    last_list = []
    for i in last:
        for j in i:
            for a in coord_list:
                if a[0] == j:
                    last_list.append(a)
    print(len(last_list))

    # 이미지 저장
    dir_list = os.listdir('data/template_syll/')

    f_name = model_name
    if f_name not in dir_list:
        os.mkdir('data/template_syll/'+model_name)
        os.mkdir('data/template_syll/'+model_name + '/ori')


    k = 0
    for coord in last_list:
        img1 = ret_img[coord[0][1] + 10:coord[1][1] - 10, coord[0][0] + 10:coord[1][0] - 10]
        cv2.imwrite("data/template_syll/" + model_name + '/ori/' + str(k) + ".png", img1)
        k += 1


if __name__ == '__main__':
    find_axis('my2')