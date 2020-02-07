import cv2
import numpy as np


# 얼굴 노이즈
def face_noise(image) :
    tmp = image.copy()

    # Emboss
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    Kernel_emboss = np.array([[-21, -1, 0], [-1, 1, 1], [0, 1, 21]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_emboss)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # Histogram_Equalization
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    cv2.equalizeHist(tmp, tmp)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # MedianBlur
    tmp2 = cv2.medianBlur(image, 23)

    tmp2 = cv2.cvtColor(tmp2, cv2.COLOR_RGB2GRAY)

    # Roberts
    Kernel_X = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
    Kernel_Y = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])

    grad_x = cv2.filter2D(tmp2, cv2.CV_16S, Kernel_X)
    grad_y = cv2.filter2D(tmp2, cv2.CV_16S, Kernel_Y)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    tmp2 = cv2.addWeighted(abs_grad_x, 10, abs_grad_y, 10, 0)

    tmp2.astype(np.uint8)

    # THRESH_BINARY
    ret, tmp2 = cv2.threshold(tmp2, 140, 255, cv2.THRESH_BINARY)
    tmp2 = cv2.cvtColor(tmp2, cv2.COLOR_GRAY2RGB)

    add_img = cv2.addWeighted(tmp, 1, tmp2, 1, 0, 0, 0)

    # 필터 통과한 이미지 변수에 넣기
    image = add_img.copy()

    return image

# face_black_area_color
def face_black_area_color(image) :
    # 세번째 필터
    # Gamma_correction
    gamma = 50
    lookUpTable = np.empty((1, 256), np.uint8)
    for j in range(256):
        lookUpTable[0, j] = np.clip(pow(j / 255.0, 1.0 / (gamma / 10)) * 255.0, 0, 255)

    tmp = cv2.LUT(image, lookUpTable)

    # Bilateral
    tmp = cv2.bilateralFilter(tmp, 27, 153, 153)

    # #Sharpen
    ksize_sharp = 18
    Kernel_sharpen = np.array(
        [[0, -ksize_sharp, 0], [-ksize_sharp, 1 + 4 * ksize_sharp, -ksize_sharp], [0, -ksize_sharp, 0]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_sharpen)

    # cv2.imshow("bbb",tmp)
    # cv2.waitKey()

    # 필터 통과한 이미지 변수에 넣기
    image = tmp.copy()

    return image

def face_black_area(image):
    tmp = image.copy()

    # 네번째 필터
    # CLAHE
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(tmp)
    clahe = cv2.createCLAHE(clipLimit=0, tileGridSize=(6, 6))
    dst = clahe.apply(l)
    l = dst.copy()
    tmp = cv2.merge((l, a, b))
    tmp = cv2.cvtColor(tmp, cv2.COLOR_LAB2RGB)

    # Emboss
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    Kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_emboss)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # Bilateral
    tmp = cv2.bilateralFilter(tmp, 27, 153, 153)

    # #Sharpen
    ksize_sharp = 1
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    Kernel_sharpen = np.array(
        [[0, -ksize_sharp, 0], [-ksize_sharp, 1 + 4 * ksize_sharp, -ksize_sharp], [0, -ksize_sharp, 0]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_sharpen)

    ###### could not broadcast input array from shape 에러시
    # 차원 관련 문제니 gray2rgb 적용
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
    # 필터 통과한 이미지 변수에 넣기
    image = tmp.copy()

    return image

# 눈썹에 세로 선
def eyebrow_vertical_line(image) :
    tmp = image.copy()

    # Emboss
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    Kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_emboss)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # Histogram_Equalization
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    cv2.equalizeHist(tmp, tmp)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # THRESH_BINARY
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    ret, tmp = cv2.threshold(tmp, 95, 255, cv2.THRESH_BINARY)

    ###### could not broadcast input array from shape 에러시
    # 차원 관련 문제니 gray2rgb 적용
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
    # 필터 통과한 이미지 변수에 넣기
    image = tmp.copy()
    return image


# 눈썹 두줄
def eyebrow_doubleline(image) :
    tmp = image.copy()

    # 세번째 필터

    # Emboss
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    Kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_emboss)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # Histogram_Equalization
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    cv2.equalizeHist(tmp, tmp)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # Gamma_correction
    gamma = 30
    lookUpTable = np.empty((1, 256), np.uint8)
    for j in range(256):
        lookUpTable[0, j] = np.clip(pow(j / 255.0, 1.0 / (gamma / 10)) * 255.0, 0, 255)

    tmp = cv2.LUT(tmp, lookUpTable)

    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)

    # THRESH_BINARY
    ret, tmp = cv2.threshold(tmp, 180, 255, cv2.THRESH_BINARY)
    ###### could not broadcast input array from shape 에러시
    # 차원 관련 문제니 gray2rgb 적용
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # 필터 통과한 이미지 변수에 넣기
    image = tmp.copy()


    return image

# 미간에 콧구멍
def nose_in_b(image) :
    tmp = image.copy()
    # THRESH_BINARY
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    ret, tmp = cv2.threshold(tmp, 55, 255, cv2.THRESH_BINARY)

    ###### could not broadcast input array from shape 에러시
    # 차원 관련 문제니 gray2rgb 적용
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # 필터 통과한 이미지 변수에 넣기
    image = tmp.copy()
    return image


# 코 노이즈
def nose_noise(image) :
    tmp = image.copy()

    # Emboss
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    Kernel_emboss = np.array([[-11, -1, 0], [-1, 1, 1], [0, 1, 11]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_emboss)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # # Histogram_Equalization
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    cv2.equalizeHist(tmp, tmp)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # cv2.imshow("bbb",tmp)
    # cv2.waitKey()

    # 필터 통과한 이미지 변수에 넣기
    image = tmp.copy()

    return image

# 입 아래 가로 검은 선
def mouth_h_b(image) :
    tmp = image.copy()

    # Emboss
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    Kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_emboss)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # Histogram_Equalization
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    cv2.equalizeHist(tmp, tmp)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # Gamma_correction
    gamma = 60
    lookUpTable = np.empty((1, 256), np.uint8)
    for j in range(256):
        lookUpTable[0, j] = np.clip(pow(j / 255.0, 1.0 / (gamma / 10)) * 255.0, 0, 255)

    tmp = cv2.LUT(tmp, lookUpTable)

    # Bilateral
    tmp = cv2.bilateralFilter(tmp, 13, 51, 51)

    # Roberts
    Kernel_X = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
    Kernel_Y = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])

    grad_x = cv2.filter2D(tmp, cv2.CV_16S, Kernel_X)
    grad_y = cv2.filter2D(tmp, cv2.CV_16S, Kernel_Y)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    tmp = cv2.addWeighted(abs_grad_x, 10, abs_grad_y, 10, 0)

    tmp.astype(np.uint8)

    # cv2.imshow("bbb",tmp)
    # cv2.waitKey()

    # 필터 통과한 이미지 변수에 넣기
    image = tmp.copy()
    return image
