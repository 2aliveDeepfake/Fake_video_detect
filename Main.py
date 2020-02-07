from find_face import load_face_model, detect_face
from find_face_part import f_p_load_models, detect_face_part
from find_fake import detect_fake
import os

# VIDEO_PATH = "./test_videos/test.mp4"

# 모델을 다 불러와서 session 에 올려놓고 시작
# face 모델 불러오기
f_sess, f_detection_graph, f_category_index = load_face_model()
# 가짜 특징 모델 다 불러옴
f_p_sess, f_p_detection_graph, f_p_category_index = f_p_load_models()

# 처리할 비디오 경로
# folder_path = "F:\\fucking_face\\"
folder_path = "./test_videos/"
folder_list = os.listdir(folder_path)

# num = input("영상에서 몇 프레임마다 detect 할지 입력 : ")
# count = input("몇번째 프레임을 확인할지 입력 : ")
num = 10
count = 0
for video_item in folder_list:
    # 영상을 몇 프레임마다 추출할 건지 입력
    number = int(num)
    # 몇번째 프레임을 추출할지 입력
    count_num = int(count)
    print(video_item)
    PATH_TO_VIDEO = folder_path+video_item

    # face_list 가  안넘어가는것 같은데 왤까
    # face_list = str(face_list).replace('],','],\n')
    # 드디어 잘 넘겼다ㅏ str로 바꿨더니 문자 하나하나 i 값으로 구분하네

    # video에서 얼굴 찾는 모델 불러와서 얼굴 넘기기
    face_list, load_face_model_time = \
        detect_face(PATH_TO_VIDEO, number, count_num, f_sess, f_detection_graph, f_category_index)

    # 얼굴에서 눈, 코, 입 찾기
    load_part_model_time = \
        detect_face_part(face_list, PATH_TO_VIDEO, number, count_num, f_p_sess, f_p_detection_graph, f_p_category_index)

    # load_part_model_time = \
    #     detect_fake(face_list, PATH_TO_VIDEO, number, count_num, sess, detection_graph, category_index)
    print ("총 걸린시간 : " + str(round(load_face_model_time + load_part_model_time)))
    print()