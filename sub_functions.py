import face_recognition
from scipy.spatial import distance as dist
import os


# 判断是否闭眼
def eye_close_detection(face_landmarks, ear_threshold):
    left_eye = face_landmarks[0]['left_eye']
    right_eye = face_landmarks[0]['right_eye']

    # ear即eye aspect ratio(眼睛纵横比)，用于判断眼睛是否张开/合上
    ear_left = get_ear(left_eye)
    ear_right = get_ear(right_eye)

    if ear_left <= ear_threshold and ear_right <= ear_threshold:
        return True
    else:
        return False


# 计算眼睛纵横比 eye aspect ratio
def get_ear(eye):

    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


# 找到图像中(面积)最大的脸
def find_main_face(face_locations):
    max_area = 0
    max_face = face_locations[0]
    max_face_location = []
    for face in face_locations:
        area = abs((face[0] - face[2]) * (face[1] - face[3]))
        if area > max_area:
            max_area = area
            max_face = face

    max_face_location.append(max_face)

    return max_face_location


# 人脸匹配，返回对应的名字的下标
def recognition(known_face_encodings, main_face_encoding):
    matches = face_recognition.compare_faces(known_face_encodings, main_face_encoding[0])

    if True in matches:
        index = matches.index(True)
        return index
    else:
        return None


# 判断是否张嘴
def mouth_open_detection(face_landmarks, mar_threshold):
    mouth = face_landmarks[0]['mouth']

    mouth_mar = get_mar(mouth)
    if mouth_mar >= mar_threshold:
        return True
    else:
        return False


# 计算嘴巴纵横比 mouth aspect ratio
def get_mar(mouth):
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
    B = dist.euclidean(mouth[4], mouth[8]) # 53, 57

    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

    # compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)

    # return the mouth aspect ratio
    return mar


# 从文件夹中加载图片
def load_known_persons(path):
    known_faces_encodings = []
    known_faces_names = []

    for roots, dirs, files in os.walk(path):
        for file in files:
            file_fullname = os.path.join(roots, file)
            img = face_recognition.load_image_file(file_fullname)
            face_encoding = face_recognition.face_encodings(img)[0]
            known_faces_encodings.append(face_encoding)
            name = file.split('.')[0]
            known_faces_names.append(name)

    return known_faces_encodings, known_faces_names

