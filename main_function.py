import face_recognition
import cv2
import sub_functions


ear_threshold = 0.25
mar_threshold = 0.75

# 加载已知人脸的照片
path = 'C:\\Users\\73113\\Desktop\\SRTP\\demo\\pictures'

# 编码文件夹下所有照片中的人脸，人名默认为每张照片的名字，返回值作为人脸识别的参照组
known_faces_encodings, known_faces_names = sub_functions.load_known_persons(path)


def main():
    video_capture = cv2.VideoCapture(0)

    total_blinks = 0
    total_mouth_open = 0
    count_eye = 0
    count_mouth = 0

    person = "fake"
    real = False
    mouth_can_open = False
    can_blink = False

    first_loop = True

    font = cv2.FONT_HERSHEY_DUPLEX

    while True:
        ret, frame = video_capture.read(0)
        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame)

        # 若摄像头中检测不到人脸则跳过此次循环
        if len(face_locations) == 0:
            continue

        # 找出主要的人脸位置
        main_face_location = sub_functions.find_main_face(face_locations)

        top = main_face_location[0][0]
        right = main_face_location[0][1]
        bottom = main_face_location[0][2]
        left = main_face_location[0][3]

        # 获取主要人脸的特征点
        main_face_landmarks = face_recognition.face_landmarks_2(rgb_frame, main_face_location)

        # -----------------------人脸识别-----------------------
        # index为已知编码列表中与视频人脸编码相匹配的已知编码的下标
        main_face_encoding = face_recognition.face_encodings(rgb_frame, main_face_location)
        index = sub_functions.recognition(known_faces_encodings, main_face_encoding)
        if index is not None:
            name = known_faces_names[index]
        else:
            name = "unknown"

        # 把本帧中检测到的脸与上一帧的脸作对比，如果与上一帧中不是同一个人，重新认定为假人
        if not first_loop:
            match_last_face = face_recognition.compare_faces(face_encoding_backup, main_face_encoding[0])
            if not match_last_face[0]:
                count_eye = 0
                total_blinks = 0
                can_blink = False

                count_mouth = 0
                total_mouth_open = 0
                mouth_can_open = False

                person = "fake"
                real = False
            face_encoding_backup = main_face_encoding

        if first_loop:
            face_encoding_backup = main_face_encoding

        # 框脸
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
        cv2.rectangle(frame, (left, bottom - 20), (right, bottom + 30), (0, 0, 255), cv2.FILLED)

        # 标名字
        cv2.putText(frame, name, (left + 6, bottom - 1), font, 0.8, (255, 255, 255), 1)

        # 标注是真人还是假人
        cv2.putText(frame, person, (left + 6, bottom + 24), font, 0.8, (255, 255, 255), 1)

        # -----------------------活体检测-----------------------

        # 若此帧中的眼睛闭合，则count加1
        # 眼睛闭合时间大于等于2帧后再睁开才算作一次眨眼
        if sub_functions.eye_close_detection(main_face_landmarks, ear_threshold):
            count_eye += 1
        else:
            if count_eye >= 1:
                total_blinks += 1
                count_eye = 0
                can_blink = True

        # 张嘴检测
        if sub_functions.mouth_open_detection(main_face_landmarks, mar_threshold):
            count_mouth += 1
        else:
            if count_mouth >= 1:
                total_mouth_open += 1
                count_mouth = 0
                mouth_can_open = True

        # 能眨眼且张嘴即为真人
        if can_blink and mouth_can_open:
            person = "real"
            real = True

        cv2.putText(frame, "Total Blinks: {}".format(total_blinks), (25, 50), font, 0.8, (0, 0, 255), 1)
        cv2.putText(frame, "Total Mouth Open: {}".format(total_mouth_open), (25, 80), font, 0.8, (0, 0, 255), 1)
        cv2.putText(frame, "press 'q' to quit", (25, 110), font, 0.7, (255, 0, 0), 1)
        cv2.imshow('Video', frame)

        first_loop = False

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
