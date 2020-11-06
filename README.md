# 人脸识别（含活体检测）

该项目通过读取笔记本电脑内置摄像头的画面，在调用开源人脸识别库 *[face_recognition](https://github.com/ageitgey/face_recognition)* 中的函数的基础上实现了简单的人脸识别及活体检测功能。

> 本项目仅为人脸识别及活体检测功能的具体实现，不包含其中的原理



## 功能简述

1. 找出每一帧图像中的主要人脸（即面积最大的人脸）并进行识别
2. 以眨眼和张嘴动作进行活体判定
3. 若识别过程中主要人脸发生变化，新的脸会被重新认定为假人



## 环境配置

- Python 3.3+ or Python 2.7

请**务必**参考开源项目 *[face_recognition](https://github.com/ageitgey/face_recognition)* 中的需求



## 活体检测思路

- 获取人脸特征点，计算`ear`*(eye aspect ratio, 眼睛纵横比 )*和`mar`*(mouth aspect ratio, 嘴巴纵横比)*
- 当`ear`小于一定阈值`ear_threshold`时，判定人物闭眼
- 当`mar`大于一定阈值`mar_threshold`时，判定人物嘴巴张开
- 眼睛闭上超过一帧后再睁开算作一次眨眼
- 嘴巴张开超过一帧后再合上算作一次张嘴
- 能眨眼且能张嘴才会被认定为真人`real`，否则为假人`fake`

> 眨眼动作过快，摄像头可能无法捕捉到闭眼的图像



## 使用方法

### 一、准备工作

> `face_recognition`库中返回人脸特征点的函数`face_landmarks`返回的字典*(dictionary)*中将嘴巴分为了上嘴唇`“top_lip”`和 `“bottom_lip”` 两个部分。为了直接得到整个嘴巴的特征点，方便后续对嘴巴纵横比的计算，本人新定义了函数`face_landmarks_2`，函数执行内容不变，返回值在`face_landmarks`的基础上加入了键`"mouth"`，对应的值为20个嘴部特征点。

1. 复制以下代码

   ```python
   def face_landmarks_2(face_image, face_locations=None, model="large"):
       
       landmarks = _raw_face_landmarks(face_image, face_locations, model)
       landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]
   
       # For a definition of each point index, see https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
       if model == 'large':
           return [{
               "chin": points[0:17],
               "left_eyebrow": points[17:22],
               "right_eyebrow": points[22:27],
               "nose_bridge": points[27:31],
               "nose_tip": points[31:36],
               "left_eye": points[36:42],
               "right_eye": points[42:48],
               "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
               "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]],
               "mouth": points[48:68] # 此行为新增的内容
           } for points in landmarks_as_tuples]
       elif model == 'small':
           return [{
               "nose_tip": [points[4]],
               "left_eye": points[2:4],
               "right_eye": points[0:2],
           } for points in landmarks_as_tuples]
       else:
           raise ValueError("Invalid landmarks model type. Supported models are ['small', 'large'].")
   ```

   

2. 在本机的python安装目录中找到`face_recognition`库的安装目录 

   > 例：C:\python3\Lib\site-packages\face_recognition 

   

3. 打开该目录中的`api.py`文件，将复制的内容粘贴至最后，保存

   

4. 打开该目录中的`__init__.py`文件，在第7行的末尾加上”, face_landmarks_2“，保存

   示例：
   
   ```python
   from .api import load_image_file, face_locations, batch_face_locations, face_landmarks, face_encodings, compare_faces, face_distance, face_landmarks_2
   ```
   
   



### 二、运行程序

1. 打开本项目中的`main_function.py`文件

2. 找到该行代码，将路径改为本机上存有人脸照片的文件夹路径

   ```python
   # 加载已知的人脸照片
   path = 'C:\\Users\\73113\\Desktop\\SRTP\\demo\\pictures'
   ```

3. 运行程序

4. 按下英文字母`q`推出程序



## 额外说明

本项目中的`sub_functions.py`文件包含了本人额外编写的有关活体检测等功能的函数

