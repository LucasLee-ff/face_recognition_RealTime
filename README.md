



# 人脸识别（含活体检测）

该项目通过读取笔记本电脑内置摄像头的画面，在调用开源人脸识别库 *[face_recognition](https://github.com/ageitgey/face_recognition)* 中的函数的基础上实现了简单的人脸识别及活体检测功能。

> 本项目仅为人脸识别等功能的具体实现，不包含其中的原理



## 目录

- #### [功能简述](#jump1)

- #### [环境配置](#jump2)

- #### [活体检测思路](#jump3)

- #### [使用方法](#jump4)

  - ##### [准备工作](#jump5)
  - ##### [运行程序](#jump6)

- #### [额外说明](#jump7)



## <span id="jump1">功能简述</span>

1. 找出每一帧图像中的主要人脸（即面积最大的人脸）并进行识别
2. 以眨眼和张嘴动作进行活体判定
3. 若识别过程中主要人脸发生变化，新的脸会被重新认定为假人



## <span id="jump2">环境配置</span>

- Python 3.3+ or Python 2.7

请**务必**参考开源项目 *[face_recognition](https://github.com/ageitgey/face_recognition)* 中***Installation***栏目中的的需求，里面有不同系统安装face_recognition的具体教程



## <span id="jump3">活体检测思路</span>

- 获取人脸特征点，计算`ear`*(eye aspect ratio, 眼睛纵横比 )*和`mar`*(mouth aspect ratio, 嘴巴纵横比)*
- 当`ear`小于一定阈值`ear_threshold`时，判定人物闭眼
- 当`mar`大于一定阈值`mar_threshold`时，判定人物嘴巴张开
- 眼睛闭上超过一帧后再睁开算作一次眨眼
- 嘴巴张开超过一帧后再合上算作一次张嘴
- 能眨眼且能张嘴才会被认定为真人`real`，否则为假人`fake`

> 眨眼动作过快，摄像头可能无法捕捉到闭眼的图像



## <span id="jump4">使用方法</span>

### <span id="jump5">一、准备工作</span>

> `face_recognition`库中返回人脸特征点的函数`face_landmarks`返回的字典(dictionary)中将嘴巴分为了上嘴唇`“top_lip”`和 `“bottom_lip”` 两个部分。为了直接得到整个嘴巴的特征点，方便后续对嘴巴纵横比的计算，新定义了函数`face_landmarks_2`，函数执行内容与`face_landmarks`相同，返回值在`face_landmarks`的基础上加入了键`"mouth"`，对应的值为20个嘴部特征点。

1. 复制本项目中`face_landmarks_2.py`文件中的所有代码

2. 在本机的python安装目录中找到`face_recognition`库的安装目录 

   > 例：C:\python3\Lib\site-packages\face_recognition 

   

3. 打开该目录中的`api.py`文件，将复制的内容粘贴至最后，保存

   

4. 打开该目录中的`__init__.py`文件，在`from .api import`的末尾加上”, face_landmarks_2“，保存

   示例：
   
   ```python
   from .api import load_image_file, face_locations, batch_face_locations, face_landmarks, face_encodings, compare_faces, face_distance, face_landmarks_2
   ```
   
   



### <span id="jump6">二、运行程序</span>

1. 打开本项目中的`main_function.py`文件

2. 找到该行代码，将路径改为本机上存有人脸照片的**文件夹**路径

   ```python
   # 例：
   path = 'C:\\Users\\windows\\Desktop\\SRTP_FILE\\demo\\pictures'
   ```

   文件夹内容如下：

   ![Image text](https://github.com/LucasLee-ff/face_recognition_RealTime/tree/master/img_show/example.jpg)

   > 注：加载已知人脸图片后，已知人名列表由人脸图片名自动生成

3. 运行程序

4. 按下英文字母`q`退出程序



## <span id="jump7">额外说明</span>

本项目中`main_function.py`文件为主体，`sub_functions.py`文件包含了额外编写的有关活体检测等功能的函数

