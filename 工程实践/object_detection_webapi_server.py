#!/usr/bin/env python
# coding: utf-8

# Object Detection WEB API Server
# ==
# 欢迎使用Object Detection WEB API Server。
# 
# 该文件会启动webapi服务器，使用官方收集的预训练模型，接受客户端通过POST上传的图片并检测其中的目标，以json的格式返回预测结果。
# 
# - 确保从[TensorFlow Models
# ](https://github.com/tensorflow/modelsd)拉取Tensorflow Models放置在Tensorflow目录下(我估计不用下全，只下载Object Detection应该也可以)。Tensorflow目录可通过
# 
# - 按照[Tensorflow Object Detection API安装步骤](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)安装Tensorflow Object Detection API
# 
# - 安装flask
# ```pip install flask```
#     
# - 安装PIL
# ```pip install Pillow```
# 

# # Imports

# In[ ]:


import numpy as np
import tensorflow as tf
import io, os, sys

from distutils.version import StrictVersion
from collections import defaultdict
from PIL import Image

# from matplotlib import pyplot as plt

TF_PATH = os.path.split(tf.__file__)[0]
TF_MR_PATH = os.path.join(TF_PATH, 'models', 'research')
TF_SLIM_PATH = os.path.join(TF_MR_PATH, 'slim')
sys.path.append(TF_MR_PATH)
sys.path.append(TF_SLIM_PATH)
from tensorflow.models.research.object_detection.utils import label_map_util
from tensorflow.models.research.object_detection.utils import ops as utils_ops
from tensorflow.models.research.object_detection.utils import visualization_utils as vis_util

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# In[ ]:


from flask import Flask, request, jsonify
import base64
import uuid

# # Variables can be change

# In[ ]:


TF_OD_PATH = os.path.join(TF_MR_PATH, 'object_detection')

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# List of the strings that is used to add correct label for each box.
LABELS_PATH = os.path.join(TF_OD_PATH, 'data', 'mscoco_label_map.pbtxt')

# # Model preparation

# ## Variables
# 
# 任何使用`export_inference_graph.py`工具导出的模型，都可以通过修改`PATH_TO_FROZEN_GRAPH`变量指向新的.pb文件（模型）来加载
# 
# 我们默认使用"SSD with Mobilenet"模型，可以访问[detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)查看其他可以具有不同速度和精度的开箱使用模型

# In[ ]:


# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILENAME = MODEL_NAME + '.tar.gz'
MODEL_PATH = os.path.join(TF_OD_PATH, MODEL_FILENAME)
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
FROZEN_GRAPH_PATH = os.path.join(TF_OD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# ## Download Model and Extract frozen_inference_graph.pb
# 
# 如果模型文件存在将会加载，否则将会解压或者下载并解压

# In[ ]:


if not os.path.exists(FROZEN_GRAPH_PATH):
    if not os.path.exists(MODEL_PATH):
        import six.moves.urllib as urllib

        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILENAME, MODEL_PATH)

    import tarfile

    tar_file = tarfile.open(MODEL_PATH)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, TF_OD_PATH)


# ## Load a (frozen) Tensorflow model into memory.

# In[ ]:


def load_model(PATH_TO_FROZEN_GRAPH, graph=None):
    if graph is None:
        graph = tf.Graph()

    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[ ]:


category_index = label_map_util.create_category_index_from_labelmap(LABELS_PATH, use_display_name=True)


# # Detection function

# 对单个图片进行推理预测，并返回模型预测结果

# In[ ]:


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


# 将模型预测结果转化为更加易于理解和传输的结构化格式

# In[ ]:


def convert_to_structure_format(
        boxes,
        classes,
        scores,
        category_index,
        image_shape=None,
        use_normalized_coordinates=True,
        max_boxes=None,
        min_score_thresh=0.5
):
    """
    Args:
      boxes: a numpy array of shape [N, 4]
      classes: a numpy array of shape [N]. Note that class indices are 1-based,
        and match the keys in the label map.
      scores: a numpy array of shape [N] or None.
      category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
      use_normalized_coordinates: whether boxes is to be interpreted as
        normalized coordinates or not.
      max_boxes: maximum number of boxes. If None, convert all boxes.
      min_score_thresh: minimum score threshold for a box to be convert

    Returns:
      dict which contain number of objects and a list about object's name, box, score, etc
    """

    if not max_boxes:
        max_boxes = boxes.shape[0]

    #   image_shape-->(h, w) in Numpy, image_size-->(w, h) in PIL
    if use_normalized_coordinates:
        #     im_width, im_height = image_size
        im_height, im_width = image_shape
    else:
        im_height, im_width = (1, 1)

    objects = [{"name": category_index[classes[i]]['name'] if classes[i] in category_index.keys() else "",
                "bndbox": {"xmin": int(boxes[i][1] * im_width), "ymin": int(boxes[i][0] * im_height),
                           "xmax": int(boxes[i][3] * im_width), "ymax": int(boxes[i][2] * im_height)},
                "score": float(scores[i])}
               for i in range(min(max_boxes, boxes.shape[0]))
               if scores[i] is None or scores[i] > min_score_thresh]

    #   return {"number":len(objects), "objects":objects}
    return objects


# print(result)


# 对nparray格式的图片推理预测，并返回结构化结果和可视化效果（可选）

# In[ ]:


def predict_on_image_np(image_np, graph, visualization=False):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    output_dict = run_inference_for_single_image(image_np, graph)
    #     print(image_np.shape)

    result_list = convert_to_structure_format(
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        image_np.shape[-3:-1],
        use_normalized_coordinates=True)

    if visualization:
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            max_boxes_to_draw=None,
            line_thickness=8)
    else:
        image_np = None

    return result_list, image_np


# # Flask setting
# 设置flask服务器 
# 
# ## API说明
# 检测图片中的目标的位置、种类、可信度
# ### 请求说明
# - HTTP方法：`POST`
# - 请求URL：`/api/object_detect/predict`
# - Header:
# 
# |参数|值|
# |:---|:--|
# Content-Type|application/x-www-form-urlencoded
# 
# - Body:
# 
# |参数|是否必选|类型|可选范围|说明|
# |:---|:----|:---|:----|:---|
# image|true|string|-|图像数据，base64编码
# image_visual|false|boolean|-|是否返回可视化结果。默认false
# 
# ### 返回说明
# 
# 返回结果是json格式
# 
# #### 返回参数
# 
# |参数|是否必选|类型|说明|
# |:-|:-|:-|:-|
# |log_id|是|UUID|唯一的log id，用于问题定位|
# |result|是|list|预测结果|
# |+bndbox|是|字典|box信息|
# |++xmax|是|int|box右下角的水平坐标|
# |++xmin|是|int|box左上角的水平坐标|
# |++ymax|是|int|box右下角的垂直坐标|
# |++ymin|是|int|box左上角的垂直坐标|
# |+name|是|string|目标类别|
# |+score|是|float|评分，可以理解为置信度|
# |result_num|是|int|检测出目标数目|
# |success|是|boolean|是否成功预测|
# |image_visual|否|string|base64编码的可视化结果|
# 
# #### 返回示例
# ```
# {'log_id': 'c4808689-f3f1-4d01-907c-2fa662626f8b',
#  'result': [{'bndbox': {'xmax': 323, 'xmin': 19, 'ymax': 554, 'ymin': 24},
#              'name': 'dog',
#              'score': 0.9406907558441162},
#             {'bndbox': {'xmax': 996, 'xmin': 412, 'ymax': 588, 'ymin': 69},
#              'name': 'dog',
#              'score': 0.9345026612281799}],
#  'result_num': 2,
#  'success': True}
#  ```

# In[ ]:


app = Flask(__name__)
URL_PRED = "/api/object_detect/predict"


@app.route("/")
def homepage():
    return "Welcome to the object detection REST API!\nPlease use " + URL_PRED


import gevent


def b64decode(image):
    image_b64decode = base64.b64decode(image)
    # gr3.switch(image_b64decode)
    return image_b64decode


def bio(image):
    imageIO = io.BytesIO(image)
    return imageIO


def image_open(image):
    image = Image.open(image)
    return image


@app.route(URL_PRED, methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        res_dict = {'success': False}
        visualization = request.form.get("visual")

        if request.form.get("image"):
            image_b64 = request.form.get("image")
            image_b64decode = base64.b64decode(image_b64)
            imageIO = io.BytesIO(image_b64decode)
            image = Image.open(imageIO)
            image_np = np.asarray(image).copy() if visualization else np.asarray(image)

            res_dict['result'], res_image_np = predict_on_image_np(image_np, detection_graph, visualization)
            res_dict['result_num'] = len(res_dict['result'])

            if visualization:
                res_image = Image.fromarray(res_image_np)
                img_buffer = io.BytesIO()
                res_image.save(img_buffer, format='JPEG')
                res_image_b64 = base64.b64encode(img_buffer.getvalue())
                res_dict['image_visual'] = res_image_b64.decode('utf-8')

            res_dict['success'] = True
            res_dict['log_id'] = str(uuid.uuid4())

        #         print(res_dict)
        return jsonify(res_dict)
    elif request.method == "GET":
        return "Please use POST method."
    else:
        return "Please POST a image."


from flask import Flask, redirect, url_for


@app.route('/api/object_detect_car/predict', methods=["GET", "POST"])
def predict_car():
    return redirect(url_for('predict'))


# # Start web api server

# In[ ]:


if __name__ == '__main__':
    print("Loading model...")
    detection_graph = load_model(FROZEN_GRAPH_PATH)
    print("Starting web api server...")
    app.run(debug=True,threaded=True)
