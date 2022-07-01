import io
import re
import cv2
import json
import onnx
import base64
import numpy as np
import onnxruntime as ort

from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

#####################################################################################################

class CFG(object):
    def __init__(self, infer_type: str):
        self.infer_type = infer_type
        self.ort_session = None

        if re.match(r"^classify$", self.infer_type, re.IGNORECASE):
            self.path = "static/classifier.onnx"
            self.size = 768
            self.labels = json.load(open("static/labels_cls.json", "r"))
        
        elif re.match(r"^detect$", self.infer_type, re.IGNORECASE):
            self.path = "static/detector.onnx"
            self.labels = json.load(open("static/labels_det.json", "r"))
        
        elif re.match(r"^segment$", self.infer_type, re.IGNORECASE):
            self.path = "static/segmenter.onnx"
            self.size = 520
            self.labels = json.load(open("static/labels_seg.json", "r"))
    
    def setup(self):
        model = onnx.load(self.path)
        onnx.checker.check_model(model)
        self.ort_session = ort.InferenceSession(self.path)

    def infer(self, image: np.ndarray, disp_image=None):

        if re.match(r"^classify$", self.infer_type, re.IGNORECASE):
            image = image / 255
            image = cv2.resize(src=image, dsize=(self.size, self.size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
            for i in range(image.shape[0]):
                image[i, :, :] = (image[i, :, :] - MEAN[i]) / STD[i]
            image = np.expand_dims(image, axis=0)
            input = {self.ort_session.get_inputs()[0].name : image.astype("float32")}

            return self.labels[str(np.argmax(self.ort_session.run(None, input)))].split(",")[0].title()
        
        elif re.match(r"^detect$", self.infer_type, re.IGNORECASE):
            h, w, _ = image.shape
            image = np.expand_dims(image, axis=0)
            input = {self.ort_session.get_inputs()[0].name : image.astype("uint8")}
            
            result = self.ort_session.run(None, input)
            boxes, labels, _, num_detections = result[0].squeeze(), \
                                                    result[1].squeeze(), \
                                                    result[2].squeeze(), \
                                                    int(result[3])

            label = "No Detections"
            x1, y1, x2, y2 = 0, 0, 0, 0
            if num_detections != 0:
                y1, x1, y2, x2 = boxes[0][0] * h, boxes[0][1] * w, boxes[0][2] * h, boxes[0][3] * w
                x1, y1, x2, y2 = int(max(0, np.floor(x1 + 0.5))), \
                                 int(max(0, np.floor(y1 + 0.5))), \
                                 int(min(w, np.floor(x2 + 0.5))), \
                                 int(min(h, np.floor(y2 + 0.5)))

                label = self.labels[str(int(labels[0]))].title()
            return label, (x1, y1, x2, y2)
    
        elif re.match(r"^segment$", self.infer_type, re.IGNORECASE):
            detected_labels = []
            h, w, _ = image.shape
            
            image = image / 255
            image = cv2.resize(src=image, dsize=(self.size, self.size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
            for i in range(image.shape[0]):
                image[i, :, :] = (image[i, :, :] - MEAN[i]) / STD[i]
            image = np.expand_dims(image, axis=0)

            input = {self.ort_session.get_inputs()[0].name : image.astype("float32")}
            result = self.ort_session.run(None, input)
            class_index_image = np.argmax(result[0].squeeze(), axis=0)
            disp_image = cv2.resize(src=segmenter_decode(class_index_image), dsize=(w, h), interpolation=cv2.INTER_AREA)
            
            class_indexes = np.unique(class_index_image)
            for index in class_indexes:
                if index != 0:
                    detected_labels.append(self.labels[str(index)].title())
            return disp_image, detected_labels

#####################################################################################################

def segmenter_decode(class_index_image: np.ndarray) -> np.ndarray:
    colors = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                       (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                       (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                       (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r, g, b = np.zeros(class_index_image.shape, dtype=np.uint8), \
              np.zeros(class_index_image.shape, dtype=np.uint8), \
              np.zeros(class_index_image.shape, dtype=np.uint8)

    for i in range(21):
        indexes = (class_index_image == i)
        r[indexes] = colors[i][0]
        g[indexes] = colors[i][1]
        b[indexes] = colors[i][2]
    return np.stack([r, g, b], axis=2)

#####################################################################################################

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def decode_data(data: str) -> np.ndarray:
    encoded_image = np.array(bytearray(data), dtype="uint8")
    return cv2.cvtColor(src=cv2.imdecode(encoded_image, cv2.IMREAD_COLOR), code=cv2.COLOR_BGRA2RGB)


def decode_image(imageData) -> np.ndarray:
    header, imageData = imageData.split(",")[0], imageData.split(",")[1]
    image = np.array(Image.open(io.BytesIO(base64.b64decode(imageData))))
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGRA2RGB)
    return header, image


def encode_image_to_base64(header: str = "image/jpeg", image: np.ndarray = None) -> str:
    assert image is not None, "Image is None"
    _, imageData = cv2.imencode(".jpeg", image)
    imageData = base64.b64encode(imageData)
    imageData = str(imageData).replace("b'", "").replace("'", "")
    imageData = header + "," + imageData
    return imageData

#####################################################################################################