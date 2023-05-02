### PyTorch To ONNX Converter:


This tool can convert any PyTorch computer vision model to the ONNX format. The script supports models in different formats, including torch.nn, Ordereddict, and torch.jit. If the model is in Ordereddict format, a config file may be required to load the model architecture. The script can automatically detect the model format and convert it.


#### Usage: 
To use this steps follow these steps:
                
1. Clone the repository to local machine 
2. Install the required packages by running pip  `pip install -r requirements.txt`
3. Prepare the model and select the flag according to model library
4. Run the command ```python3 input.py [flag] [model.path]``` to start the conversion(it will download some specific modules if require)


| Model  | Flag |
| ------------- | ------------- |
| Simple script model(classification) | 's’  |
| ultralytics(yolov5, 8)| ‘ul’  |
| yolov6| 'y6’  |
| yolov7| ‘y7’  |
| openmmlab| ‘mm’  |
| mm detection | ‘md’  |
| mm classification| ‘mc’  |
| detectron2  |'det'|

### Output:
The script will create an ONNX file with the same name as model_name with .onnx extension. The ONNX file will be saved in the same directory as the PyTorch model file.

### Files:
This tool consist of two Python files:
1. Pytorch_to_onnx.py : the main script that perform PyTorch to ONNX conversion.
2. Input.py : User friendly script that takes model path and flag input and pass them to pytorch_to_onnx.py to perform conversion.


### Model Conversion tabel
|library| model | onnx| tflite
| ------------- | ------------- |------------- | ------------- |
|PyTorch|Custom Vgg|✅|✅|
|PyTorch| Vgg  |✅|✅|
|PyTorch|mobile net|✅|✅|
| PyTorch| Vgg  |✅|✅|
| PyTorch |resnet|✅|✅|
| PyTorch| inception net|✅|✅|
| PyTorch/ mmcls|efficient net|✅|✅|
| Timms |efficent_ns|❔|❔|
|ultralytics| YOLOv5  |✅|✅|
|ultralytics| YOLOv8  |✅|✅|
|meituan| YOLOv6 |✅|✅|
|WongKinYiu| YOLOv7|✅|✅|
|OpenMMLab| classification|✅|✅|
|OpenMMLab| detection|✅|✅|
|OpenMMLab| segmentation  |✅|✅|
|Detectron| detection|❔|❔|
|Detectron|segmentation|❔|❔|
