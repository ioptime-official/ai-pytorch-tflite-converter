# ai-pytorch-tflite-converter
## Introduction: 
This script will convert a Torch model to ONNX, ONNX to OpenVINO, and OpenVINO to TFLite.
## Workflow:
install dependencies
pip install requirement.txt

First, give the model. 
There are three ways to load and save the model:
1) If the model is saved in OrderedDict format, loading the model requires the model definition. After that, the model should be saved in torch.script format and given as input.
2) If the entire model is saved with torch.save, it can be loaded and saved in script format.
3) If the model is saved with torch.script method, the model can be given directly.

# run this commond
python3 onnx_to_tf.py model.pt -m model.onnx
this will create the file structre like this
## File structure 
```

├── mian
│   ├── model.pt
|   ├── requiremnet.txt
│   ├── pt_to_tflite.py
│   ├── model.onnx
|   ├── openvino 
│   │    ├── model.bin
│   │    ├── model.xml
│   │    ├── model.mapping
│   ├── opnevinotoensorflow
│        ├──model.tflite

```
