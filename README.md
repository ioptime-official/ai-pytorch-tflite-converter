# ai-pytorch-tflite-converter
## Introduction: 
This script will convert a Torch model to ONNX, ONNX to OpenVINO, and OpenVINO to TFLite.
## Workflow:
First, give the model. There are three ways to load and save the model:
1) If the model is saved in OrderedDict format, loading the model requires the model definition. After that, the model should be saved in torch.script format and given as input.
2) If the entire model is saved with torch.save, it can be loaded and saved in script format.
3) If the model is saved with torch.script method, the model can be given directly.



## setps
## Getting the Model:
To load the model saved with the torch.save method, we need to load the model.pt file along with the model definition (model instantiation). 
To load the model saved with torchscript does not require model instantiation, only require .pt file.

## Exporting PyTorch to ONNX format:
Using torch.onnx.export, we can export the model to the ONNX format while also defining the input shape of the model by passing a dummy array with the same shape as the input.

## Checking the ONNX Model and Printing the Graph:
To ensure that the model is exported correctly, we can check it and print the graph.

## Converting the ONNX Model to TFLite:
Since the PyTorch and ONNX formats differ from TensorFlow, we need to use the OpenVINO API. First, the ONNX model is converted to the OpenVINO format and then to TFLite.

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
