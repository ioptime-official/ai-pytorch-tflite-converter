import os
import torch
from collections import OrderedDict
import shutil

class PtOnnx:
    """         
    A class to convert PyTorch models to ONNX format.
    """
    device = torch.device('cpu')

    def __init__(self, flag, path):
        """
        Initializes the class.

        Args:
            flag (str): The type of model to convert.
            path (str): The path to the PyTorch model.
        """
        self.flag = flag
        self.path = path

        if self.flag == 's':
            self.convert_script()

        if self.flag == 'ul':
            self.convert_yolo()

        if self.flag == 'y6':
            self.convert_yolov6()

        if self.flag == 'y7':
            self.convert_yolov7()

    def convert_script(self):
        """
        Converts a simple PyTorch model to ONNX format.
        """
        model = torch.jit.load(self.path)
        model.to(torch.device('cpu'))

        dummy_input = torch.randn(1, 3, 224, 224)
        dummy_input = dummy_input.to(self.device)

        torch.onnx.export(
            model,
            dummy_input,
            'model.onnx',
            export_params=True
        )

    def convert_yolo(self):
        """
        Converts a ultralytics yolo v5 and v8 PyTorch model to ONNX format.
        """
        # pip install ultralytics
        from ultralytics import YOLO

        model = YOLO(self.path)
        model.export(format='onnx')

    def convert_yolov6(self):
        """

        # pip install torch>=1.8.0
        # pip install onnx>=1.10.0 
        Converts a YOLOv6 PyTorch model to ONNX format.
        """
        
        if not os.path.exists("yolov6"):
            if not os.path.exists("YOLOv6"):
                print("Cloning YOLOv6 repository...")
                os.system("git clone https://github.com/meituan/YOLOv6")
            shutil.move("YOLOv6/yolov6", "./")
        else:
            print("Using existing YOLOv6 directory...")




        if isinstance(torch.load(self.path), (dict, OrderedDict)):
          print(f'model is of dict /collect ::: {type(torch.load(self.path))} type  ')
          ckpt= torch.load(self.path)
          x=torch.zeros(1,3,320,192)
          model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
          torch.onnx.export(model, x, 'model.onnx')
          print('model is converted')

        else: 

          print(f'model is of : {type(torch.load(self.path))} type')
          model= torch.load(self.path)
          x=torch.zeros(1,3,320,192)
          torch.onnx.export(model, x, 'model.onnx')
          print('ent mod is converted')


    def convert_yolov7(self):
        """
        convert yolov7 model to onnx
        """
        # import collection
        # if isinstance(torch.load(self.path), (dict, OrderedDict)):

        model= torch.hub.load('WongKinYiu/yolov7', 
            'custom',
             self.path,
             force_reload= True)
        # else:
        #   model= torch.load(self.path)
        img= torch.rand(1, 3, 640, 640)
        torch.onnx.export(model, img, 'yolov7.onnx', verbose=False, opset_version=12)

      
      # to load open mmmodel 
      # 1. model save state dict than to load if model is cls thean init_model apis will use else init_detector
      # 2. when model is save with torch.save to loas
    

    def convert_mm(self):
        """ 
        convet open mm model to onnx format

        * find the model is of cls of det to automate process\
        * we can find fom the last layer that it is det or cls model
        * cheak model is insatance of which mmcls or mmdet


		
        if model is of classifiction it will be load using init_model
        detection and segmentation model is load using inti_detector
        single shot detector model is converetted directly (yolo, ssd, convoldet)
        multiple shot cannot convert directly(mask, faster(r-cnn))
        it can be converted usign mmdeploy 
        


        """
        if isinstance(torch.load(self.path), (dict, OrderedDict)):
            try:
                model = init_model(self.config, self.path, device='cpu')
            
            except:
                model = init_detector(self.config, self.path, device='cpu')
        
        else:
            model= torch.load(self.path)
            

        model.forward= model.forward_dummy

        x= torch.randn(1, 3 ,224, 224)
        
        torch.onnx.export(
            model, x,'model.onnx', input_names=['input'],
            export_params=True,keep_initializers_as_inputs=True,
            do_constant_folding=True,verbose=False,opset_version=13)




        # import mmcv
        # import mmcls
        # import mmdet
        # import torch
        # # from mmcls.apis import init_model
        # # from mmdet.apis import init_detector
        # print('          done                         ')
        # a= str(input('if model is oreder dict press s, else p'))
        # # we need two file oredect dict and config file
        # if a== 's':
        #   b= str(input('for cls mod print c for dect press d'))
        #   if b=='c':
        #     model= init_model(self.config, self.path, device= 'cpu')
        #   if b== 'd':
        #     model= init_detector(self.config, self.path, devie= 'cpu')
        # if a== 'p':
        #   model = torch.load(self.path)


        # model.forward = model.forward_dummy
        # model= init_model(self.config, self.path, device= 'cpu')
        # x= torch.randn(1, 3 ,224, 224)
        # torch.onnx.export(model,
        # x,'model.onnx',
        # input_names=['input'],
        # export_params=True,
        # keep_initializers_as_inputs=True,
        # do_constant_folding=True,
        # verbose=False,
        # opset_version=13)
