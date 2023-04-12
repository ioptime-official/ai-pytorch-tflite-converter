import argparse
from pytorch_to_onnx import PtOnnx

parser = argparse.ArgumentParser(description='Export a PyTorch model to ONNX format')
parser.add_argument('flag', type=str, help='ul, mm, s')
parser.add_argument('path', type=str, help='path to the PyTorch model file')
args = parser.parse_args()

a = PtOnnx(args.flag, args.path)