# OnnxIsolation
[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)



## Idea

- When exporting the Onnx model using PyTorch, we found that the intermediate nodes in the onnx file did not have attributes such as dType/Shape.
- As you can see in the visual structure below, the red box on the right should normally have a small plus sign, but it does not (version=1.10.0)
- ![image-20220701200737107](https://tva1.sinaimg.cn/large/e6c9d24egy1h3rooh6ky8j217i0lmjsu.jpg)

- In this case, when you want to extract one operator or part of the onnx for analysis, the exported graph cannot be run by onnxruntime (mainly without input, and can also not run if the dimension number of output is not specified), nor can it be converted into the TRT engine of Tensorrt.
- The idea for 1.0.0 is pretty naive: find out shape and then do ISOLATE. The way to find out shape was to set the target node as onnxruntime's outputs 

## Installation

- install from pip

```sh
pip install nvidia-pyindex
pip install onnxisolation
```

- install from source 

```sh
cd onnxisolation
python setup.py install
```

## Example

```python
import onnxisolation.export as export 

model_path = "path_to_source_path.onnx"
save_path = "path_to_save_path.onnx"
tmp_path = "/home/qcraft/code/bev/tmp/tmp.onnx"

node_input = ["img", "lidar2img", "img2lidar"]
node_output = ["all_cls_scores", "all_bbox_preds"]
# node_input = ["all_cls_scores"]
# node_output = ["all_bbox_preds"]

# 后续可能会使用它来更新input_shape
input_shapes = [
    [1, 5, 3, 512, 1024],
    [1, 5, 4, 4],
    [1, 5, 4, 4]
]

export.export_onnx(model_path=model_path, save_path=save_path, tmp_path=tmp_path, node_input=node_input, node_output=node_output, input_shapes=input_shapes)
    
```





## Dev Env

- Ubuntu 20
- python 3.9
- Pytorch 1.10



## License

MIT

  
