

# OnnxIsolation

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

- 我们使用pytorch导出onnx模型的过程中发现onnx文件中的中间节点并没有诸如dtype/shape等属性。
- 如下图可视化结构所示，右侧红框在正常情况下应当有小加号，实际上用torch.onnx.export导出就没有（版本为1.10.0）

![image-20220701200737107](https://tva1.sinaimg.cn/large/e6c9d24egy1h3rooh6ky8j217i0lmjsu.jpg)

- 这种情况下想要摘出某个算子或者某个部分做分析的时候，导出的graph就无法使用onnxruntime运行（主要是没有input，同时如果不指定output的维度数，也无法运行），同时也无法转换成tensorrt的trt engine。



## Installation

- pip安装
```
pip install nvidia-pyindex
pip install onnxisolation
```

- 源码安装 	

```
cd onnxisolation
python setup.py install
```




## Example

- 1.0.0的想法很原始：找出shape然后再做isolate，找出shape的方式是设置为onnxruntime的输出节点

- ```python
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

  

