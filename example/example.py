import onnxisolation.export as export 




model_path = "/home/qcraft/code/bev/onnxyaml/model_1point_1layer_24epoch.onnx"
save_path = "/home/qcraft/code/bev/tmp/model_exported.onnx"
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
    