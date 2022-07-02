from onnxisolation.isolate import isolate
import onnx
import onnx_graphsurgeon as gs
from onnxisolation.reshape import infer_onnx_shapes
from onnxisolation.get_shape import get_node_shape, get_node_shapev2
import pdb

"""
1、设置参数
    input_shapes    : 原始模型的初始输入
    node_input      : 导出模型的输入名称
    node_output     : 导出模型的输出名称
    model_path      :
    save_path       :
    tmp_path        : get shape 的中间保存path，这个可以固定，反正后期不用
2、推导需要shape的名称，导出一个list，传入 get shape
    判断node_input、node_output是否为None，若为None，则赋值为graph.inputs/outputs,同时也不用参与下面的判断
    判断tensor名称是否在graph的输入输出中
        如果是输入则直接赋值
        如果是value则加入list，传入get shape
        如果是输出，则不管，到时候get shape的返回值中会有shape
3、输入输出的shape赋值，然后进入isolate，然后导出
"""

def load_model(model_path):
    return onnx.load(model_path) 

def save_model(model, save_path):
    onnx.save(model, save_path)

def get_names(tensors):
    return [tensor.name for tensor in tensors]

def infer_node_name(model, node_list : list):
    # to find the node not in graph.inputs/outputs
    # 为什么不用onnx下的model.graph.value_info？ 因为有的onnx有可能没有value_info为空，当然可以通过shape infer解决
    graph = gs.import_onnx(model)
    
    tensors = graph.tensors()

    input_names = get_names(graph.inputs)
    output_names = get_names(graph.outputs)
    value_names = [key for key in tensors.keys() if key not in input_names and key not in output_names]
    
    return [name for name in node_list if  name in value_names], [name for name in node_list if name in input_names], [name for name in node_list if name in output_names]

def infer_node_namev2(model, node_list : list):
    # to find the node not in graph.inputs/outputs
    # 为什么不用onnx下的model.graph.value_info？ 因为有的onnx有可能没有value_info为空，当然可以通过shape infer解决
    model = infer_onnx_shapes(model)

    input_names = get_names(model.graph.input)
    output_names = get_names(model.graph.output)
    value_names = get_names(model.graph.value_info)
    
    return [name for name in node_list if  name in value_names], [name for name in node_list if name in input_names], [name for name in node_list if name in output_names]

def get_input_shape(model, node_list : list):
    # 从node_list中得到shape的kv对

    # input
    graph = gs.import_onnx(model)
    inputs = graph.inputs

    # pdb.set_trace()
    shape_map = {}

    for in_tensor in inputs:
        if in_tensor.name in node_list:
            shape_map[in_tensor.name] = in_tensor.shape

    # pdb.set_trace()
    assert len(shape_map.keys()) == len(node_list), "node_list != inputs, please check"

    return shape_map

def get_output_shape(model, node_list : list):
    # 从node_list中得到shape的kv对

    # input
    graph = gs.import_onnx(model)
    outputs = graph.outputs

    # pdb.set_trace()
    shape_map = {}

    for out_tensor in outputs:
        if out_tensor.name in node_list:
            shape_map[out_tensor.name] = out_tensor.shape

    # pdb.set_trace()
    assert len(shape_map.keys()) == len(node_list), "node_list != outputs, please check"

    return shape_map

def get_node_shape(node_list, node_map):
    node_shape = []
    for node in node_list:
        node_shape.append(node_map[node])
    return node_shape


def export_onnx(model_path, save_path, tmp_path="./tmp.onnx", node_input = None, node_output = None, input_shapes = []):
    
    # pdb.set_trace()
    ndoe_need_shape = node_input + node_output

    model = load_model(model_path=model_path)
    
    node_in_value, node_in_input, node_in_output = infer_node_namev2(model, ndoe_need_shape)
    
    # get input shape 
    shape_map = {}
    if len(node_in_input) > 0: 
        print("Get input shape.")
        shape_map.update(get_input_shape(model, node_list=node_in_input))
    if len(node_in_output) > 0:
        print("Get output shape.")
        shape_map.update(get_output_shape(model, node_list=node_in_output))
    if len(node_in_value) > 0:
        print("Get node shape.")
        shape_map.update(get_node_shapev2(model_path, tmp_path, input_shapes=input_shapes, output_names=node_in_value))
    print("shape_map = ", shape_map) 

    node_input_shape = get_node_shape(node_input, shape_map)
    node_output_shape = get_node_shape(node_output, shape_map)
    
    isolate(node_input, node_input_shape, node_output, node_output_shape, model_path, save_path)



