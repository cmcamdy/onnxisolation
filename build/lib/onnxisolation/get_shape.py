import onnx
import onnxruntime as ort
import numpy as np
from onnxisolation.reshape import infer_onnx_shapes
# import cv2
import pdb


def set_model_output(model_path, tmp_path, output_names):
    model = onnx.load(model_path)
    model = infer_onnx_shapes(model)
    graph = model.graph
    out_index = len(graph.output)
    output_map = {}

    # 将需要推导的节点添加到输出，这样就可以得到这个节点的shape了
    for index, tensor in enumerate(graph.value_info):
        if(tensor.name in output_names):
            # pdb.set_trace()
            graph.output.insert(out_index, tensor)
            output_map[tensor.name] = out_index
            out_index += 1
        # print(tensor.name)

    onnx.save(model, tmp_path)
    # pdb.set_trace()
    return output_map


def build_onnx_model(model_path):
    # https://onnxruntime.ai/docs/api/python/api_summary.html#api-overview
    # ONNX运行时通过执行提供程序编排操作符内核的执行。先 openvino 后 cpu
    return ort.InferenceSession(model_path, providers=['OpenVINOExecutionProvider', 'CPUExecutionProvider'])

def generate_input_data(input_shapes):
    
    in_tensors = []
    # pdb.set_trace()
    for index in range(len(input_shapes)):
        in_tensor = (np.random.rand(*input_shapes[index]).astype(np.float32)) * 2 - 1
        in_tensors.append(in_tensor)

    return in_tensors

def set_onnx_IO(in_tensors, onnx_session):

     # onnx set
    inputs = onnx_session.get_inputs()

    # input set 
    onnx_inputs = {}
    for index in range(len(inputs)):
        input = inputs[index]
        in_tensor = in_tensors[index]
        onnx_inputs[input.name] = in_tensor

    # output set 
    onnx_output_names = []
    for out in onnx_session.get_outputs():
        onnx_output_names.append(out.name)

    return onnx_inputs, onnx_output_names


def run_onnx_session(onnx_session, onnx_output_names, onnx_inputs):
    return onnx_session.run(onnx_output_names, onnx_inputs)



def get_output_shapes(onnx_outputs):

    output_shapes = []
    for index in range(len(onnx_outputs)):
        output_shapes.append(onnx_outputs[index].shape)
    return output_shapes

def get_output_map(onnx_outputs, output_names):

    output_map = {}
    for index in range(len(output_names)):
        output_map[output_names[index]] = onnx_outputs[index].shape
    return output_map


def get_node_shape(model_path, tmp_path, input_shapes, output_names):
    
    output_map = set_model_output(model_path, tmp_path, output_names)

    # Currently ONNX operator set version: 16 is unsupported. Falling back to: 15
    onnx_session = build_onnx_model(tmp_path)

    # run onnx  
    in_tensors = generate_input_data(input_shapes)
    onnx_inputs, onnx_output_names = set_onnx_IO(in_tensors, onnx_session)
    onnx_outputs = run_onnx_session(onnx_session, onnx_output_names, onnx_inputs)
    
    # get shapes
    output_shapes = get_output_shapes(onnx_outputs)

    return output_shapes, output_map


def get_node_shapev2(model_path, tmp_path, input_shapes, output_names):
    
    output_map = set_model_output(model_path, tmp_path, output_names)

    # Currently ONNX operator set version: 16 is unsupported. Falling back to: 15
    onnx_session = build_onnx_model(tmp_path)

    # run onnx  
    in_tensors = generate_input_data(input_shapes)
    onnx_inputs, onnx_output_names = set_onnx_IO(in_tensors, onnx_session)
    onnx_outputs = run_onnx_session(onnx_session, onnx_output_names, onnx_inputs)
    
    # pdb.set_trace()

    # get shapes
    output_shapes = get_output_shapes(onnx_outputs)

    # reset map
    for key in output_map.keys():
        output_map[key] = output_shapes[output_map[key]]

    # update map, add model output
    output_map.update(get_output_map(onnx_outputs, onnx_output_names))
    # pdb.set_trace()

    return output_map
