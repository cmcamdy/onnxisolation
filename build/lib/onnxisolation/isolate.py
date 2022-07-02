

import onnx
import onnx_graphsurgeon as gs
# from reshape_onnx import *
from onnxisolation.reshape import *
# import pdb

def isolate(node_input, input_shape, node_output, output_shape, model_path, save_path):
    '''
        - read onnx model, get graph,tensors
        - check tensors[node].shape tensors[node].type
            - if None,
    
    '''
    # read onnx 
    model = onnx.load(model_path)

    model = infer_onnx_shapes(model)
    graph = gs.import_onnx(model)
    
    tensors = graph.tensors()
    # print(tensors.keys())

    # pdb.set_trace()

    # set io
    if node_input is not None:
        graph.inputs = [tensors[node_input[index]].to_variable(shape=input_shape[index], dtype=tensors[node_input[index]].dtype) for index in range(len(node_input))]
    else:
        node_input = [input.name for input in graph.inputs]

    if node_output is not None:
        # pdb.set_trace()
        graph.outputs = [tensors[node_output[index]].to_variable(shape=output_shape[index], dtype=tensors[node_output[index]].dtype) for index in range(len(node_output))]
    else:
        node_output = [output.name for output in graph.outputs]

    
    graph.cleanup()
    
    # check model
    try:
        onnx.checker.check_model(gs.export_onnx(graph))
        # save model
        onnx.save(gs.export_onnx(graph), save_path)
        print("Graph save succeed!")
        return node_input, node_output
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Model check error! Please check your input/output node set.")
    
    return None, None


