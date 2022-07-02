# from onnxsim import simplify
from onnx.shape_inference import infer_shapes


def infer_onnx_shapes(model):
    return infer_shapes(model)


# def simplity_onnx(model):

#     model_simp, check = simplify(model)
#     assert check, "Simplified ONNX model could not be validated"
    
#     return model_simp

