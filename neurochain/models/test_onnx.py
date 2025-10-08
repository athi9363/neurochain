import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("models/tiny_mnist.onnx")
inp_name = sess.get_inputs()[0].name
dummy = np.random.randn(1,1,28,28).astype(np.float32)
out = sess.run(None, {inp_name: dummy})
print("Output shape:", out[0].shape)
