import onnxruntime as ort

session = ort.InferenceSession(
    r"C:\Users\amina\autonomous_lane_detect\models\tusimple_18.onnx",
    providers=["CPUExecutionProvider"]
)

print("INPUTS")
for x in session.get_inputs():
    print(x.name, x.shape)

print()

print("OUTPUTS")
for x in session.get_outputs():
    print(x.name, x.shape)