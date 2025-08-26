import numpy as np
import tritonclient.http as httpclient

# Triton client
client = httpclient.InferenceServerClient("localhost:8000")

# Example patient record
input_data = np.array([[55, 130, 210, 7]], dtype=np.float32)

inputs = []
inputs.append(httpclient.InferInput("float_input", input_data.shape, "FP32"))
inputs[0].set_data_from_numpy(input_data)

outputs = []
outputs.append(httpclient.InferRequestedOutput("output_label"))

result = client.infer(model_name="patient_risk", inputs=inputs, outputs=outputs)

print("Predicted risk (probabilities):", result.as_numpy("output_label"))
