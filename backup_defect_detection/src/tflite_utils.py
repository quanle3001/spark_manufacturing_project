import numpy as np
import cv2


def load_tflite_interpreter(model_path):
    """
    Try tflite_runtime first because it is common on Raspberry Pi.
    Fall back to TensorFlow if installed.
    """
    try:
        from tflite_runtime.interpreter import Interpreter
        print("[TFLite] Using tflite_runtime.")
        return Interpreter(model_path=str(model_path))
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
            print("[TFLite] Using tensorflow.lite Interpreter.")
            return Interpreter(model_path=str(model_path))
        except ImportError as exc:
            raise RuntimeError(
                "No TensorFlow Lite interpreter found. Install one of these:\n"
                "  pip3 install tflite-runtime\n"
                "or install TensorFlow if supported on your machine."
            ) from exc


def load_labels(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]

    if not labels:
        raise RuntimeError(f"Labels file is empty: {labels_path}")

    return labels


def describe_tflite_model(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n[TFLite] Model details")
    print(f"  Input shape:  {input_details[0]['shape']}")
    print(f"  Input dtype:  {input_details[0]['dtype']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    print(f"  Output dtype: {output_details[0]['dtype']}")

    return input_details, output_details


def prepare_input_image(roi_crop, input_details):
    """
    Prepare ROI image for a common image-classification TFLite model.

    Expected input shape:
        [1, height, width, channels]

    Supports:
        float32: normalized to 0..1
        uint8/int8: raw integer input, with int8 shifted if needed by quantization
    """
    input_info = input_details[0]
    input_shape = input_info["shape"]
    input_dtype = input_info["dtype"]

    if len(input_shape) != 4:
        raise RuntimeError(
            f"Expected a 4D image input like [1, height, width, channels], got {input_shape}"
        )

    height = int(input_shape[1])
    width = int(input_shape[2])
    channels = int(input_shape[3])

    resized = cv2.resize(roi_crop, (width, height))

    if channels == 3:
        # OpenCV camera is BGR, most TensorFlow image models expect RGB.
        image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    elif channels == 1:
        image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=-1)
    else:
        raise RuntimeError(f"Unsupported input channel count: {channels}")

    if input_dtype == np.float32:
        input_data = image.astype(np.float32) / 255.0
    elif input_dtype == np.uint8:
        input_data = image.astype(np.uint8)
    elif input_dtype == np.int8:
        input_data = image.astype(np.float32)

        quantization = input_info.get("quantization", (0.0, 0))
        scale, zero_point = quantization

        if scale and scale > 0:
            input_data = input_data / scale + zero_point

        input_data = np.clip(input_data, -128, 127).astype(np.int8)
    else:
        input_data = image.astype(input_dtype)

    return np.expand_dims(input_data, axis=0)


def dequantize_output(output_data, output_info):
    output_dtype = output_info.get("dtype")

    if output_dtype in (np.uint8, np.int8):
        scale, zero_point = output_info.get("quantization", (0.0, 0))

        if scale and scale > 0:
            return scale * (output_data.astype(np.float32) - zero_point)

    return output_data.astype(np.float32)


def predict_classification(interpreter, input_details, output_details, roi_crop, labels):
    """
    Run TFLite classification on the fixed ROI crop.

    This expects model output to be class scores/probabilities where the number
    of output values matches the labels file.
    """
    input_data = prepare_input_image(roi_crop, input_details)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output_raw = interpreter.get_tensor(output_details[0]["index"])
    output = np.squeeze(output_raw)
    output = dequantize_output(output, output_details[0]).flatten()

    if output.size != len(labels):
        raise RuntimeError(
            "This TFLite model output does not look like a simple classifier.\n"
            f"Output size: {output.size}\n"
            f"Number of labels: {len(labels)}\n"
            "If your .tflite model is YOLO/object-detection style, its post-processing "
            "must be different from this classification version."
        )

    best_index = int(np.argmax(output))
    confidence = float(output[best_index])
    label = labels[best_index]

    return label, confidence, output
