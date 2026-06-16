import argparse

from config import TFLITE_MODEL_PATH, LABELS_PATH
from tflite_utils import load_tflite_interpreter, load_labels, describe_tflite_model


def main():
    parser = argparse.ArgumentParser(description="Validate TensorFlow Lite model and labels.")
    parser.add_argument("--model", default=str(TFLITE_MODEL_PATH))
    parser.add_argument("--labels", default=str(LABELS_PATH))
    args = parser.parse_args()

    labels = load_labels(args.labels)
    print(f"Labels ({len(labels)}): {labels}")

    interpreter = load_tflite_interpreter(args.model)
    interpreter.allocate_tensors()
    input_details, output_details = describe_tflite_model(interpreter)

    output_shape = output_details[0]["shape"]
    output_size = 1
    for value in output_shape:
        output_size *= int(value)

    print(f"\nOutput total values: {output_size}")
    print(f"Labels count:        {len(labels)}")

    if output_size == len(labels):
        print("\nOK: model output size matches labels count. Classification mode should work.")
    else:
        print("\nWarning: model output size does NOT match labels count.")
        print("This may mean your model is object-detection/YOLO style, not simple classification.")
        print("Only src/tflite_utils.py post-processing should need to change for that model type.")


if __name__ == "__main__":
    main()
