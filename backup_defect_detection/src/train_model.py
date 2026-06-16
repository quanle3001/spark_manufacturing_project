"""
Compatibility wrapper for the old Random Forest workflow.

In the TensorFlow Lite version, this file does NOT train a Random Forest model.
It creates the automatic dataset label/index files so your dataset is still organized
the same way as before.

Real TFLite training/export should be done separately, then put the model at:

    models/roi_tflite_defect_sorter.tflite

Run:
    python3 src/train_model.py
"""

from prepare_dataset_index import main


if __name__ == "__main__":
    main()
