## Project Structure

```text
Character-Recognition-ML/
│
├── dataset/
│   └── (MNIST CSV files generated locally)
│
├── models/
│   └── (trained model files generated locally)
│
├── results/
│   ├── confusion_matrix.png
│   └── model_comparison.png
│
├── src/
│   ├── compare_models.py
│   ├── create_test_image.py
│   ├── download_dataset.py
│   ├── evaluate.py
│   ├── load_dataset.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
│
├── test_images/
│   ├── digit_5.png
│   └── mnist_5.png
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md