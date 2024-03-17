# Breast Cancer Detection with Gradio

This project aims to predict breast cancer in patients using machine learning techniques. The model is trained on a dataset containing various features extracted from breast mass images.

## Dataset

The dataset used in this project contains information about cell nuclei taken from breast mass images, including features such as radius, texture, perimeter, area, smoothness, and more. The target variable, 'diagnosis', indicates whether a patient has benign or malignant breast cancer.

## Installation

1. Clone this repository:
git clone https://github.com/yourusername/breast-cancer-detection.git 

2. Install the required dependencies:
pip install -r requirements.txt


3. Download the dataset and place it in the project directory:
C:/Nirmala/GUVI/breast cancer/breast_cancer_dataset.csv


## Usage

1. Run the `breast_cancer_detection.py` script:
python breast_cancer_detection.py


2. Access the Gradio interface in your web browser:
http://localhost:7860


3. Input the values for various features of the breast mass, and the model will predict whether the tumor is benign or malignant.

## Model Details

- Model: Random Forest Classifier
- Features: Mean, standard error, and worst values of various cell features
- Preprocessing: Standardization of features using StandardScaler

## Contributing

Contributions to this project are welcome. Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please contact [Your Name](mailto:your.email@example.com).
