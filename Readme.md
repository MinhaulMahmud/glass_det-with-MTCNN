# Glasses Detection using Face Recognition

This project focuses on detecting human faces in images and classifying whether the person is wearing glasses. It uses the MTCNN model for face detection and OpenCV for image processing. The project implements a full pipeline from data loading and face extraction to training a machine learning model for classification.

## Features

- **Face Detection**: Leverages the MTCNN (Multi-task Cascaded Convolutional Networks) to detect faces in images.
- **Glasses Classification**: Classifies faces into two categories: with glasses and without glasses.
- **Data Preprocessing**: Loads images, detects faces, resizes them, and creates labels.
- **Machine Learning Model**: Trains a model using Scikit-learn to classify whether glasses are present in the detected faces.

## Technologies Used

- **Python**: Core programming language for the project.
- **OpenCV**: Used for image loading, processing, and manipulation.
- **MTCNN**: Face detection model for accurate face localization.
- **Scikit-learn**: For splitting datasets and performing machine learning tasks.
- **NumPy**: Data handling and manipulation.

## Project Setup

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required Python packages (specified below)

### Installation

1. Clone the repository:
   ```bash
    git clone https://github.com/MinhaulMahmud/glass_det-with-MTCNN.git
   ```
2. Navigate to the project directory:
   ```bash
   cd glass-det-with-MTCNN
   ```

### Dataset Preparation

- Prepare two folders: one containing images of people wearing glasses (`with_glasses`) and another for people without glasses (`without_glasses`).
- Place the images in their respective folders before running the Jupyter notebook.

### Running the Project

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Execute the cells in the notebook to:
   - Load images.
   - Detect faces using MTCNN.
   - Resize the faces for classification.
   - Train the machine learning model on the dataset.
   - Test the model on new data.

## Folder Structure

```
glasses-detection/
│
├── datasets/                # Folder containing images (with_glasses/ without_glasses)
├── notebooks/               # Jupyter notebooks for model training and testing
├── src/                     # Core Python scripts for detection and classification
└── README.md                # Project documentation
```

## Usage

- The model can be trained and tested within the provided Jupyter notebook.
- Customize the dataset by adding or replacing images in the `datasets` folder to improve classification performance.

## Future Improvements

- **Model Optimization**: Fine-tune the model for faster and more accurate results.
- **Augmentation**: Apply data augmentation techniques to increase dataset size.
- **Real-Time Detection**: Implement real-time face detection and glasses classification using a webcam or video stream.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For any queries or issues, feel free to open an issue in the repository or contact the project owner.

```

This README provides a comprehensive overview and makes the project look professional. Let me know if you'd like to make any tweaks!
