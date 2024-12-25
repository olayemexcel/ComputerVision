AI-Driven Diagnosis of Pneumonia Using Chest X-Ray Imaging
Comparative Analysis of EfficientNetB0 and ResNet50 Architectures with Class Balancing Techniques
By: Musiliu Bello

Research Overview
This project explores the application of advanced deep learning models, ResNet50 and EfficientNetB0, for pneumonia classification using chest X-ray images. The research addresses critical challenges such as class imbalance, overfitting, and generalization issues using techniques like SMOTE (Synthetic Minority Over-sampling Technique), class weighting, and fine-tuning of pre-trained models.

Key contributions of this research include:

Comparative performance analysis of ResNet50 and EfficientNetB0 on pneumonia detection.
Application of dataset balancing techniques to mitigate class skewness.
Deployment-ready web implementation using Streamlit for real-time prediction and visualization.
The dataset used includes labeled chest X-ray images categorized as "Normal" or "Pneumonia," offering a robust testbed for evaluating the models.

Key Features
Model Architectures: ResNet50 and EfficientNetB0 pre-trained on ImageNet.
Techniques Used: SMOTE, class weighting, image augmentation, and fine-tuning.
Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
Tools and Frameworks: TensorFlow, Keras, PyTorch, Scikit-learn, Matplotlib, and Streamlit.
Source Code:
Model training and evaluation: ComputerVision_Codes.ipynb (Python Jupyter Notebook).
Web implementation: app.py (Python script for Streamlit application).
Installation Instructions
To replicate the research results and run the web-based application, follow these steps:

Python Installation
Ensure Python 3.7 or later is installed on your system.

Install Required Libraries
Use the following commands to install the necessary Python libraries:

bash
Copy code
pip install tensorflow keras torch torchvision scikit-learn matplotlib imbalanced-learn streamlit pandas numpy opencv-python
Clone the Repository
Clone the GitHub repository containing all application files and models:https://github.com/olayemexcel/ComputerVision

bash
Copy code
git clone <https://github.com/olayemexcel/ComputerVision>
cd <repository_name>
Run the Web Application
Launch the Streamlit application with the command:

bash
Copy code
streamlit run app.py
Dataset
The dataset used for this research is the Chest X-Ray Images (Pneumonia) dataset, which contains labeled X-ray images categorized as "Normal" or "Pneumonia."

Dataset Source: Kaggle Chest X-Ray Pneumonia Dataset.
Ensure the dataset is downloaded and stored in the data/ directory for preprocessing and training.

File Structure
bash
Copy code
/research_code/
│
├── /data/
│   ├── train/
│   ├── test/
│   └── val/
├── /models/
│   ├── resnet50_model.py
│   └── efficientnetb0_model.py
├── /scripts/
│   ├── data_preprocessing.py
│   ├── ComputerVision_Codes.ipynb
│   └── app.py
├── README.txt
└── requirements.txt
How to Run the Code
1. Model Training and Evaluation
The entire process for data preprocessing, model training, and evaluation is implemented in ComputerVision_Codes.ipynb.

Open the Jupyter Notebook:
bash
Copy code
jupyter notebook scripts/ComputerVision_Codes.ipynb
Follow the step-by-step instructions within the notebook to:
Preprocess the dataset.
Train models (ResNet50 or EfficientNetB0).
Evaluate models using metrics such as Accuracy, Precision, Recall, and ROC-AUC.
2. Run Web Application
For real-time pneumonia predictions:

Open the Streamlit application by running:
bash
Copy code
streamlit run app.py
Follow the instructions in the web interface to upload and analyze chest X-ray images.
Results and Known Issues
EfficientNetB0: Successfully displays predictions when users upload optimized chest X-ray images resized to 150x150 dimensions.
ResNet50: Currently encounters an input dimension mismatch error during prediction. Due to time constraints, this 
limitation will be resolved in future updates.

Future Improvements
Explore attention mechanisms and ensemble techniques for enhanced performance.
Use larger and more diverse datasets for improved generalizability.
Enhance the user interface for the web application.

License and Acknowledgments
This research is part of the AI-Driven Diagnosis of Pneumonia project.
The dataset is sourced from Kaggle, and the models are built on publicly available architectures (ResNet50 and EfficientNetB0).

All project files and implementations are available on GitHub for transparency and collaboration.

Contact Information
For inquiries or collaboration opportunities, contact:

Author: Musiliu Bello
Email: olayemexcel1@gmail.com
