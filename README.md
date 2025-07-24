# Detecting Blindness with Deep Learning

![sample](https://i.postimg.cc/dVjwCDr2/blindness.png)

## 📌 Project Summary

Diabetic retinopathy (DR) is one of the leading causes of vision loss. Early detection and treatment are crucial steps toward preventing DR.  
This project considers DR detection as an **ordinal classification task** and aims to develop a deep learning model that predicts the severity of DR based on a patient's retina photograph.

We utilize data from the [APTOS 2019 Blindness Detection competition](https://www.kaggle.com/c/aptos2019-blindness-detection/data) hosted on Kaggle.

## 🗂️ Project Structure

```bash
├── codes/
│   ├── data.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── utilities.py
│   └── efficientnet-pytorch/           # EfficientNet model wrapper
│       └── EfficientNet.py
├── efficientnet_pytorch/              # EfficientNet weights (not included)
├── examples/
├── figures/                           # Visualizations exported from notebooks
├── input/
│   ├── aptos2019-blindness-detection/ # Main dataset (2019)
│   └── diabetic-retinopathy-resized/  # Supplementary dataset (2015)
├── models/                            # Saved model weights
├── notebooks/
│   ├── code-01-data-exploration.ipynb
│   ├── code-02-pre-training.ipynb
│   ├── code-03-training.ipynb
│   └── code-04-inference.ipynb
├── submissions/                       # Inference outputs
├── LICENSE
├── README.md
├── proposal.pdf
└── report.pdf
```
🧪 Notebooks Overview
code-01-data-exploration.ipynb: Data visualization and exploration

code-02-pre-training.ipynb: Pre-train CNN on the 2015 supplementary dataset

code-03-training.ipynb: Fine-tune the CNN on the 2019 main dataset

code-04-inference.ipynb: Predict test images using the trained model

🛠️ Requirements
To run this project, set up a virtual environment using conda:
```
conda create -n aptos python=3.10
conda activate aptos
Install the required Python packages:
```
```
pip install torch torchvision
pip install efficientnet-pytorch
pip install opencv-python
pip install scikit-learn pandas numpy matplotlib seaborn scipy tqdm
```
⬇️ Data & Model Weights
⚠️ Note: Due to size constraints, the datasets and pre-trained weights are not included in this repository.

📄 References
APTOS 2019 Blindness Detection Competition


