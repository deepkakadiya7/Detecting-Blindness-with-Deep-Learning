# Detecting Blindness with Deep Learning

![sample](https://i.postimg.cc/dVjwCDr2/blindness.png)

## 📌 Project Summary

Diabetic retinopathy (DR) is one of the leading causes of vision loss. Early detection and treatment are crucial steps toward preventing DR.  
This project considers DR detection as an **ordinal classification task** and aims to develop a deep learning model that predicts the severity of DR based on a patient's retina photograph.

This project was completed under the [Udacity ML Engineer Nanodegree](https://confirm.udacity.com/LMMJDA7C).  
We utilize data from the [APTOS 2019 Blindness Detection competition](https://www.kaggle.com/c/aptos2019-blindness-detection/data) hosted on Kaggle.

📄 For a detailed walkthrough, check the [project blog post](https://kozodoi.me/python/deep%20learning/computer%20vision/competitions/2020/07/11/blindness-detection.html).  
You can also find a complete PDF description of the project in `report.pdf`.

---

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
