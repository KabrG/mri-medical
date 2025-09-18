# Early Alzheimer's Detection

This project uses MRI scans from the OASIS database to train a model for assessing Alzheimer’s risk. The model reached an accuracy of 81.0%. Since MRI scans are typically used to support clinical evaluation rather than provide a standalone diagnosis, this level of accuracy is considered reasonable. Over 85,000 samples were used in training.

Dataset: https://www.kaggle.com/datasets/ninadaithal/imagesoasis

Run the command `pip freeze > requirements.txt` to generate the requirements text file.

Run the command `pip install -r requirements.txt` to install the necessary packages


For CUDA, make sure to use .

`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`

or

`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

| **Non Dementia Patient** | **Dementia Patient**|
|----------------------|------------------|
| <img width="496" height="248" alt="Non Dementia" src="https://github.com/user-attachments/assets/7b3af35e-087a-445c-a63b-f6bf546693a0" /> | <img width="496" height="248" alt="Dementia" src="https://github.com/user-attachments/assets/b146a92d-d0db-4be7-9c6d-39f0e317b0c7" /> |



**Non Dementia Patient (Preprocessed)**

<img width="424" height="424" alt="image" src="https://github.com/user-attachments/assets/788056f9-692a-43f1-b860-1372d6256fd5" />
