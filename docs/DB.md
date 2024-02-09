# Data Structure in the db Folder

In this project, the `db` folder contains datasets divided into three subfolders: `train`, `validation`, and `test`. The split is as follows:

- **Training Dataset (`train`):**
  - Represents 60% of the total data.
  - Files in the `train/` folder:
    - `train_db_file1.csv`
    - `train_db_file2.csv`
    - ...

- **Test Dataset (`test`):**
  - Represents 20% of the total data.
  - Files in the `test/` folder:
    - `test_db_file1.csv`
    - `test_db_file2.csv`
    - ...

- **Validation Dataset (`validation`):**
  - Represents 20% of the total data.
  - Files in the `validation/` folder:
    - `validation_db_file1.csv`
    - `validation_db_file2.csv`
    - ...
  - Note: The validation dataset is intended for model evaluation but does not influence the training process.

This percentage-based split facilitates model evaluation at different stages of development. Make sure to check individual files for specific details about the format and content of each dataset.
