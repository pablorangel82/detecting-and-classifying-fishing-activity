### Detecting and Classifying Fishing Activity

#### Setting up the Environment on Windows OS

**GPU Environment (Windows 10):**
1. Install "path.reg" to enable long path support.
2. Install Python 3.10. Allow the Python installer to set the Python PATH automatically.
3. Update your NVIDIA Drivers to version 450.80.02 or higher.
4. Install CUDA® Toolkit 11.2.
5. Install cuDNN SDK 8.1.0.
6. Copy folders and files from the cuDNN path to the corresponding folders in the CUDA Development Kit path.
7. Install miniconda for Windows.
8. Finally, run `create_env_gpu.bat`.

**CPU Environment (Windows 10):**
1. Install "path.reg" to enable long path support.
2. Install Python 3.10. Allow the Python installer to set the Python PATH automatically.
3. Finally, run `create_env_cpu.bat`.

#### Preparation Phase

1. **Generate Fishing Spots and Images Representing Ship Kinematics:**
   - The source data is provided by Global Fishing Watch (GFW) and is available on their website. Access ["Anonymized AIS training data"](https://globalfishingwatch.org/data-download/datasets/public-training-data-v1) and download these files:
     - drifting_longlines.csv
     - fixed_gear.csv
     - purse_seines.csv
     - trawlers.csv

2. **Organize the Data:**
   - Place all the downloaded files inside the `data` folder.

3. **Run the Preparation Script:**
   - To complete the preparation, run the Python script called `preparation.py`. Both the fishing spot file and kinematic images will be placed in the `data` and `images` folders, respectively.
   - Note: The preparation phase must be executed only once. Once completed, you can train, test, or load the model as needed.

#### Training and Testing the Model

1. **Preparation:**
   - Ensure the preparation step (creating all necessary files as described above) is completed before training the model.

2. **Training and Testing:**
   - Execute the `train_and_test.py` script. Once done, you will see the results of the training and testing operations.

3. **Loading a Pre-trained Model:**
   - If you want to load and get the results of a previously trained model, run the `load_and_test.py` script.

#### Reproducing the Results Published in Fusion 2024 Conference

1. **Preparation:**
   - Ensure the preparation step (creating all necessary files as described above) is completed before executing the model.

2. **Download Conference Files:**
   - Download all files from ["Fusion 2024"](https://github.com/pablorangel82/detecting-and-classifying-fishing-activity/releases/tag/v1.0.0-fusion2024).

3. **Organize the Conference Data:**
   - Place the downloaded files into the `fusion_results` folder.

4. **Execute the Results Script:**
   - Run the `fusion_results.py` script. This will provide you with the results obtained as published in the Fusion Conference.

5. **Cite the Work:**
   - Please cite us using this DOI number: [NOT AVAILABLE YET].