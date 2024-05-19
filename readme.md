### Detecting and Classifying Fishing Activity

#### Creating the environment on Windows S.O.

GPU Environment (Windows 10):
- install path.reg. It will enable long path support
- Python 3.12. Let Python installer set Python PATH automatically
- Drivers de GPU NVIDIA®: 450.80.02 or higher
- CUDA® Toolkit 11.2
- cuDNN SDK 8.1.0
- Copy folders and files from CUDNN path to same folders of CUDA Development Kit path
- install miniconda for windows
- execute create_env_gpu.bat 

CPU Environment (Windows 10):
- install path.reg. It will enable long path support
- Python 3.12. Let Python installer set Python PATH automatically
- execute create_env_cpu.bat 

#### Preparation phase

1- You must generate fishing spots and images representing kinematics of ships. The source is provided by Global Fishing Watching (GFW) available on its website.
Access ["Anonymized AIS training data"](https://globalfishingwatch.org/data-download/datasets/public-training-data-v1) and download these files:
- drifiting_longlines.csv;
- fixed_gear.csv;
- purse_seines.csv;
- trawlers.csv.

2- Put all those together inside data folder.

3- To complete the preparation, run python script called preparation.py. Both fishing spot file and kinematic images will be placed at data and images folder.
The preparation phase must be executed just one time. Once performed, you can train, test or load the model as you wish.

#### Training and (or) test the model

1- Perform preparation (create all files as described in earlier topic) before train the model; 
2- Execute train_and_test.py script. Once done, you will see results of training and test operations;
3- If you want to load and get the results of previously trained model, run load_and_test.py script. 

#### If you are here to reproduce the results published in Fusion 2024 conference 

1- Perform preparation (create all files as described in earlier topic) before execute the model;
2- Download all files from ["Fusion 2024"](https://github.com/pablorangel82/detecting-and-classifying-fishing-activity/releases);
3- Put those downloaded files into fusion_results folder;
4- Execute fusion_results.py. Will give to you the results obtained according as published in Fusion Conference.