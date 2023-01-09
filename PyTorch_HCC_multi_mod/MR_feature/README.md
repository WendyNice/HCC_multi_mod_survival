# classification_PyTorch
Using Deep learning to classify the sMRI data of DG(Disease Group) and HC(Healthy Control). 

Save the preprocessed data as np.array before run the codes, each image data save as one np.array file

1. run generate_csv.py to generate the train test csv file
2. run main.py file to train model
3. run test.py file to get test result


数据预处理

1 运行register_preprocessing.py进行重采样和配准 （运行debug.py看是否配准的好）
2 运行mask_preprocessing.py进行mask的重采样
3 运行move_data.py进行文件的整理
4 运行check_img_roi.py进行文件的检查
5 运行nnUNet分割代码




