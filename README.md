# OOD-saige
Out-of-distribution detection task experiment in saigereasearch

## datasets folder
1. data_loader.py has getDataSet function for building dataset and loader.
2. Sub directory data_split has split(train/valid/test) list of each data. Structure is as below.
    root/datasets/{DATASET_NAME}/{split}.txt
    {label1}/{image_file_name1.png}
    {label2}/{image_file_name2.png}
    ...
            
