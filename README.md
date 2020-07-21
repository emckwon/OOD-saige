# OOD-saige
Out-of-distribution detection task experiment in saigereasearch.

## 1. Repository structure
### datasets
1. data_loader.py has getDataSet function for building dataset and loader.
2. Sub directory data_split has split(train/valid/test) list of each data. Structure is as below.
<pre>
<code>
    (root/datasets/{DATASET_NAME}/{split}.txt)
        {label1}/{image_file_name1.png}
        {label2}/{image_file_name2.png}
        ...
</code>
</pre>
3. saige_dataset.py has dataset class for saigeresearch's datasets.

### models
1. model_builder.py wrapping nn.Module class networks. All model call by this script in training/test codes.
2. basic_blocks.py has basic block of residual network and network block class.
3. wrn.py has wide residual network class.

### utils
1. data split ipynb file
2. losses - add additional loss function!
3. metirc
4. optimizer


### results
1. All results will be saved in this folder.
<pre>
<code>
    root/results/{exp_dir}
                        |___config.py  # backup config file of this experiment.
                        |___ckpt       # Folder for saving checkpoint file.
                        |___log        # Folder for saving tensorboard event file.
</code>
</pre>
            



