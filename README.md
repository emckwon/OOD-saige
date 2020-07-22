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
           
## 2. Implement own losses and models
### loss
1. tools/train.py를 보면, loss에는 기본적으로 하나의 batch로 concat된 inlier와 outlier의 logits과 inlier의 targets이 들어가게 됩니다. 즉,
<pre>
<code>
    (tools/train.py)
        ...
        data = torch.cat((in_set[0], out_set[0]), 0)   # Concat inlier and outlier as batch
        targets = in_set[1]                            # Inlier's targets
        ...
        logits = model(data)
        loss = loss_func(logits, targets)
        ...
</code>
</pre>

2. loss_func은 utils/losses.py에 구현되어 있어야합니다. 이 파일에 있는 **_LOSSES** dictionary에 config.py 파일에서 지정할 이름을 key로 하고, loss function를 item으로 가지도록 구현한 loss함수를 추가해주어야 합니다. loss 함수는 해당 dictionary 위에 구현되어 있어야합니다. 기본적으로 *dont_care*와 *cross_entropy_in_distribution*을 구현해두었으니 참고바랍니다.

### Model
1. tools/train.py를 보면, model은 models/model_builder.py에 구현되어 있는 *getModel* 함수를 통해 가져옵니다. model을 불러오기 위해서는 model_builder.py의 **_MODEL_TYPES** dictionary에 config.py 파일에서 지정할 이름을 key로 하고, model class를 item으로 가지도록 model class를 추가해주어야 합니다.
2. model은 해당 dictionary 위에 같은 폴더의 각 model이 구현되어있는 python script로 부터 import 해주어야합니다.
3. 현재 WideResNet class를 불러올 수 있도록 해두었으니 새로운 model을 쓰실때 참고바랍니다.

## 3. Notice
1. 현재로서는 Single GPU를 이용한 실험만 가능하도록 구현해두었습니다.
2. 기본적으로 구현되어있는 models/wrn.py의 WideResNet는 32 by 32 image input을 가정하고 있는 model 입니다. 224 by 224 image input 사용시에는 같은 파일의 WideResNet224 class를 일단 사용하면 error는 나지 않지만, 제가 임의로 만들어낸 network이기 때문에 성능 보장은 할 수 없습니다.
3. OOD-detection metric들은 아직 추가되지 않았습니다. 최대한 빨리 추가예정이고, 필요하신 metric이 있으면 말씀해주세요.(train code도 건들여야해서 추가방법을 말씀드리겠습니다)

## 4. Running Code
1. {home}/OOD-saige/에 있는 config.py 파일을 실험에 맞게 설정해줍니다.(실험이 정상 시작되면, 해당 config.py는 해당 실험 폴더에 복사되어 저장됩니다. default_config.py는 수정하지 말아주세요.)
2. Terminal의 {home}/OOD-saige/ 폴더에 있는 상태에서 python으로 실행시켜주셔야합니다. 즉
<pre>
<code>
    :~{home}/OOD-saige >> python tools/train.py 
</code>
</pre>




