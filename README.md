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
2. losses - **add additional loss function!**
3. metirc - **add additional metric!**
4. optimizer
5. detector - **add additional detector!**



### results
1. All results will be saved in this folder.
<pre>
<code>
    root/results/{exp_dir}
                        |___config.py  # backup config file of this experiment.
                        |___ckpt       # Folder for saving checkpoint file.
                        |___log        # Folder for saving tensorboard event file.
                        |___valid      # Folder for saving validation results
                                |___valid_config.py # Validation exp. config file
                                |___{Dataset name}  # Result folder exp. with {out-distribution dataset} which use in validation.
</code>
</pre>
           
## 2. Implement own losses, detectors and models
### Loss
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
3. loss function은 logits와 targets, cfg['loss'] 받아서 loss를 계산하고, dict 타입의 자료형을 리턴합니다.
4. 리턴하는 dictionary는 반드시 'loss': (loss value)의 key, value pair를 가지고 있어야합니다. loss value는 보통의 loss 값 처럼 실수형이여야 합니다.
5. 이외에도 dictionary에는 임의의 key와 해당 key의 value를 담아 return 할 수 있게 하여, 특수한 구현의 경우 메인 코드로 loss function 내부에서 생성되는 값을 전달하여, 메인 코드에서 해당 값을 활용하거나, 로그할 수 있도록 쓸 수 있습니다.

### Model
1. tools/train.py를 보면, model은 models/model_builder.py에 구현되어 있는 *getModel* 함수를 통해 가져옵니다. model을 불러오기 위해서는 model_builder.py의 **_MODEL_TYPES** dictionary에 config.py 파일에서 지정할 이름을 key로 하고, model class를 item으로 가지도록 model class를 추가해주어야 합니다.
2. model은 해당 dictionary 위에 같은 폴더의 각 model이 구현되어있는 python script로 부터 import 해주어야합니다.
3. 현재 WideResNet class를 불러올 수 있도록 해두었으니 새로운 model을 쓰실때 참고바랍니다.

### Detector
1. outlier와 같이 train이나 validation을 진행하는 경우, inlier와 outlier의 logit을 보고 confidence score를 계산하는 부분입니다. 이로부터 계산된 confidence score는 다양한 metric에서 사용될 수 있습니다.
2. 기본적으로 loss와 매우 유사합니다. input으로 받는 것(logits과 targets, cfg['detector'])이나 utils/losses.py와 utils/detectors.py의 구조나 호출방법은 거의 같다고 보시면 됩니다.
3. loss와의 차이점은 return하는 dictionary는 반드시 'confidences' key와 해당 key에 대한 item으로 각 sample에 대한 confidence score을 가져야한다는 점입니다. confidence score는 가령 input으로 받은 logits의 차원이 (batch_size, logit_size)였다면, (batch_size, 1)의 차원을 가지는 tensor입니다.
4. 또한 'inlier_mean'과 'outlier_mean' key도 가져서 아래의 값들은 필수 입니다.
<pre>
<code>
    ...
    return {
        'confidences': confidences,
        'inlier_mean': confidences[:len(targets)].data.cpu().mean(),
        'outlier_mean': confidences[len(targets):].data.cpu().mean()
    }
</code>
</pre>

## 3. Datasets
## Avaliable now
CIFAR-10/100, SVHN, TinyImageNet, Severtal, HBT, SDI, DAGM, ImageNet_resize, Lsun_resize

## Dataset Configuration
config.py에서 dataset에 관한 옵션을 설정하는 방법에 대한 설명입니다. 기본적으로 *In-Dataset config*와 *Out-Dataset config*로 두 section으로 구분되어 있으며, 각각 *cfg['in_dataset']*과 *cfg['out_dataset']* dictionary에서 옵션 설정이 가능합니다.

1. ['dataset']의 item으로 사용할 dataset의 선택이 가능합니다.
2. ['targets']로 해당 dataset에서 실제로 사용할 class만 선택가능하지만, /Openset/ 폴더 내에 있는 데이터셋에서만 사용가능한 옵션입니다.(모두 쓸려면, 모든 class의 모든 라벨을 열거해야합니다)
3. ['~_transform']로 각 split에서 사용할 transform을 지정해야합니다.
4. ['data_root']는 실제 data가 들어있는 directory의 root path가 지정되어야 합니다.
5. ['split_root']는 사용할 train/valid/test.txt split이 들어있는 directory의 root path가 지정되어야 합니다.
6. **[Important]** 만약 out-distribution을 사용하지 않으려면 cfg['out_dataset']=None을 지정하고, cfg['out_dataset']에 관한 모든 하위 옵션은 주석처리 해주셔야합니다.
7. ['split']으로 해당 데이터 셋의 어떤 split을 사용할지 custom하게 정할 수 있습니다.(아예 입력을 안하면 기본적으로 train code에서는 train split이, valid code에서는 valid split이 사용됩니다. 그리고 ['split']옵션 값에 따라 transform도 결정되니 주의하십시오.)
8. ['batch_size']로 in/out distribution 각 dataset의 batch size를 custom하게 조절할 수 있습니다.(지금까지는 거의 1:1로 sampling)

['dataset']에 가능한 목록은 다음과 같습니다.
<pre>
<code>
    (32x32 datasets)
    'cifar10', 'cifar100', 'svhn', 'tinyimagenet', 'imagenet_resize', 'lsun_resize'
    (256x256, etc...)
    'Severstal', 'DAGM', 'HBT/LAMI', 'HBT/NUDE', 'SDI/34Ah', 'SDI/37Ah', 'SDI/60Ah'
</code>
</pre>

다음은 예시입니다. 해당 옵션으로는 Inlier로, 'Severstal'을 쓰지만 2번째 결함 class는 in-distribution으로 사용하지 않습니다. Outlier는 'cifar10'의 모든 data를 이용해 학습합니다.
<pre>
<code>
    (config.py)
        ...
        # In-Dataset config
        cfg['in_dataset'] = dict()
        cfg['in_dataset']['split'] = 'train' # train transform will be used.
        cfg['in_dataset']['batch_size'] = 64
        cfg['in_dataset']['dataset'] = 'Severstal'
        cfg['in_dataset']['targets'] = ['ok','1', '3', '4']
        ...
        
        # Out-Dataset config
        cfg['out_dataset'] = dict()
        cfg['out_dataset']['split'] = 'valid' # valid transform will be used.
        cfg['out_dataset']['batch_size'] = 32
        cfg['out_dataset']['dataset'] = 'cifar10'
        #cfg['out_dataset']['targets'] = []  # This option will not be activated.
        ...
</code>
</pre>


## 4. Notice
1. 현재로서는 Single GPU를 이용한 실험만 가능하도록 구현해두었습니다.
2. PGD (Projected Gradient Descent) attack module이 구현되었습니다. 현재로서는 valid.py 코드에서 validation시 선택적으로 사용가능한 옵션입니다. 필요할 경우 각 논문 구현시 training 코드에 적용시켜 사용할 수도 있습니다. cfg['PGD']라는 새로운 옵션에서 ['epsilon'], ['iters'], ['iter_size']를 설정해주셔야 사용가능합니다.

## 5. Running Code
## Training
Model의 training과 finetuning을 목적으로 사용하는 script입니다.

1. {home}/OOD-saige/에 있는 config.py 파일을 실험에 맞게 설정해줍니다.(실험이 정상 시작되면, 해당 config.py는 해당 실험 폴더에 복사되어 저장됩니다. default_config.py는 백업용이니 되도록 수정하지 말아주세요.)
2. Terminal의 {home}/OOD-saige/ 폴더에 있는 상태에서 python으로 실행시켜주셔야합니다. 즉
<pre>
<code>
    :~{home}/OOD-saige >> python tools/train.py 
</code>
</pre>
3. 학습시 사용한 config.py는 각 실험 폴더 내부에 백업됩니다.

**IMPORTANT** 기존의 model을 가져와서 다른 loss나 데이터셋으로 **finetuning**을 하는 경우 되도록 **cfg['exp_dir']** 을 별도로 지정하여 새로운 실험 폴더를 만들어서 하는 것을 추천드립니다.(기존 model의 training config나 checkpoint가 덮어씌워질 수도 있습니다.) finetuning 할 모델은 **cfg['load_ckpt']** 옵션에 checkpoint 파일의 절대경로를 지정하면 됩니다.

## Validation
Model의 evaluation을 목적으로 사용하는 script입니다.

1. {home}/OOD-saige/에 있는 config.py 파일을 실험에 맞게 설정해줍니다.(실험이 정상 시작되면, 해당 config.py는 해당 실험 폴더에 복사되어 저장됩니다. default_config.py는 수정하지 말아주세요.)
2. Dataset config 옵션은 이전과 동일하게 설정하면 됩니다. 중요한 점은 **cfg['load_ckpt']** 옵션에 불러 올 checkpoint의 절대 경로를 명시해주셔야 합니다.
3. Terminal의 {home}/OOD-saige/ 폴더에 있는 상태에서 python으로 실행시켜주셔야합니다. 즉
<pre>
<code>
    :~{home}/OOD-saige >> python tools/valid.py 
</code>
</pre>
4. Validation 시 사용한 config.py는 각 실험 폴더 내부에 **valid** 폴더 내부에 결과와 함께 백업됩니다.

## Custom Train/Valid code
코드 구조를 작성하면서 최대한 하나의 train.py와 valid.py에서 모든 구현에 대해 사용이 가능하도록 짜려고 노력은 하였으나, 제 역량 부족과 예외가 많은 구현 사항이 늘어나면서 이를 하나의 코드로 관리하기에는 심하게 비효율적이게 되었습니다. 따라서 특정 task를 구현하다가 기본 train.py, valid.py로 학습/검증이 불가능하거나 복잡성이 너무 올라가는 경우. 그냥 tools 폴더 내부에 해당 구현을 위한 code를 하나더 작성하시고 사용하시면 됩니다. 하지만 여러 기능이 밀접히 연관되어 있는 만큼, 되도록 train/valid epoch 함수 내부만 건드리시는 걸 추천드립니다.




