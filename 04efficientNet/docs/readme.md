# 文件说明

## 1\. 目录结构说明

### 1.1 dataset_images（数据来源多中心）

* images为所有原图，结构目录为imags/access_no/sop_uid.jpg
* us_images经过超声裁剪模型处理后的图像，结构目录同上。
* nodule_images在超声裁剪后的图片基础上，使用“结节分割+检出”组合逻辑（参考2.推理逻辑step2）进行处理的图像，结构目录同上。
* logs为复制数据图片记录，包括缺失的数量。分为train数据表"/For_Eton/dataset_table/all_matched_sops_ds_v3_1015_v2.csv"和独立验证集"/For_Eton/dataset_table/val/all_verify_sop_with_predictions.csv"两种数据集的迁移。
* 其他补充信息：存在缺失图像，有可能未检测到超声区域或者结节区域。已提供分割模型和检测模型，结节检出逻辑参考2.1推理逻辑。

### 1.2 models

* nodule_feature_cnn_v75为最新模型训练中间内容
* config.yaml模型阈值、映射字典、模型功能和架构
* verification_report_preprocessed_comparisonv75 最新模型在独立验证集上的指标

### 1.3 dataset_table

* "train/all_matched_sops_ds_v3_with_tr13_1016.csv" 为12w图片数据，
    - 其中8w来源于"For_Eton/dataset_table/all_matched_sops_ds_v3_1015_v2.csv"（穿刺病理+手术病理匹配超声图像），
    - 其中4.5w来源于"For_Eton/dataset_table/none_single_tr13.csv"（tr1-3良性数据）;
* "train/all_matched_sops_ds_v3_with_tr13_0926_with_OOF_suspect.csv"为12w数据的基础上，使用"For_Eton/sop5_OOF_analysis_sop7_all_match_0928.py"进行OOF处理，得到的表，**训练时：剔除02.202408/02.202409开头的access数据（作验证）、p_true<0.2（OOF认为可能有问题的数据）**

* "val/all_verify_sop_with_predictions.csv"为多个 ==分中心数据子集== （区分子集通过type列）的独立验证集，参考 verification_report_preprocessed_comparisonv75，其中 ==0809系列集合== 中，
    - v2版本质量差，原代码匹配版本；
    - **v3版本为怀珠金标准，参考价值最高**；
    - v4版本为在train/all_matched_sops_ds_v3_with_tr13_1016.csv中的0809数据，
    - v4_subset为剔除oof小于0.2的低质量数据子集；
suda系列中数据质量不高，也是本次临床试验中分中心表现最差的。jida表现最好，但数量少。 

### 1.4 candidate_table

* 所有候选数据（训练数据处理中间表）
* all_nodules_path_ds_v2.csv 病理结节表 key： exam_no+nindex（p_index）
* all_nodules_us_ds_v2.csv超声结节表 key：access_no+nindex
* all_candidated_ultrasound_v2.csv原始超声信息表 us_1_3_candidate.csv部分tr1-3的超声数据 key：access_no
* all_pathology.csv原始病理数据表 key：exam_no

### 1.5 code

* create_sop4_training_images_backup.py  - 准备训练集（裁图）
* create_sop4_verify_images.py  - 准备独立验证集（裁图）
* train_nodule_feature_cnn_model_v75.py - 训练结节鉴别模型
* sop5_OOF_analysis_sop7_all_match_0928.py  \& sop5_OOF_analysis_sop7_all_match_tr13_0927.py  - 做oof的代码
* inference_verify.py  - 独立验证集的推理代码

## 2\. 目前训练情况与瓶颈

### 2.1 推理逻辑

* step1. 对于一张图片，使用超声区域检测模型进行裁剪，得到us_image。
* step2. 同时调用分割模型和检测模型，优先使用分割模型结果（超过阈值config计算最小矩形框），失败则使用检测模型（超过阈值）.
* step3. 取超过设定阈值的检测框，进行扩增裁剪2W\*2H，得到nodule_image。
* step4. 进行结节鉴别，使用结节分类模型，根据置信度阈值得到良恶性标签。（图片级别的模型指标即可计算）
* step5. 由于一个结节有横纵两个切面图片，考虑综合两张图片加权计算出结节级别良恶标签。（结节级别的模型acc预计会比图片级别高3%）逻辑如下：

- **计算逻辑**:
```text
# 示例：图1阴性80% + 图2阳性60%;
# 图1阳性概率 = 1 - 0.8 = 0.2;
# 图2阳性概率 = 0.6;
# 综合阳性概率 = 0.2 \* 0.5 + 0.6 \* 0.5 = 0.4 (40%);
# 结果：{阴性} 60%;
```

### 2.2 瓶颈

* 模型本身指标不高，指标在独立验证集上的指标效果不好。
* 训练的sop表是代码进行匹配的，有可能数据本身质量不高
* 如何根据图片判断良恶标签计算结节级别良恶标签？平权？
* 检出不准？



### 3\. 目标与需求

### 3.1 模型指标

* 临床试验指标specificity和sensitivity，模型辅助医生阅片在两个指标上均高于医生独立阅片（低年资acc~83%，spec~98%），且spec与sensitivity的差小于10%。
* 科委要求：acc指标高于85%
* 行业对家指标acc92%，综上考虑acc在89%以上，需要兼容三家分中心的指标。

### 3.2 模型训练

* 训练时需要配套**生成onnx格式模型**，软件开发嵌套需要！
