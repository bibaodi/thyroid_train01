1. estimate_mul_batch.py是支持标注数据作为输入, 同时进行dice计算的模型处理脚本.
2. estimate_dcm_batch.py是支持dcm文件作为输入, 仅进行预测, 以两张并列图片形式进行显示. 此程序是为了与手机端程序模型识别精度进行比对所生成的.
	- 使用方式python estimate_dcm_batch.py dcm_file 运行完成后, 会在dcm_file所在文件生成一个包含处理后图片的文件夹. 
3. evaluate_plaque_tflite.py 是基于斑块模型对斑块的识别进行评估.
	- 使用方法参见内部提示.
4. evaluate_thyroid_nodule.py是针对detect模型的甲状腺结节识别, 模型来自小白世纪. 程序参考tensorflow的tflite的detect的demo.
5. cal_imt_base_tflite.py是基于包含IM识别的模型进行IMT的计算
	- 使用方式与斑块的那个一样