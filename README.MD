## AI特征预测项目
这是用于预测`OLT接收光功率(dBm),发送光功率(dBm),接收光功率(dBm)`三个属性的代码和训练好的模型包。所有数据位于data\standard_1文件夹下  
所有训练好的模型参数位于save_models文件夹下。data_preprocessing.py包含了数据清洗，数据预处理和数据增强的代码。model.py包含了尝试使用的模型的代码。
main_v2.py, main.py是主要的用于训练的代码。test.py, test_v2.py是用于预测新数据的代码。所有代码文件均含注释。  
目前，v2的两个文件是主要使用的文件  

### 数据
目前的模型采用了提供的”标准数据“压缩包中的数据进行训练，即八月中上旬的数据。如果需要增添新的数据，解压”标准数据2“压缩包或者其他导出数据，
对所有.xlsx文件执行操作（数据详细操作中会提到），然后放到data/standard_1下面即可。注意，如果要增添文件，则要在main.py, main_v2.py里面修改文件的个数。(files这个参数)

### 数据详细操作
1. 变成CSV文件（utf-8,逗号分隔），这个可以用将所有数据另外保存到一个excel里面，保存时候选择格式的方法得到  
2. 训练时，##需要去掉这几栏：网元IP，框号，操作状态，ONU TYPE，最后一次上线，离线时间，最后一次离线原因，保存数据前直接excel选中列去掉，
最新更新，和测试一样，不需要去掉栏目啦  
3. 测试时不用去掉那几栏，只需要excel转csv即可

### 模型(目前能用的只有XGB和MLP--即参数xgb和reg对应的模型)
deep model:  
MLP: unknown
xgb model: unknown
LSTM model：74/880
目前，模型有Xgbboost, MLP神经网络，LSTM时序模型，LSTM经过训练可以检测出880个异常值中的74个

### 使用
在根目录下打开电脑命令行（也就是点进去大文件夹）电脑命令行依次输入如下命令：  
`conda create -n forecasting python=3.8`  
`conda activate forecasting`  
`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`  
`python main_v2.py`  
`python test_v2.py`  
目前保持test_v2.py的配置不要动

###预测结果
结果在save_results文件夹下面，叫results.csv，也可以自己跑  
注意这个是清洗过的数据

###模型调优思路
XGB模型：main.py写好了adj_param函数，可以调整min_depth等等诸多参数（在params={}这个字典里加！）  
比如，想调整eta（学习率），增加：params={'eta':[0.01,0.03, 0.05, 0.1, 0.3]}等（网上有相关资料）  
MLP等深度学习模型，调节隐藏层参数多少，层数，学习率，优化器等等
