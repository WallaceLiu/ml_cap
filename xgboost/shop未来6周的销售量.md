Boosting分类器属于集成学习模型，它基本思想是把成百上千个分类准确率较低的树模型组合起来，成为一个准确率很高的模型。这个模型会不断地迭代，每次迭代就生成一颗新的树。对于如何在每一步生成合理的树，大家提出了很多的方法，我们这里简要介绍由Friedman提出的Gradient Boosting Machine。它在生成每一棵树的时候采用梯度下降的思想，以之前生成的所有树为基础，向着最小化给定目标函数的方向多走一步。关于GBDT的理论介绍，生活实例以及代码，可以看下面链接
http://blog.csdn.net/tuntunwang/article/details/66969726

在合理的参数设置下，我们往往要生成一定数量的树才能达到令人满意的准确率。在数据集较大较复杂的时候，我们可能需要几千次迭代运算，如果生成一个树模型需要几秒钟，那么这么多迭代的运算耗时，应该能让你专心地想静静…

现在，我们希望能通过xgboost工具更好地解决这个问题。xgboost的全称是eXtreme Gradient Boosting。正如其名，它是Gradient Boosting Machine的一个c++实现（只是代码实现上的创新），作者为正在华盛顿大学研究机器学习的大牛陈天奇。他在研究中深感自己受制于现有库的计算速度和精度，因此在一年前开始着手搭建xgboost项目，并在去年夏天逐渐成型。xgboost最大的特点在于，它能够自动利用CPU的多线程进行并行，同时在算法上加以改进提高了精度。它的处女秀是Kaggle的希格斯子信号识别竞赛，因为出众的效率与较高的预测准确度在比赛论坛中引起了参赛选手的广泛关注，在1700多支队伍的激烈竞争中占有一席之地。随着它在Kaggle社区知名度的提高，最近也有队伍借助xgboost在比赛中夺得第一。为了方便大家使用，陈天奇将xgboost封装成了python库。

这里的背景是预测2000个shop未来6周的销售量。训练数据是2015-7-1至2016-10-30的流量（天池IJICAI）

数据下载地址 https://pan.baidu.com/s/1miz8CrA
```python
#encoding=utf-8
import pandas as pd
import numpy as np
import time
from sklearn import cross_validation
import xgboost as xgb
DATA_DIR="/home/wangtuntun/IJCAI/Data/"
shop_info_column_names=["shop_id","city_name","location_id","per_pay","score","comment_cnt","shop_level","cate_1","cate_2","cate_3"]
# user_pay_colimn_names=["user_id","shop_id","time_stamp"]#用python实现groupby方法不好实现，利用spark的sparkcontext.sql()实现，然后存取文件
shop_info=pd.read_csv(DATA_DIR+"shop_info.txt",names=shop_info_column_names)
flow_path="/home/wangtuntun/IJCAI/Data/ml_flow_raw_data_file.txt/part-00000"#这个文件是用sparkContext.sql()实现的，在本代码中不做代码展示。
merge_data_path="/home/wangtuntun/shop_info_flow.csv" #将合并后的特征存入该文件
feature_save_path="/home/wangtuntun/feature_data.csv"#将最终生成的特征存入该文件
#获取所有的城市名称
def get_all_city(shop_info):
    city_colomn=set(shop_info["city_name"])
    city_list=list(city_colomn)
    return city_list#一共122个城市
#获取所有的分类名称
def get_all_cate(shop_info):
    cate1_list=list(shop_info["cate_1"])
    cate2_list = list(shop_info["cate_2"])
    cate3_list = list(shop_info["cate_3"])
    cate_total=cate1_list+cate2_list+cate3_list
    cate_total=list(set(cate_total))
    return cate_total#一共67个城市
#将中文转为一个list   one-hot
def chinese2list(all_chinese,word_name):
    return_list=[]
    for i in all_chinese:
        if i == word_name:
            return_list.append(1)
        else:
            return_list.append(0)
    return return_list
#将数字转为list 如 星期一，只需要传入7,1就会返回  1,0,0,0,0,0,0  one-hot
def number2list(max_num,num):
    return_list=[]
    day=int(num)
    for i in range(1,max_num+1):
        if i == day:
            return_list.append(1)
        else:
            return_list.append(0)
    return return_list

#将shop_info的信息进行清洗
def clean_shop_info(shop_info):
    #缺失数据的填充
    shop_info.fillna({"city_name":"其他","cate_1":"其他","cate_2":"其他","cate_3":"其他"},inplace=True)
    shop_info.fillna(0,inplace=True)

#从外部文件读入flow并返回df
def get_flow(flow_path):
    f=open(flow_path,"r+")
    raw_data=f.readlines()
    id_list=[]
    time_stamp_list=[]
    year_list=[]
    month_list=[]
    day_list=[]
    day_of_week_list=[]
    flow_list=[]
    for ele in raw_data:
        ele=ele.split("(")[1]
        ele=ele.split(")")[0]
        ele=ele.split(",")
        id_list.append(ele[0].strip())
        date=ele[9].strip()
        time_stamp_list.append(date)
        time_format=time.strptime(date,'%Y-%m-%d')
        year_list.append(time_format.tm_year)
        month_list.append(time_format.tm_mon)
        day_list.append(time_format.tm_mday)
        day_of_week_list.append(time_format.tm_wday+1)
        print time_format
        flow_list.append(ele[10].strip())
    return_df=pd.DataFrame({"shop_id":id_list,"date":time_stamp_list,"year":year_list,"month":month_list,"day":day_list,"week_of_day":day_of_week_list,"flow":flow_list})
    return_df["shop_id"]=return_df["shop_id"].astype(int)
    return_df["flow"] = return_df["flow"].astype(float)
    return return_df

#将shop_info和flow进行合并并存入文件
def merge_shop_info_flow(flow_path,merge_save_path):#每次运行都需要进行merge操作，比较话费花费时间，所以直接先存入文件，以后直接取
    clean_shop_info(shop_info)
    flow_df=get_flow(flow_path)
    shop_info_flow=shop_info.merge(flow_df,on="shop_id",how="inner")
    shop_info_flow.to_csv(merge_save_path,index=False)#存入文件含有dataframe的列名，可以自己手动删除第一行。

#读取merge后的文件并提取特征存入文件
def build_features(merge_data_path,feature_path):#由于生成一次特征需要花费较长时间，一次性写入文件，之后读取。
    all_city_list = get_all_city(shop_info)
    all_cate_list = get_all_cate(shop_info)
    #读取初始特征
    merge_data=pd.read_csv(merge_data_path)
    #将dataframe转为二维array
    data=pd.np.array(merge_data)
    #获取每个shop的flow的max，min，ave
    max_dict={}
    min_dict={}
    ave_dict={}
    sum_dict={}
    count_dict={}
    all_shop_id_list=[]
    for line in data:
        all_shop_id_list.append(line[0])
    all_shop_id_set=set(all_shop_id_list)
    for shop in all_shop_id_set:
        max_dict[shop]=0
        min_dict[shop]=10000
        ave_dict[shop]=0
        sum_dict[shop]=0
        count_dict[shop]=0
    for line in data:
        flow=line[12]
        shop=line[0]
        sum_dict[shop] += flow
        count_dict[shop] += 1
        if max_dict[shop] < flow:
            max_dict[shop]=flow
        if min_dict[shop] > flow:
            min_dict[shop]= flow
    for shop in all_shop_id_set:
        ave_dict[shop]=sum_dict[shop] / count_dict[shop]
    #将city_name转为ont-hot编码
    transform_data=[]
    for line in data:
        list_temp=[]
        shop_id=line[0]
        list_temp.append(shop_id)#shop_id
        city_name_list=chinese2list(all_city_list,line[1])
        list_temp += city_name_list
        list_temp.append(line[2])#location_id
        list_temp.append(line[3])#per_pay
        list_temp.append(line[4])#score
        list_temp.append(line[5])#comment_cnt
        list_temp.append(line[6])#shop_level
        cate1_list=chinese2list(all_cate_list,line[7])
        list_temp += cate1_list
        cate2_list=chinese2list(all_cate_list,line[8])
        list_temp += cate2_list
        cate3_list=chinese2list(all_cate_list,line[9])
        list_temp += cate3_list
        #直接跳过line[10]   date
        day_list=number2list(31,line[11])#每个月最多有31天
        list_temp += day_list
        list_temp.append(line[12])#flow
        month_list=number2list(12,line[13])#每年做多有12个月
        list_temp += month_list
        week_of_day_list=number2list(7,line[14])#每个星期做多有7天
        list_temp += week_of_day_list
        list_temp.append(line[15])#year字段，如果把2015转为独热编码，字段就太多了
        list_temp.append(max_dict[shop_id])
        list_temp.append(min_dict[shop_id])
        list_temp.append(ave_dict[shop_id])
        transform_data.append(list_temp)
    pd.DataFrame(transform_data).to_csv(feature_path,index=False)

def get_features_target(data):
    data_array=pd.np.array(data)#传入dataframe，为了遍历，先转为array
    features_list=[]
    target_list=[]
    for line in data_array:
        temp_list=[]
        for i in range(0,384):#一共有384个特征
            if i == 360 :#index=360对应的特征是flow
                target_temp=int(line[i])
            else:
                temp_list.append(int(line[i]))
        features_list.append(temp_list)
        target_list.append(target_temp)
    # return features_list, target_list
    return pd.DataFrame(features_list),pd.DataFrame(target_list)

#得到评价指标rmspe_xg训练模型
def rmspe_xg(yhat, y):
    #y DMatrix对象
    y = y.get_label()
    #y.get_label 二维数组
    y = np.exp(y)#二维数组
    yhat = np.exp(yhat)#一维数组
    rmspe = np.sqrt(np.mean((y - yhat) ** 2))
    return "rmspe", rmspe

#该评价指标用来评价模型好坏
def rmspe(zip_list):
    # w = ToWeight(y)
    # rmspe = np.sqrt(np.mean((y - yhat) ** 2))
    sum_value=0.0
    count=len(zip_list)
    for real,predict in zip_list:
        v1=(real-predict)**2
        sum_value += v1
    v2=sum_value / count
    v3=np.sqrt(v2)
    return v3

def get_shop_number_dict():
    data = pd.read_csv(feature_save_path)
    data_array=pd.np.array(data)
    max_dict = {}
    min_dict = {}
    ave_dict = {}
    sum_dict = {}
    count_dict = {}
    all_shop_id_list = []
    for line in data_array:
        all_shop_id_list.append(line[0])
    all_shop_id_set = set(all_shop_id_list)
    for shop in all_shop_id_set:
        max_dict[shop] = 0
        min_dict[shop] = 10000
        ave_dict[shop] = 0
        sum_dict[shop] = 0
        count_dict[shop] = 0
    for line in data_array:
        flow = line[360]
        shop = line[0]
        sum_dict[shop] += flow
        count_dict[shop] += 1
        if max_dict[shop] < flow:
            max_dict[shop] = flow
        if min_dict[shop] > flow:
            min_dict[shop] = flow
    for shop in all_shop_id_set:
        ave_dict[shop] = sum_dict[shop] / count_dict[shop]
    return max_dict,min_dict,ave_dict

def predict_with_XGBoosting():
    '''
    #将两张表进行合并并存入文件
    merge_shop_info_flow(flow_path,merge_data_path)
    #提取特征并存入文件
    build_features(merge_data_path,feature_save_path)
    '''

    #获取训练集测试集验证集的 feature 和 target
    data=pd.read_csv(feature_save_path)
    data_other,data=cross_validation.train_test_split(data,test_size=0.001,random_state=10)#为了减少代码运行时间，方便测试
    train_and_valid,test=cross_validation.train_test_split(data,test_size=0.2,random_state=10)
    train,valid=cross_validation.train_test_split(train_and_valid,test_size=0.01,random_state=10)
    train_feature,train_target=get_features_target(train)
    test_feature,test_target=get_features_target(test)
    valid_feature,valid_target=get_features_target(valid)
    dtrain=xgb.DMatrix( train_feature,np.log(train_target) )#取log是为了数据更稳定
    dvalid=xgb.DMatrix( valid_feature,np.log(valid_target) )
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]

    #设置参数
    num_trees=450
    # num_trees=45
    params = {"objective": "reg:linear",
              "eta": 0.15,
              "max_depth": 8,
              "subsample": 0.7,
              "colsample_bytree": 0.7,
              "silent": 1
              }

    #训练模型
    gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=50, feval=rmspe_xg,verbose_eval=True)

    #获取shop_id_list
    shop_id_list = []
    test_feature_array=pd.np.array(test_feature)
    for line in test_feature_array:
        shop_id_list.append(line[0])

    #将测试集代入模型进行预测
    print("Make predictions on the test set")
    test_probs = gbm.predict(xgb.DMatrix(test_feature))
    predict_flow=list(np.exp(test_probs))

    #将shop_id,predicted_flow,real_flow  放在一起
    test_target_array = pd.np.array(test_target)
    test_target_list=[]
    for ele in test_target_array:
        test_target_list.append(ele[0])
    list_zip_id_real_predict=zip(shop_id_list,test_target_list,predict_flow)

    # 对预测结果进行矫正
    max_dict, min_dict, ave_dict = get_shop_number_dict()
    predict_flow_improve = []
    for shop_id,real,predict in list_zip_id_real_predict:
        # print shop_id,real,predict,max_dict[shop_id],min_dict[shop_id],ave_dict[shop_id]
        if predict > max_dict[shop_id]:
            predict = ave_dict[shop_id]
        if predict < min_dict[shop_id]:
            predict = ave_dict[shop_id]
        predict_flow_improve.append(predict)
    #计算误差
    list_zip_real_predict_improve = zip(test_target_list, predict_flow_improve)
    error = rmspe(list_zip_real_predict_improve)
    print('error', error)

if __name__ == '__main__':
    predict_with_XGBoosting()
```