import os
import sys
import numpy as np
from datetime import datetime
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_PATH))
print sys.path
from offline import BaseSparkSession
 
default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)
 
os.environ['PYSPARK_PYTHON'] = 'F:\develop\python\Python27\python.exe'
os.environ['HADOOP_HOME'] = 'F:\develop\hadoop\hadoop-2.10.0'
os.environ['HADOOP_CONF_DIR'] = 'F:\develop\hadoop\hadoop-2.10.0-conf'
os.environ['SPARK_HOME'] = 'F:\develop\spark\spark-2.4.4-bin-hadoop2.7'
'''
需要将HBase和Hive相关的jar包拷贝到Spark的jars目录下
需要在SparkConf里面添加以下配置支持Hive查询HBase
("hbase.zookeeper.quorum", "192.168.0.1"),
("hbase.zookeeper.property.clientPort", "2181")
'''
 
 
class CtrSortModel(BaseSparkSession):
 
    def __init__(self):
        self.SPARK_APP_NAME = 'ctr_sort_model'
        self.SPARK_MASTER_URL = 'yarn'
        self.SPARK_YARN_QUEUE = 'queue3'
        self.ENABLE_HIVE_SUPPORT = True
        self.spark_session = self.create_spark_session()
 
    # 生成用户召回结果
    def gen_lr_sort_model(self):
        self.spark_session.sql("use portal")
        # 用户文章点击行为
        sql = "select user_id, article_id, channel_id, click_flag from t_user_behaviour"
        user_article_click_df = self.spark_session.sql(sql)
        user_article_click_df.show()
 
        # 获取用户画像 基础数据、偏好喜好数据
        sql = "select split(user_id, ':')[1] user_id, basic_info.gender, basic_info.age, preference_info " \
              "from t_user_profile"
        user_profile_df = self.spark_session.sql(sql)
        user_profile_df.show()
 
        user_article_click_df = user_article_click_df.join(user_profile_df, on=["user_id"], how="left")
 
        # 抽取文章所属频道关键词向量特征
        def extract_channel_keyword_feature(partition):
            from pyspark.ml.linalg import Vectors
            for row in partition:
                try:
                    weights = sorted([row.preference_info[key] for key in row.preference_info.keys()
                                     if key.split(':')[0] == row.channel_id], reverse=True)[:10]
                except Exception as e:
                    print e.message
                    weights = [0.0] * 10
                yield row.article_id, row.channel_id, row.user_id, int(row.gender), int(row.age), \
                    Vectors.dense(weights if weights else [0.0] * 10), row.click_flag
 
        user_article_click_df = user_article_click_df.rdd.mapPartitions(extract_channel_keyword_feature) \
            .toDF(["article_id", "channel_id", "user_id", "gender", "age", "channel_weights", "click_flag"])
        user_article_click_df.show()
 
        # 获取文章画像
        article_profile_df = self.spark_session.sql("select * from t_article_profile")
 
        # 抽取文章关键词向量特征
        def extract_feature(partition):
            from pyspark.ml.linalg import Vectors
            for row in partition:
                try:
                    weights = sorted(row.keywords.values(), reverse=True)[:10]
                except Exception as e:
                    print e.message
                    weights = [0.0] * 10
                yield row.article_id, Vectors.dense(weights if weights else [0.0] * 10)
        article_profile_df = article_profile_df.rdd.mapPartitions(extract_feature).toDF(["article_id", "article_weights"])
        article_profile_df.show()
 
        user_article_click_df = user_article_click_df.join(article_profile_df, on=["article_id"], how="inner")
        user_article_click_df.show()
 
        # 获取文章向量
        article_vector_df = self.spark_session.sql("select article_id, vector from t_article_vector")
 
        def array_to_vector(partition):
            from pyspark.ml.linalg import Vectors
            for row in partition:
                yield row.article_id, Vectors.dense(row.vector)
        article_vector_df = article_vector_df.rdd.mapPartitions(array_to_vector).toDF(["article_id", "article_vector"])
        article_vector_df.show()
 
        user_article_click_df = user_article_click_df.join(article_vector_df, on=["article_id"], how="inner")
        user_article_click_df.show()
 
        # 收集特征
        from pyspark.ml.feature import VectorAssembler
        input_cols = ["channel_id", "gender", "age", "channel_weights", "article_weights", "article_vector"]
        user_article_click_df = VectorAssembler().setInputCols(input_cols) \
                                                 .setOutputCol("features") \
                                                 .transform(user_article_click_df)
        user_article_click_df.show()
 
        # Logistic Regression
        from pyspark.ml.classification import LogisticRegression
        logistic_regression = LogisticRegression()
        logistic_regression_model = logistic_regression.setFeaturesCol("features") \
                                                       .setLabelCol("click_flag")\
                                                       .fit(user_article_click_df)
        logistic_regression_model.write().overwrite().save(
            "hdfs://192.168.0.1:9000/user/models/logistic_regression/lr.model")
 
        from pyspark.ml.classification import LogisticRegressionModel
        logistic_regression_model = LogisticRegressionModel.load(
            "hdfs://192.168.0.1:9000/user/models/logistic_regression/lr.model")
        logistic_regression_result = logistic_regression_model.transform(user_article_click_df)
        logistic_regression_result.select(["click_flag", "probability", "prediction"]).show()
 
        # ROC
        def vector_to_double(row):
            return float(row.click_flag), float(row.probability[1])
        score_labels = logistic_regression_result.select(["click_flag", "probability"]).rdd.map(vector_to_double)
        score_labels.collect()
        from pyspark.mllib.evaluation import BinaryClassificationMetrics
        binary_classification_metrics = BinaryClassificationMetrics(scoreAndLabels=score_labels)
        area_under_roc = binary_classification_metrics.areaUnderROC
        print area_under_roc
 
    @staticmethod
    def gen_lr_sort_model_metrics(test_df):
        from pyspark.ml.classification import LogisticRegressionModel
        logistic_regression_model = LogisticRegressionModel.load(
            "hdfs://192.168.0.1:9000/user/models/logistic_regression/lr.model")
        lr_result = logistic_regression_model.evaluate(test_df).predictions
        lr_result.show()
 
        def vector_to_double(row):
            return float(row.click_flag), float(row.probability[1])
        score_labels = lr_result.select(["click_flag", "probability"]).rdd.map(vector_to_double)
        score_labels.collect()
 
        from pyspark.mllib.evaluation import BinaryClassificationMetrics
        binary_classification_metrics = BinaryClassificationMetrics(scoreAndLabels=score_labels)
        area_under_roc = binary_classification_metrics.areaUnderROC
        print area_under_roc
 
        tp = lr_result[(lr_result.click_flag == 1) & (lr_result.prediction == 1)].count()
        tn = lr_result[(lr_result.click_flag == 0) & (lr_result.prediction == 1)].count()
        fp = lr_result[(lr_result.click_flag == 0) & (lr_result.prediction == 1)].count()
        fn = lr_result[(lr_result.click_flag == 1) & (lr_result.prediction == 0)].count()
        print "tp {} tn {} fp {} fn {}".format(tp, tn, fp, fn)
        print('accuracy is : %f' % ((tp + tn) / (tp + tn + fp + fn)))
        print('recall is : %f' % (tp / (tp + fn)))
        print('precision is : %f' % (tp / (tp + fp)))
 
 
if __name__ == '__main__':
    ctr_sort_model = CtrSortModel()
    ctr_sort_model.gen_lr_sort_model()
