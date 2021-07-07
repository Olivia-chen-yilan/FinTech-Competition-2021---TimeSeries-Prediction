import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime


# TODO：写注释
def bucket_data(lines):
    # bucket的类型为dictionary
    bucket = {}
    # 厉遍每一行数据
    for line in lines:
        # minute_series作为key
        time_series = line[-2]
        bucket[time_series] = []
    for line in lines:
        time_series, y1 = line[-2:]
        # 删除time_series数据
        line = np.delete(line, -2, axis=0)
        # 将其他特征的数值作为bucket的value内容
        bucket[time_series].append(line)
    # print(bucket)
    return bucket


def cross_valid(regressor, bucket, lagging):
    valid_loss = []
    last = [[] for i in range(len(bucket[list(bucket.keys())[0]]))]
    # last=[[],[],[],[],[]]
    # print('last',last)
    for time_series in sorted(bucket.keys(), key=float):
        # if time_series >= 120:
        # if int(time_series) in range(120, 120 + lagging * 2, 2):
        if int(time_series) in range(0, 0 + lagging * 30, 30):
            last = np.concatenate((last, np.array(bucket[time_series], dtype=float)[:, -1].reshape(-1, 1)), axis=1)
        else:
            # print('bucket[time_series]长度：',len(bucket[time_series][0]))
            batch = np.array(bucket[time_series], dtype=float)
            y = batch[:, -1]
            batch = np.delete(batch, -1, axis=1)
            batch = np.concatenate((batch, last), axis=1)
            # print('batch:',batch)
            # print('batch 长度：' ,len(batch[0]))
            y_pre = regressor.predict(batch)
            last = np.delete(last, 0, axis=1)
            last = np.concatenate((last, y_pre.reshape(-1, 1)), axis=1)
            loss = np.sum(np.abs((y - y_pre) / (y+1))) / len(y)
                # np.mean(abs(np.expm1(y) - np.expm1(y_pre)) / np.expm1(y)) #emp1= exp(x)-1
            valid_loss.append(loss)
    # print 'day: %d loss: %f' % (int(day), day_loss)
    return np.mean(valid_loss)


# def mape_ln(y, d):
#     c = d.get_label()
#     result = np.sum(np.abs((np.expm1(y) - np.expm1(c)) / np.expm1(c))) / len(c)
#     return "mape", result

# 计算MAPE
def mape_fintech(y_true, y_pred):
    c = y_pred.get_label()
    result = np.sum(np.abs((y_true - c) / (y_true+1))) / len(y_true)
    return "mape", result

# 查看特征的重要性程度
def feature_vis(regressor, train_feature):
    importances = regressor.feature_importances_
    indices = np.argsort(importances)[::-1]
    selected_features = [train_feature[e] for e in indices]
    plt.figure(figsize=(20, 10))
    plt.title("train_feature importances")
    plt.bar(range(len(train_feature)), importances[indices],
            color="r", align="center")
    plt.xticks(range(len(selected_features)), selected_features, rotation=70)
    plt.show()


# ------------------------------------------------Submission ---------------------------------------------

# 提交数据集
def fintech_submission(train_feature, regressor, df, task1_file, task2_file):
    '''
    输出比赛提交数据
    :param train_feature:训练集中的特征
    :param regressor: 调试好后的模型
    :param df: 整理好格式后的测试数据集
    :param task1_file: 任务1的输出文件名
    :param task2_file: 任务2的输出文件名
    :return: 无
    '''
    # df为与训练数据集格式相同的测试集df
    test_df = df
    # 进行一次lagging前移
    test_df['lagging5'] = test_df['lagging4']
    test_df['lagging4'] = test_df['lagging3']
    test_df['lagging3'] = test_df['lagging2']
    test_df['lagging2'] = test_df['lagging1']
    test_df['lagging1'] = test_df['amount']

    with open(task1_file, 'w'):
        pass
    with open(task2_file, 'w'):
        pass

    # 读取test_v1_periods.csv
    test_v1_periods_df = pd.read_csv('target/test_v2_periods.csv', parse_dates=['date'])
    test_v1_periods_df.drop(['amount'], axis=1, inplace=True)
    # 读取test_v1_day.csv
    test_v1_day_df = pd.read_csv('target/test_v2_day.csv', parse_dates=['date'])
    test_v1_day_df.drop(['amount'], axis=1, inplace=True)
    # 循环预测30个周期，30天
    result = pd.DataFrame(columns=['date', 'post_id', 'periods', 'amount'])
    for day in range(1,31):
        # TODO:根据测试集的开始日期进行修改
        # 单独处理某一天的数据
        test_filter_df = test_df.loc[((test_df['date'].dt.year == 2020) & (test_df['date'].dt.month == 12)
                          & (test_df['date'].dt.day == day) )].copy()
        # print(test_filter_df.head())
        # 循环预测20个周期，从8点半到18点半
        # for period in range(20):
        test_X = test_filter_df[train_feature]
        # print('test_x',test_X)
        y_prediction = regressor.predict(test_X.values)
        # y_prediction=np.ceil(y_prediction)
        test_filter_df['lagging5'] = test_filter_df['lagging4']
        test_filter_df['lagging4'] = test_filter_df['lagging3']
        test_filter_df['lagging3'] = test_filter_df['lagging2']
        test_filter_df['lagging2'] = test_filter_df['lagging1']
        test_filter_df['lagging1'] = y_prediction

        #提交文件中的列date	post_id	periods	amount
        target_df = test_filter_df[['date', 'post_id', 'periods']]
        # 进行格式处理
        target_df['date'] = target_df['date'].map(lambda x: x.strftime('%Y/%m/%d'))
        target_df['post_id'] = target_df['post_id']
        target_df['periods'] = target_df['periods']

        # TODO:进行选择，根据是否对原始数据进行了取对数处理
        target_df['amount'] = y_prediction
        result=result.append(target_df)
            # print(target_df)
            # target_df['amount'] = np.expm1(y_prediction)

            # 处理任务2文件
            # 根据date和post_id，还有periods进行分组处
    target_df=result.copy()
    # target_df.to_csv('data/targetdf.csv')
    target_task2_group_df=target_df.groupby(['date','post_id','periods'],as_index=False)['amount'].sum()
    # print('target_task2_group_df',target_task2_group_df)
    target_task2_group_df['date']=pd.to_datetime(target_task2_group_df['date'])
    test_v1_periods_merge_df = pd.merge(test_v1_periods_df, target_task2_group_df, on=['date', 'post_id', 'periods'],
                                          how='left')
    # for i in range(len(test_v1_periods_df)):
    #     date = test_v1_periods_df.iloc[i]['date']
    #     post_id = test_v1_periods_df.iloc[i]['post_id']
    #     periods = test_v1_periods_df.iloc[i]['periods']
    #     if date in list(target_task2_group_df['date']):
    #         if post_id in list(target_task2_group_df['post_id']):
    #             if periods in list(target_task2_group_df['periods']):
    #                 test_v1_periods_df['amount'][i] = target_task2_group_df.iloc[i]['amount']

    # 循环预测完30天和每天的20个时段后，再填充没有数据的数值为0
    test_v1_periods_final_df = test_v1_periods_merge_df.fillna(0)
    # 输出为csv文件
    test_v1_periods_final_df['amount']=test_v1_periods_final_df['amount'].round(0)
    test_v1_periods_final_df['amount'] = test_v1_periods_final_df['amount'].astype('int')
    test_v1_periods_final_df[['date', 'post_id', 'periods', 'amount']].to_csv(task2_file, mode='w', header=True, index=False, sep=',', encoding = "utf-8")

    # 处理任务1文件
    # 根据date和post_id进行分组处理
    target_task1_group_df=test_v1_periods_final_df.groupby(['date','post_id'])['amount'].sum()
    # 与test_df进行合并
    test_v1_day_merged_df = pd.merge(test_v1_day_df, target_task1_group_df,
                                         on=['date', 'post_id'], how='left')
    # 填充没有数据的数值为0
    test_v1_day_final_df = test_v1_day_merged_df.fillna(0)
    # 输出为csv文件
    test_v1_day_final_df[['date', 'post_id', 'amount']].to_csv(task1_file, mode='w', header=True,
                                                                              index=False, sep=',',
                                                                                  encoding="utf-8")


# 提交数据集
def submission(train_feature, regressor, df, file1, file2, file3, file4):
    test_df = df.loc[((df['date'].dt.year == 2017) & (df['date'].dt.month == 7)
                      & (df['date'].dt.hour.isin([7, 14, 17])) & (
                              df['date'].dt.minute == 58))].copy()

    test_df['lagging5'] = test_df['lagging4']
    test_df['lagging4'] = test_df['lagging3']
    test_df['lagging3'] = test_df['lagging2']
    test_df['lagging2'] = test_df['lagging1']
    test_df['lagging1'] = test_df['amount']

    with open(file1, 'w'):
        pass
    with open(file2, 'w'):
        pass
    with open(file3, 'w'):
        pass
    with open(file4, 'w'):
        pass

    for i in range(30):
        test_X = test_df[train_feature]
        y_prediction = regressor.predict(test_X.values)

        test_df['lagging5'] = test_df['lagging4']
        test_df['lagging4'] = test_df['lagging3']
        test_df['lagging3'] = test_df['lagging2']
        test_df['lagging2'] = test_df['lagging1']
        test_df['lagging1'] = y_prediction

        test_df['predicted'] = np.expm1(y_prediction)
        test_df['datetime'] = test_df['datetime'] + pd.DateOffset(minutes=30)
        test_df['time_interval'] = test_df['datetime'].map(
            lambda x: '[' + str(x) + ',' + str(x + pd.DateOffset(minutes=30)) + ')')
        test_df.time_interval = test_df.time_interval.astype(object)
        # TODO:输出提交文件格式

        # if i < 7:
        #     test_df[['link_ID', 'date', 'time_interval', 'predicted']].to_csv(file1, mode='a', header=False,
        #                                                                       index=False,
        #                                                                       sep=';')
        # elif (7 <= i) and (i < 14):
        #     test_df[['link_ID', 'date', 'time_interval', 'predicted']].to_csv(file2, mode='a', header=False,
        #                                                                       index=False,
        #                                                                       sep=';')
        # elif (14 <= i) and (i < 22):
        #     test_df[['link_ID', 'date', 'time_interval', 'predicted']].to_csv(file3, mode='a', header=False,
        #                                                                       index=False,
        #                                                                       sep=';')
        # else:
        #     test_df[['link_ID', 'date', 'time_interval', 'predicted']].to_csv(file4, mode='a', header=False,
        #                                                                       index=False,
        #