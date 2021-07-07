import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.interpolate import UnivariateSpline
from sklearn import linear_model
import xgboost as xgb
# from ultis import *

def expasion_test(file,to_file):
    df = pd.read_csv(file, parse_dates=['date'])
    df_A = df[df['post_id'] == 'A']
    df_B = df[df['post_id'] == 'B']
    df_B = df_B.reset_index()

    result = pd.DataFrame(columns=['date', 'post_id', 'periods', 'biz_type', 'amount'])
    for i in range(len(df_A)):
        result = result.append(pd.DataFrame({'date': [df_A.iloc[i]['date']] * 13,
                                             'post_id': [df_A.iloc[i]['post_id']] * 13,
                                             'biz_type': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
                                                          'A11', 'A12', 'A13'],
                                             'periods': [df_A.iloc[i]['periods']] * 13,
                                             'amount': [df_A.iloc[i]['amount']] * 13}), ignore_index=True)

    for i in range(len(df_B)):
        result = result.append(pd.DataFrame({'date': [df_B.iloc[i]['date']],
                                             'post_id': [df_B.iloc[i]['post_id']],
                                             'biz_type': ['B1'],
                                             'periods': [df_B.iloc[i]['periods']],
                                             'amount': [df_B.iloc[i]['amount']]}), ignore_index=True)
    result.to_csv(to_file, header=True, index=False, mode='w')

def create_feature(file, to_file):
    train_data_df = pd.read_csv(file, parse_dates=['date'])


    # 日期基本特征
    train_data_df['minute'] = train_data_df['periods'].map(lambda x: 30 if x % 2 == 0 else 0)
    train_data_df['hour'] = train_data_df['periods'].map(lambda x: int((x-1)/2))
    train_data_df['day_of_week'] = train_data_df['date'].map(lambda x: x.weekday() + 1)
    train_data_df['day'] = train_data_df['date'].dt.day
    train_data_df['month'] = train_data_df['date'].dt.month
    train_data_df['year'] = train_data_df['date'].dt.year

    # 假期特征
    date_infos = pd.read_csv('data/wkd_v1.csv', parse_dates=['ORIG_DT'])
    date_infos.rename(columns={'ORIG_DT':'date'},inplace = True)
    train_data_withvacation_df = pd.merge(train_data_df, date_infos, on='date', how='left')
    train_data_withvacation_df['WKD_TYP_CD_en']=train_data_withvacation_df['WKD_TYP_CD'].map({'WN':1,'SN': 0, 'NH': 0, 'SS': 0, 'WS': 1})

    # minute_series for CV
    train_data_withvacation_df['minute_series']= (train_data_withvacation_df['periods']-18) *30

    # df2.loc[df2['time_interval_begin'].dt.hour.isin([6, 7, 8]), 'minute_series'] = \
    #     df2['time_interval_begin'].dt.minute + (df2['time_interval_begin'].dt.hour - 6) * 60
    #
    # df2.loc[df2['time_interval_begin'].dt.hour.isin([13, 14, 15]), 'minute_series'] = \
    #     df2['time_interval_begin'].dt.minute + (df2['time_interval_begin'].dt.hour - 13) * 60
    #
    # df2.loc[df2['time_interval_begin'].dt.hour.isin([16, 17, 18]), 'minute_series'] = \
    #     df2['time_interval_begin'].dt.minute + (df2['time_interval_begin'].dt.hour - 16) * 60

    # 星期几的特征
    # day_of_week_en feature
    train_data_withvacation_df.loc[train_data_withvacation_df['day_of_week'].isin([1, 2, 3]), 'day_of_week_en'] = 1
    train_data_withvacation_df.loc[train_data_withvacation_df['day_of_week'].isin([4, 5]), 'day_of_week_en'] = 2
    train_data_withvacation_df.loc[train_data_withvacation_df['day_of_week'].isin([6, 7]), 'day_of_week_en'] = 3

    # 是否是工作时间 （8：30-18：30）
    # 工作时间：1
    # 非工作时间：0
    train_data_withvacation_df.loc[train_data_withvacation_df['periods'].isin(range(18, 38)), 'workhour_en'] = 1
    train_data_withvacation_df.loc[~train_data_withvacation_df['periods'].isin(range(18, 38)), 'workhour_en'] = 0

    # 上午下午时间
    # 上午工作时间：1
    # 下午工作时间： 2
    # 非工作时间：0
    # TODO: 转换为哑变量
    train_data_withvacation_df.loc[train_data_withvacation_df['periods'].isin(range(0, 18)), 'morningafternoon_en'] = 0
    train_data_withvacation_df.loc[train_data_withvacation_df['periods'].isin(range(38, 49)), 'morningafternoon_en'] = 0
    train_data_withvacation_df.loc[train_data_withvacation_df['periods'].isin(range(18, 28)), 'morningafternoon_en'] = 1
    train_data_withvacation_df.loc[train_data_withvacation_df['periods'].isin(range(28, 38)), 'morningafternoon_en'] = 2
    train_data_withvacation_df['morningafternoon_en'] = pd.get_dummies(train_data_withvacation_df['morningafternoon_en'])

    # 星期几和工作时间的组合特征week_hour feature
    train_data_withvacation_df['week_hour'] = train_data_withvacation_df["day_of_week"].astype('str') + "," + train_data_withvacation_df["periods"].astype('str')
    train_data_withvacation_df['week_hour'] = pd.get_dummies(train_data_withvacation_df['week_hour'])

    # df2.boxplot(by=['week_hour'], column='amount')
    # plt.show()
    # TODO: 去掉注释
    # train_data_withvacation_df = pd.get_dummies(train_data_withvacation_df, columns=['week_hour'])


    # print(train_data_withvacation_df.head(20))

    train_data_withvacation_df.to_csv(to_file, header=True, index=None, mode='w')

def create_lagging(df, df_original, i):
    # df1是原始数据的副本
    df1 = df_original.copy()
    # 对df1中的datetime向后推迟半小时，也就是推迟一个时间段
    df1['datetime'] = df1['datetime'] + pd.DateOffset(minutes=i * 30)
    # 重命名amount为laggingi
    df1 = df1.rename(columns={'amount': 'lagging' + str(i)})
    # 将df和df1进行合并，按照岗位类型和datetime进行合并
    df2 = pd.merge(df, df1[['biz_type', 'datetime', 'lagging' + str(i)]],
                   on=['biz_type', 'datetime'],
                   how='left')
    return df2

def lagging_feature(file, to_file,lagging=5):
    train_data_df = pd.read_csv(file, parse_dates=['date'])
    # datetime中存储的是日期和时间，目的是为了转换为时间类型后进行时间的错后
    train_data_df['datetime'] = train_data_df['year'].map(str)+'-'+train_data_df['month'].map(str)+'-'+train_data_df['day'].map(str)+' '+train_data_df['hour'].map(str)+':'+train_data_df['minute'].map(str)
    train_data_df['datetime'] = pd.to_datetime(train_data_df['datetime'])

    # lagging feature
    # 进行一阶滞后
    df1 = create_lagging(train_data_df, train_data_df, 1)
    # 进行多阶滞后
    for i in range(2, lagging + 1):
        df1 = create_lagging(df1, train_data_df, i)
    df1.to_csv(to_file, header=True, index=None, mode='w')

def samples_prepare(file, to_file):
    df = pd.read_csv(file, parse_dates=['date'])
    # 只筛选工作日的数据
    #TODO：对于AB岗位只筛选工作日的数据
    df = df[df['WKD_TYP_CD_en'] == 1]

    # 只筛选8点半到18点半的数据
    # df_B = df_B.loc[df_B['periods'].isin(range(18, 39))]
    # 到18点半是18-37（含37）
    df = df.loc[df['periods'].isin(range(18, 38))]
    df.to_csv(to_file, header=True, index=False, mode='w')

def get_week_day(date):
  day = date.weekday()
  return day+1


def periods_trend(group):
    df = pd.read_csv('/Users/oliviachen/PycharmProjects/TimeSeriesPrediction-master-20210510-v1 1/data/train_v4_tt.csv',
                     parse_dates=['date'])
    df = df.groupby('periods').mean().reset_index()
    tmp = group.groupby('periods').mean().reset_index()

    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    y = df['amount'].values

    nans, xxxx = nan_helper(y)

    regr = linear_model.LinearRegression()
    regr.fit(xxxx(~nans).reshape(-1, 1), y)
    tmp['periods_trend'] = regr.predict(tmp.index.values.reshape(-1, 1)).ravel()
    group = pd.merge(group, tmp[['periods_trend', 'periods']], on='periods', how='left')

    # plt.plot(tmp.index, tmp['year_trend'], 'o', tmp.index, tmp['amount'], 'ro')
    # plt.title(group.biz_type.values[0])
    # plt.show()
    return group

def day_trend(group):
    df = pd.read_csv('/Users/oliviachen/PycharmProjects/TimeSeriesPrediction-master-20210510-v1 1/data/train_v4_tt.csv',
                     parse_dates=['date'])
    df = df.groupby('day').mean().reset_index()
    tmp = group.groupby('day').mean().reset_index()

    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    y = df['amount'].values

    nans, xxxx = nan_helper(y)

    regr = linear_model.LinearRegression()
    regr.fit(xxxx(~nans).reshape(-1, 1), y)
    tmp['day_trend'] = regr.predict(tmp.index.values.reshape(-1, 1)).ravel()
    group = pd.merge(group, tmp[['day_trend','day']], on='day', how='left')
    # plt.plot(tmp.index, tmp['day_trend'], 'o', tmp.index, tmp['amount'], 'ro')
    # plt.title(group.biz_type.values[0])
    # plt.show()
    return group

def weekday_trend(group):
    df = pd.read_csv('/Users/oliviachen/PycharmProjects/TimeSeriesPrediction-master-20210510-v1 1/data/train_v4_tt.csv',
                     parse_dates=['date'])
    df = df.groupby('day_of_week').mean().reset_index()
    tmp = group.groupby('day_of_week').mean().reset_index()

    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    y = df['amount'].values

    nans, xxxx = nan_helper(y)
    # if group.link_ID.values[0] in ['3377906282328510514', '3377906283328510514', '4377906280784800514',
    #                                '9377906281555510514']:
    #     tmp['date_trend'] = group['travel_time'].median()
    # else:
    regr = linear_model.LinearRegression()
    regr.fit(xxxx(~nans).reshape(-1, 1), y)
    tmp['day_of_week_trend'] = regr.predict(tmp.index.values.reshape(-1, 1)).ravel()
    group = pd.merge(group, tmp[['day_of_week_trend','day_of_week']], on='day_of_week', how='left')
    # plt.plot(tmp.index, tmp['day_of_week_trend'], 'o', tmp.index, tmp['amount'], 'ro')
    # plt.title(group.biz_type.values[0])
    # plt.show()
    return group

def month_trend(group):
    df = pd.read_csv('/Users/oliviachen/PycharmProjects/TimeSeriesPrediction-master-20210510-v1 1/data/train_v4_tt.csv',
                     parse_dates=['date'])
    df = df.groupby('month').mean().reset_index()
    tmp = group.groupby('month').mean().reset_index()

    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    y = df['amount'].values

    nans, xxxx = nan_helper(y)

    regr = linear_model.LinearRegression()
    regr.fit(xxxx(~nans).reshape(-1, 1), y)
    tmp['month_trend'] = regr.predict(tmp.index.values.reshape(-1, 1)).ravel()
    group = pd.merge(group, tmp[['month_trend','month']], on='month', how='left')
    # plt.plot(tmp.index, tmp['month_trend'], 'o', tmp.index, tmp['amount'], 'ro')
    # plt.title(group.biz_type.values[0])
    # plt.show()
    return group

def year_trend(group):
    df = pd.read_csv('/Users/oliviachen/PycharmProjects/TimeSeriesPrediction-master-20210510-v1 1/data/train_v4_tt.csv',
                     parse_dates=['date'])
    df = df.groupby('year').mean().reset_index()
    tmp = group.groupby('year').mean().reset_index()

    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    y = df['amount'].values

    nans, xxxx = nan_helper(y)

    regr = linear_model.LinearRegression()
    regr.fit(xxxx(~nans).reshape(-1, 1), y)
    tmp['year_trend'] = regr.predict(tmp.index.values.reshape(-1, 1)).ravel()
    group = pd.merge(group, tmp[['year_trend','year']], on='year', how='left')
    # plt.plot(tmp.index, tmp['year_trend'], 'o', tmp.index, tmp['amount'], 'ro')
    # plt.title(group.biz_type.values[0])
    # plt.show()
    return group

def imputation_with_spline(file, to_file):
    data_df = pd.read_csv(file, parse_dates=['date'])
    data_df['amount2'] = data_df['amount']

    # 处理年度效应
    data_df = data_df.groupby('biz_type').apply(year_trend)
    data_df = data_df.drop([ 'biz_type'], axis=1)
    data_df = data_df.reset_index()
    data_df = data_df.drop('level_1', axis=1)
    # data_df['amount'] = data_df['amount'] - data_df['year_trend']

    # 处理月度效应
    data_df = data_df.groupby('biz_type').apply(month_trend)
    data_df = data_df.drop([ 'biz_type'], axis=1)
    data_df = data_df.reset_index()
    data_df = data_df.drop('level_1', axis=1)
    # data_df['amount'] = data_df['amount'] - data_df['month_trend']

    # 处理天效应
    data_df = data_df.groupby('biz_type').apply(day_trend)
    data_df = data_df.drop([ 'biz_type'], axis=1)
    data_df = data_df.reset_index()
    data_df = data_df.drop('level_1', axis=1)
    # data_df['amount'] = data_df['amount'] - data_df['day_trend']

    # 处理星期几效应
    data_df = data_df.groupby('biz_type').apply(weekday_trend)
    data_df = data_df.drop([ 'biz_type'], axis=1)
    data_df = data_df.reset_index()
    data_df = data_df.drop('level_1', axis=1)
    # data_df['amount'] = data_df['amount'] - data_df['day_of_week_trend']

    # 处理时间段效应
    data_df = data_df.groupby('biz_type').apply(periods_trend)
    data_df = data_df.drop([ 'biz_type'], axis=1)
    data_df = data_df.reset_index()
    data_df = data_df.drop('level_1', axis=1)
    # data_df['amount'] = data_df['amount'] - data_df['periods_trend']

    data_df['biz_type'] =  data_df['biz_type'].map(
        {'A1': 0, 'A2': 1, 'A3': 2, 'A4': 3, 'A5': 4,'A6':5,'A7':6,'A8':7,'A9':8,'A10':9,'A11':10,'A12':11,'A13':12,'B1':13})

    # print(data_df.head(20))

    data_df.to_csv(to_file, header=True,index=False, mode='w')

if __name__ == '__main__':
    expasion_test('data/test_v2_periods.csv', 'data/test_v2_tt.csv')
    create_feature('data/test_v2_tt.csv','data/test_v3_tt.csv')
    lagging_feature('data/test_v3_tt.csv','data/test_v4_tt.csv')
    samples_prepare('data/test_v4_tt.csv','data/test_v5_tt.csv')
    imputation_with_spline('data/test_v5_tt.csv','data/test_v6_tt.csv')
