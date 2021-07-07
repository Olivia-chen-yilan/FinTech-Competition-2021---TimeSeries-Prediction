from preprocess import *
import xgboost as xgb
from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib
from sklearn.model_selection import ParameterGrid
from ultis import *
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def xgboost_submit(df, params):
    train_df = df.loc[df['date'] < pd.to_datetime('2020-12-01')]

    train_df = train_df.dropna()
    X = train_df[train_feature].values
    y = train_df['amount2'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    eval_set = [(X_test, y_test)]
    regressor = xgb.XGBRegressor(learning_rate=params['learning_rate'], n_estimators=params['n_estimators'],
                                 booster='gbtree', objective='reg:linear', n_jobs=-1, subsample=params['subsample'],
                                 colsample_bytree=params['colsample_bytree'], random_state=0,
                                 max_depth=params['max_depth'], gamma=params['gamma'],
                                 min_child_weight=params['min_child_weight'], reg_alpha=params['reg_alpha'])
    regressor.fit(X_train, y_train, verbose=True, early_stopping_rounds=10, eval_metric=mape_fintech,
                  eval_set=eval_set)
    # feature_vis(regressor, train_feature)  # 查看特征的重要性程度
    # joblib.dump(regressor, 'model/xgbr.pkl')
    # print('xgboost submit 模型：',regressor)
    df_test = pd.read_csv('data/test_v6_tt.csv', parse_dates=['date'])
    # print(df_test.head())
    fintech_submission(train_feature, regressor, df_test,
                       'target/submission_task1.txt', 'target/submission_task2.txt')


def fit_evaluate(df, df_test, params):
    # 去掉df中的缺失值
    df = df.dropna()
    # 取df中train_feature列的数值
    X = df[train_feature].values
    # 取df中amount作为y
    y = df['amount2'].values
    # 分割x和y为训练和测试数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    # 取df_test中valid_feature列的数值
    df_test = df_test[valid_feature].values
    valid_data = bucket_data(df_test)
    # 组合X_test, y_test为评价数据集
    eval_set = [(X_test, y_test)]
    # 建立XGBRegressor
    regressor = xgb.XGBRegressor(learning_rate=params['learning_rate'], n_estimators=params['n_estimators'],
                                 booster='gbtree', objective='reg:linear', n_jobs=-1, subsample=params['subsample'],
                                 colsample_bytree=params['colsample_bytree'], random_state=0,
                                 max_depth=params['max_depth'], gamma=params['gamma'],
                                 min_child_weight=params['min_child_weight'], reg_alpha=params['reg_alpha'])
    # 拟合训练数据集
    regressor.fit(X_train, y_train, verbose=False, early_stopping_rounds=10, eval_metric=mape_fintech,
                  eval_set=eval_set)

    # feature_vis(regressor, train_feature)
    return regressor, cross_valid(regressor, valid_data,
                                  lagging=lagging), regressor.best_iteration, regressor.best_score


def train(df, params, best, vis=False):
    # TODO:修改日期范围

    train1 = df.iloc[0:40660, :]
    train2 = df.iloc[40660:81320, :]
    train3 = df.iloc[81320:121980, :]
    train4 = df.iloc[121980:162640, :]
    train5 = df.iloc[162640:, :]

    # train1 = df.iloc[0:400, :]
    # train2 = df.iloc[400:800, :]
    # train3 = df.iloc[800:1200, :]
    # train4 = df.iloc[1200:1600, :]
    # train5 = df.iloc[1600:, :]
    # 使用不同的组合进行拟合，得到相应的结果
    print("CV strat")
    regressor, loss1, best_iteration1, best_score1 = fit_evaluate(pd.concat([train1, train2, train3, train4]), train5,
                                                                  params)
    print('regressor:', regressor, best_iteration1, best_score1, loss1)

    regressor, loss2, best_iteration2, best_score2 = fit_evaluate(pd.concat([train1, train2, train3, train5]), train4,
                                                                  params)
    print('regressor:', regressor, best_iteration2, best_score2, loss2)

    regressor, loss3, best_iteration3, best_score3 = fit_evaluate(pd.concat([train1, train2, train4, train5]), train3,
                                                                  params)
    print('regressor:', regressor, best_iteration3, best_score3, loss3)

    regressor, loss4, best_iteration4, best_score4 = fit_evaluate(pd.concat([train1, train3, train4, train5]), train2,
                                                                  params)
    print('regressor:', regressor, best_iteration4, best_score4, loss4)

    regressor, loss5, best_iteration5, best_score5 = fit_evaluate(pd.concat([train2, train3, train4, train5]), train1,
                                                                  params)
    print('regressor:', regressor, best_iteration5, best_score5, loss5)

    # 可视化结果
    if vis:
        xgb.plot_tree(regressor, num_trees=5)
        results = regressor.evals_result()
        epochs = len(results['validation_0']['rmse'])
        x_axis = range(0, epochs)
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
        ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
        ax.legend()
        plt.ylabel('rmse Loss')
        plt.ylim((0.2, 0.3))
        plt.show()
    # 记录训练结果
    loss = [loss1, loss2, loss3, loss4,loss5]
    params['loss_std'] = np.std(loss)
    params['loss'] = str(loss)
    params['mean_loss'] = np.mean(loss)
    params['n_estimators'] = str([best_iteration1, best_iteration2, best_iteration3, best_iteration4,best_iteration5])
    params['best_score'] = str([best_score1, best_score2, best_score3, best_score4,best_score5])
    print("params记录结束")
    print(str(params))
    print('loss mean:', np.mean(loss))
    # 寻找最佳的一次结果
    if np.mean(loss) <= best:
        best = np.mean(loss)
        print("best with: " + str(params))
        feature_vis(regressor, train_feature)
    return best


lagging = 5


# create_feature('data/train_v2.csv', 'data/train_v2_tt.csv')
# lagging_feature('data/train_v2_tt.csv', 'data/train_v3_tt.csv', lagging=5)
# samples_prepare('data/train_v3_tt.csv', 'data/train_v4_tt.csv')
# imputation_with_spline('data/train_v4_tt.csv', 'data/train_v5_tt.csv')


df = pd.read_csv('data/train_v5_tt.csv', parse_dates=['date'])
# df = pd.read_csv('data/train_v4_ttA.csv',  parse_dates=['date'])
lagging_feature = ['lagging%01d' % e for e in
                   range(lagging, 0, -1)]  # output:['lagging5', 'lagging4', 'lagging3', 'lagging2', 'lagging1']
base_feature = [x for x in df.columns.values.tolist() if
                x not in ['date', 'amount', 'day_of_week', 'WKD_TYP_CD', 'post_id', 'datetime', 'amount2'                   
                          'minute_series']]
base_feature = [x for x in base_feature if x not in lagging_feature]
train_feature = list(base_feature)
train_feature.extend(lagging_feature)
valid_feature = list(base_feature)
valid_feature.extend(['minute_series', 'amount2'])
print('train feature 长度：', len(train_feature))
print('vaild feature 长度：', len(valid_feature))
# print (train_feature)


# ----------------------------------------Train-------------------------------------------
# params_grid = {
#     'learning_rate': [0.05],
#     'n_estimators': [100],
#     'subsample': [0.6],
#     'colsample_bytree': [0.6],
#     'max_depth': [7],
#     'min_child_weight': [1],
#     'reg_alpha': [2],
#     'gamma': [0]
# }
#
# grid = ParameterGrid(params_grid)
# best = 1
#
# for params in grid:
#     best = train(df, params, best)

# ----------------------------------------submission-------------------------------------------
submit_params = {
    'learning_rate': 0.05,
    'n_estimators': 100,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'max_depth': 7,
    'min_child_weight': 1,
    'reg_alpha': 2,
    'gamma': 0
}
#
#
# # submit_params= {'colsample_bytree': 0.6,
# #                 'gamma': 0,
# #                 'learning_rate': 0.05,
# #                 'max_depth': 7,
# #                 'min_child_weight': 1,
# #                 'n_estimators': '[0, 0, 0, 0, 0]',
# #                 'reg_alpha': 2,
# #                 'subsample': 0.6,
# #                 'loss_std': 0.07808659373027632,
# #                 'loss': '[0.8478164489034056, 0.6704634645668051, 0.6680367606999092, 0.836363714255377, 0.7295597848372674]',
# #                 'mean_loss': 0.7504480346525528,
# #                 'best_score': '[6.418721, 7.739574, 8.086737, 6.387212, 7.947385]'}
xgboost_submit(df, submit_params)
