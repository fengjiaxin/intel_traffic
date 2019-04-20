#!/usr/bin/env python
# coding=utf-8
import warnings 
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os
import logging
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import linear_model
logging.basicConfig(level=logging.INFO)

'''
    author:fengjiaxin
    本文件主要用于预处理文件
'''


def cast_log_outliers(to_file):
    '''
        本function主要用于合并2017年3月份的数据 和2017年4，5，6月份的数据
            同时进行聚合，剪裁数据，只提取特定月份的数据，然后转换成新的数据
    '''
    if os.path.exists(to_file):
        logging.info(to_file+' 文件已经存在')
        return
    
    link_travel_time_part3_path = '../data/gy_link_travel_time_part3.txt'
    link_travel_time_part2_path = '../data/gy_link_travel_time_part2.txt'
    
    df_2 = pd.read_csv(link_travel_time_part2_path,delimiter=';',dtype={'linkID':object})
    df_2 = df_2.rename(columns={"linkID": "link_ID"})
    
    
    df_3 = pd.read_csv(link_travel_time_part3_path,delimiter=';',dtype={'link_ID':object})
    logging.info('load df_2 and df_3 success')
    
    df_3['time_interval_begin'] = pd.to_datetime(df_3['time_interval'].map(lambda x:x[1:20]))
    df_2['time_interval_begin'] = pd.to_datetime(df_2['time_interval'].map(lambda x:x[1:20]))
    
    # 截取2017年3月份的数据
    df_2 = df_2.loc[(df_2['time_interval_begin'] >= pd.to_datetime('2017-03-01'))
                  & (df_2['time_interval_begin'] <= pd.to_datetime('2017-03-31'))]
    
    # 合并数据
    df_3 = pd.concat([df_3,df_2])
    df_3 = df_3.drop(['time_interval'], axis=1)
    df_3['travel_time'] = np.log1p(df_3['travel_time'])
    
    # 剪裁数据
    def quantile_clip(group):
        group[group < group.quantile(0.05)] = group.quantile(0.05)
        group[group > group.quantile(0.95)] = group.quantile(0.95)
        return group
    
    df_3['travel_time'] = df_3.groupby(['link_ID', 'date'])['travel_time'].transform(quantile_clip)
    df_3 = df_3.loc[(df_3['time_interval_begin'].dt.hour.isin([6, 7, 8, 13, 14, 15, 16, 17, 18]))]
    
    df_3.to_csv(to_file, header=True, index=None, sep=';', mode='w')
    logging.info('to csv success')
    
    
def imputation_prepare(file,to_file):
    '''
        本函数主要是将路的数据和时间数据进行笛卡尔积，然后找出一些缺失数据和预测数据
            因为要预测的是2017年每月的第8，15，18小时，认为每个时段的前两个小时的数据比较重要
            只选择每天6，7，8，13，14，15，16，17，18小时，然后去掉2017年7月每天的第8，15，18小时
    '''
    if not os.path.exists(file):
        logging.info(file+' 文件不存在')
        return
    
    if os.path.exists(to_file):
        logging.info(to_file+' 文件已经存在')
        return
    
    link_info_path = '../data/gy_link_info.txt'
    
    df = pd.read_csv(file, delimiter=';', parse_dates=['time_interval_begin'], dtype={'link_ID': object})
    link_infos = pd.read_csv(link_info_path,delimiter=';',dtype={'link_ID':object})
    logging.info('load raw data,link infos success')
    
    date_range = pd.date_range("2017-03-01 00:00:00", "2017-07-31 23:58:00", freq='2min')
    new_index = pd.MultiIndex.from_product([link_infos['link_ID'].unique(), date_range],
                                           names=['link_ID', 'time_interval_begin'])

    new_df = pd.DataFrame(index=new_index).reset_index()
    
    df2 = pd.merge(new_df, df, on=['link_ID', 'time_interval_begin'], how='left')
    logging.info('merge data success')
    
    # 挑选出每天的6，7，8，13，14，15，16，17，18hour的数据
    df2 = df2.loc[(df2['time_interval_begin'].dt.hour.isin([6, 7, 8, 13, 14, 15, 16, 17, 18]))]
    
    # 去掉需要预测的数据
    df2 = df2.loc[~((df2['time_interval_begin'].dt.year == 2017) & (df2['time_interval_begin'].dt.month == 7) & (df2['time_interval_begin'].dt.hour.isin([8, 15, 18])))]
    df2['date'] = df2['time_interval_begin'].dt.strftime('%Y-%m-%d')
    
    df2.to_csv(to_file, header=True, index=None, sep=';', mode='w')
    logging.info('to csv success')
    
    
def imputation_with_spline(file,to_file):
    '''
        这个函数的功能是首先根据线性函数对季节趋势进行预测
            然后利用插值函数对天的趋势进行预测
            然后利用一些基本特征对缺失值进行预测
    '''
    if not os.path.exists(file):
        logging.info(file+' 文件不存在')
        return
    
    if os.path.exists(to_file):
        logging.info(to_file+' 文件已经存在')
        return
    
    df = pd.read_csv(file, delimiter=';', parse_dates=['time_interval_begin'], dtype={'link_ID': object})
    # 将travel_time存储起来
    df['travel_time2'] = df['travel_time']
    
    # 这个函数应用到groupby，可以对分组后的函数进行计算
    def date_trend(group):
        # 新建立一个tmp
        tmp = group.groupby('date_hour').mean().reset_index()

        # 其中z是一个函数
        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        
        y = tmp['travel_time'].values
        nans, x = nan_helper(y)
        # nans是一个true/false的矩阵

        regr = linear_model.LinearRegression()
        regr.fit(x(~nans).reshape(-1, 1), y[~nans].reshape(-1, 1))
        tmp['date_trend'] = regr.predict(tmp.index.values.reshape(-1, 1)).ravel()
            
        group = pd.merge(group, tmp[['date_trend', 'date_hour']], on='date_hour', how='left')
   
        return group
    
    # 季节性趋势，以每小时进行度量
    df['date_hour'] = df.time_interval_begin.map(lambda x: x.strftime('%Y-%m-%d-%H'))
    df = df.groupby('link_ID').apply(date_trend)
    
    # 删除两列
    df = df.drop(['date_hour', 'link_ID'], axis=1)
    df = df.reset_index()
    # 把level_1的index删除
    df = df.drop('level_1', axis=1)
    # travel_time 为实际的平均旅行时间和季节趋势的残差
    df['travel_time'] = df['travel_time'] - df['date_trend']
    
    # 每天的趋势，以miunute区分
    def minute_trend(group):
        tmp = group.groupby('hour_minute').mean().reset_index()
        spl = UnivariateSpline(tmp.index, tmp['travel_time'].values, s=0.5, k=3)
        tmp['minute_trend'] = spl(tmp.index)
        group = pd.merge(group, tmp[['minute_trend', 'hour_minute']], on='hour_minute', how='left')
        return group

    df['hour_minute'] = df.time_interval_begin.map(lambda x: x.strftime('%H-%M'))
    df = df.groupby('link_ID').apply(minute_trend)

    df = df.drop(['hour_minute', 'link_ID'], axis=1)
    df = df.reset_index()
    df = df.drop('level_1', axis=1)
    
    # 此时的travel_time 为实际的travel_time - date_trend - minute_trend 
    df['travel_time'] = df['travel_time'] - df['minute_trend']
    
    logging.info('handle day minute trend success')
    
    # 提取道路相关的特征
    link_info_path = '../data/gy_link_info.txt'
    link_top_path = '../data/gy_link_top.txt'

    # 道路的基本信息
    link_infos = pd.read_csv(link_info_path,delimiter=';',dtype={'link_ID':object})

    # 每条路的上下游关系
    link_tops = pd.read_csv(link_top_path,delimiter=';',dtype={'link_ID':object})
    
    # 首先计算道路的基本信息
    # 首先定义计算上下游数量的函数

    def cal_link_nums(x):
        if pd.isnull(x):
            return 0
        else:
            return len(x.split('#'))    
    link_tops['in_links_num'] = link_tops['in_links'].apply(cal_link_nums)
    link_tops['out_links_num'] = link_tops['out_links'].apply(cal_link_nums)
    link_infos = pd.merge(link_infos,link_tops,on=['link_ID'],how = 'left')
    # 把上下游的数量组合作为一个特征

    link_infos['links_in_out_num'] = link_infos["in_links_num"].astype('str') + "," + link_infos["out_links_num"].astype('str')
    
    # 计算道路的面积
    link_infos['area'] = link_infos['length'] * link_infos['width']
    
    logging.info('handle link_infos success')
    
    df = pd.merge(df, link_infos[['link_ID', 'length', 'width', 'links_in_out_num', 'area','in_links_num','out_links_num']], on=['link_ID'], how='left')
    
    # 接下来提取一些基本的时间特征
    # vacation
    df.loc[df['date'].isin(
        ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
         '2017-05-28', '2017-05-29', '2017-05-30']), 'vacation'] = 1

    df.loc[~df['date'].isin(
        ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
         '2017-05-28', '2017-05-29', '2017-05-30']), 'vacation'] = 0
    
    df['minute'] = df['time_interval_begin'].dt.minute
    df['hour'] = df['time_interval_begin'].dt.hour
    df['day'] = df['time_interval_begin'].dt.day
    df['week_day'] = df['time_interval_begin'].map(lambda x: x.weekday() + 1)
    df['month'] = df['time_interval_begin'].dt.month
    
    # link_ID也应该作为一个特征，但是132维的one-hot过于稀疏，将其转换成数字，使得数字大小的含义对travel_time有一个相关性
    def mean_time(group):
        group['link_ID_en'] = group['travel_time'].mean()
        return group

    df = df.groupby('link_ID').apply(mean_time)
    sorted_link = np.sort(df['link_ID_en'].unique())
    df['link_ID_en'] = df['link_ID_en'].map(lambda x: np.argmin(x >= sorted_link))
    
    
    '''
        经过前两个trend的相减变换，现在的travel_time基本上在0附近波动，均值为0, 我们还可以把算出每条路的travel_time的标准差 df['travel_time_std']，用df['travel_time'] = df['travel_time'] / df['travel_time_std']来b标准化每条路的travel_time的方差为1，这种均值为0，方差为1的数据分布对于模型来说是比较理想的状态（特别是深度学习）
    '''
    def std(group):
        group['travel_time_std'] = np.std(group['travel_time'])
        return group

    df = df.groupby('link_ID').apply(std)
    df['travel_time'] = df['travel_time'] / df['travel_time_std']
    
    logging.info('handle base time feature success')
    
    # # 构建模型并训练
    params = {


        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 63,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'bagging_seed':0,
        'bagging_freq': 1,
        'verbose': 1,
        'reg_alpha':1,
        'reg_lambda':2,
        'min_child_weight':6
    }
    
    df = pd.get_dummies(df, columns=['links_num', 'width', 'minute', 'hour', 'week_day', 'day', 'month'])
    
    feature = df.columns.values.tolist()
    train_feature = [x for x in feature if
                     x not in ['link_ID', 'time_interval_begin', 'travel_time', 'date', 'travel_time2', 'minute_trend','travel_time_std', 'date_trend']]

    train_df = df.loc[~df['travel_time'].isnull()]
    test_df = df.loc[df['travel_time'].isnull()].copy()
    
    X = train_df[train_feature].values
    y = train_df['travel_time'].values
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    
    # 构建lgbs数据
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_evals = lgb.Dataset(X_test, y_test , reference=lgb_train)
    
    logging.info('begin train lgb')
    # 通过验证集调参（超参数），进行模型选择
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=[lgb_train,lgb_evals],
                    valid_names=['train','valid'],
                    early_stopping_rounds=20,
                    verbose_eval=100,
                    )
    
    
    logging.info('train success')
    logging.info('begin predict')
    test_df['prediction'] = gbm.predict(test_df[train_feature].values)
    
    df = pd.merge(df, test_df[['link_ID', 'time_interval_begin', 'prediction']], on=['link_ID', 'time_interval_begin'],
                  how='left')

    # 判断travel_time是否是预测出来的
    df['imputation1'] = df['travel_time'].isnull()
    df['travel_time'] = df['travel_time'].fillna(value=df['prediction'])
    df['travel_time'] = (df['travel_time'] * np.array(df['travel_time_std']) + np.array(df['minute_trend'])
                         + np.array(df['date_trend']))

    df[['link_ID', 'date', 'time_interval_begin', 'travel_time', 'imputation1']].to_csv(to_file, header=True,index=None,sep=';', mode='w')
    logging.info(to_file+ 'save success')
    logging.info('end imputation_with_spline')
    
    
def create_lagging(df, df_original, i):
    df1 = df_original.copy()
    df1['time_interval_begin'] = df1['time_interval_begin'] + pd.DateOffset(minutes=i * 2)
    df1 = df1.rename(columns={'travel_time': 'lagging' + str(i)})
    df2 = pd.merge(df, df1[['link_ID', 'time_interval_begin', 'lagging' + str(i)]],
                   on=['link_ID', 'time_interval_begin'],
                   how='left')
    return df2
    
def create_feature(file,to_file,lagging = 5):
    '''
        这个函数的功能是提取基本数据特征，包括基本时间特征，基本路况特征 ，基本lagging特征
    '''
    
    if not os.path.exists(file):
        logging.info(file+' 文件不存在')
        return
    
    if os.path.exists(to_file):
        logging.info(to_file+' 文件已经存在')
        return
    
    df = pd.read_csv(file, delimiter=';', parse_dates=['time_interval_begin'], dtype={'link_ID': object})

    # you can check imputation by uncomment the following:
    # vis_imputation(df)

    # lagging feature
    df1 = create_lagging(df, df, 1)
    for i in range(2, lagging + 1):
        df1 = create_lagging(df1, df, i)

    logging.info('create lagging feature success')
    # length, width feature
    link_info_path = '../data/gy_link_info.txt'
    link_top_path = '../data/gy_link_top.txt'

    # 道路的基本信息
    link_infos = pd.read_csv(link_info_path,delimiter=';',dtype={'link_ID':object})

    # 每条路的上下游关系
    link_tops = pd.read_csv(link_top_path,delimiter=';',dtype={'link_ID':object})
    
    
    def cal_link_nums(x):
        if pd.isnull(x):
            return 0
        else:
            return len(x.split('#'))    
    link_tops['in_links'] = link_tops['in_links'].apply(cal_link_nums)
    link_tops['out_links'] = link_tops['out_links'].apply(cal_link_nums)
    

    link_infos = pd.merge(link_infos, link_tops, on=['link_ID'], how='left')
    
    link_infos['links_num'] = link_infos["in_links"].astype('str') + "," + link_infos["out_links"].astype('str')
    
    link_infos['area'] = link_infos['length'] * link_infos['width']
    
    logging.info('handle link infos feature success')
    
    df2 = pd.merge(df1, link_infos[['link_ID', 'length', 'width', 'links_num', 'area']], on=['link_ID'], how='left')
    
    # df.boxplot(by=['width'], column='travel_time')
    # plt.show()
    # df.boxplot(by=['length'], column='travel_time')
    # plt.show()

    # links_num feature
    df2.loc[df2['links_num'].isin(['0.0,2.0', '2.0,0.0', '1.0,0.0']), 'links_num'] = 'other'
    # df.boxplot(by=['links_num'], column='travel_time')
    # plt.show()

    # vacation feature
    df2.loc[df2['date'].isin(
        ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
         '2017-05-28', '2017-05-29', '2017-05-30']), 'vacation'] = 1
    df2.loc[~df2['date'].isin(
        ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
         '2017-05-28', '2017-05-29', '2017-05-30']), 'vacation'] = 0

    # minute_series for CV
    df2.loc[df2['time_interval_begin'].dt.hour.isin([6, 7, 8]), 'minute_series'] = \
        df2['time_interval_begin'].dt.minute + (df2['time_interval_begin'].dt.hour - 6) * 60

    df2.loc[df2['time_interval_begin'].dt.hour.isin([13, 14, 15]), 'minute_series'] = \
        df2['time_interval_begin'].dt.minute + (df2['time_interval_begin'].dt.hour - 13) * 60

    df2.loc[df2['time_interval_begin'].dt.hour.isin([16, 17, 18]), 'minute_series'] = \
        df2['time_interval_begin'].dt.minute + (df2['time_interval_begin'].dt.hour - 16) * 60

    # day_of_week_en feature
    df2['day_of_week'] = df2['time_interval_begin'].map(lambda x: x.weekday() + 1)
    df2.loc[df2['day_of_week'].isin([1, 2, 3]), 'day_of_week_en'] = 1
    df2.loc[df2['day_of_week'].isin([4, 5]), 'day_of_week_en'] = 2
    df2.loc[df2['day_of_week'].isin([6, 7]), 'day_of_week_en'] = 3

    # hour_en feature
    df2.loc[df['time_interval_begin'].dt.hour.isin([6, 7, 8]), 'hour_en'] = 1
    df2.loc[df['time_interval_begin'].dt.hour.isin([13, 14, 15]), 'hour_en'] = 2
    df2.loc[df['time_interval_begin'].dt.hour.isin([16, 17, 18]), 'hour_en'] = 3

    # week_hour feature
    df2['week_hour'] = df2["day_of_week_en"].astype('str') + "," + df2["hour_en"].astype('str')

    # df2.boxplot(by=['week_hour'], column='travel_time')
    # plt.show()

    df2 = pd.get_dummies(df2, columns=['week_hour', 'links_num', 'width'])

    # ID Label Encode
    def mean_time(group):
        group['link_ID_en'] = group['travel_time'].mean()
        return group

    df2 = df2.groupby('link_ID').apply(mean_time)
    sorted_link = np.sort(df2['link_ID_en'].unique())
    df2['link_ID_en'] = df2['link_ID_en'].map(lambda x: np.argmin(x >= sorted_link))
    # df.boxplot(by=['link_ID_en'], column='travel_time')
    # plt.show()
    
    logging.info('handle base time feature success')


    df2.to_csv(to_file, header=True, index=None, sep=';', mode='w')
    logging.info(to_file + ' save success')
    
if __name__ == '__main__':
    cast_log_outliers('../data/raw_data.csv')
    imputation_prepare('../data/raw_data.csv','../data/pre_training.csv')
    imputation_with_spline('../data/pre_training.csv', '../data/com_training.txt')
    create_feature('../data/com_training.txt', '../data/training.txt', lagging=5)