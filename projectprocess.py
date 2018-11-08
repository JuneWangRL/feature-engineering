#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:21:41 2018
"""

import pandas as pd
from datetime import datetime

class process():
    ###返回当天产品的记录
    def getdaydata(self,trainUser,date):
        data_day=trainUser[trainUser["time_day"]==date]
        #hot_item_day=self.gethotitem(data_day,3000)
        #data_day=data_day[data_day.item_id.isin(hot_item_day)]
        return data_day
    
    ###返回当天的top_3000_hot_item
    def gethotitem(self,data_day,threshold):
        testdata_count=pd.DataFrame(data_day.groupby(['item_id'])['behavior_type'].count())
        testdata_count=testdata_count.sort_values('behavior_type',ascending=False)
        hot_item_day=testdata_count[:threshold]
        hot_item_day=hot_item_day[hot_item_day['behavior_type']>-1].index.tolist()
        return hot_item_day        
    
    def featureEngeering_label(self,data_day):
        ###date_day这一天里所有用户与hot_item的联系
        ###若user_id与item_id没有联系则将其置为0
        row=data_day["user_id"].unique()
        hot_item=self.gethotitem(data_day,threshold=3000)
        data_day=data_day[data_day.item_id.isin(hot_item)]
        data_day_3=pd.DataFrame(data_day,columns=["user_id","item_id","behavior_type"])
        records=[(x,y) for x in row for y in hot_item]
        df_records=pd.DataFrame(records,columns=["user_id","item_id"])
        res=pd.merge(df_records,data_day_3,how="left",on=["user_id","item_id"])
        res=res.fillna(0)
        res=res.rename(columns={"behavior_type":"label"}) 
        return res
    
    def featureEngeering_user(self,data_day):        
        ###构建第一个feature：用户浏览所有item的总分数
        ###totalScore是用户对所有item的打分
        view_time=pd.DataFrame(data_day.groupby(['user_id'])['behavior_type'].sum())  
        view_time["user_id"]=view_time.index
        view_time = view_time.rename(columns={'behavior_type': 'totalScore'})
        user_feature1=view_time
    
        ###构建第二个feature：用户对前3000热门产品的浏览情况
        ###ifView的特征
        top_3000=self.gethotitem(data_day,3000)
        grouped=data_day.groupby(['user_id'])["item_id"]
        dic={}
        for a,b in grouped:
            dic[a]=list(b)
        for key in dic:
            for x in dic[key]:
                if x in top_3000:
                    dic[key]=1
                    break
                dic[key]=0
        feature2=pd.DataFrame(dic,index=[0]).T
        feature2["user_id"]=feature2.index
        feature2=feature2.rename(columns={0: 'ifView_3000'})
        user_feature2=pd.merge(user_feature1,feature2,how="left",on=["user_id"])
        return user_feature2
    
    def gettop_item_score(self,data_day,topitem):
        hot_item=self.gethotitem(data_day,topitem)
        data_day=data_day[data_day.item_id.isin(hot_item)]
        grouped=pd.DataFrame(data_day.groupby(['user_id'])['behavior_type'].sum()) 
        grouped["user_id"]=grouped.index
        return grouped
    
    def featureEngeering_user2(self,user_feature2,data_day):
        ###构建特征3，4，5，用户对前1000，前2000，前3000商品的评分情况
        groped_1=self.gettop_item_score(data_day,1000)
        groped_1=groped_1.rename(columns={'behavior_type': 'allScore1000'})
        groped_2=self.gettop_item_score(data_day,2000)
        groped_2=groped_2.rename(columns={'behavior_type': 'allScore2000'})
        groped_3=self.gettop_item_score(data_day,3000)
        groped_3=groped_3.rename(columns={'behavior_type': 'allScore3000'})
        user_feature3=pd.merge(user_feature2,groped_1,how="left",on=["user_id"])
        user_feature4=pd.merge(user_feature3,groped_2,how="left",on=["user_id"])
        user_feature5=pd.merge(user_feature4,groped_3,how="left",on=["user_id"])
        return user_feature5


    def featureEngeering_item0(self,data_day):
        ###特征1:热门3000产品被打分的总和
        hot_item=self.gethotitem(data_day,3000)
        data_day=data_day[data_day.item_id.isin(hot_item)]
        feature1=pd.DataFrame(data_day.groupby(['item_id'])['behavior_type'].sum()) 
        feature1["item_id"]=feature1.index
        item_feature1=feature1.rename(columns={"behavior_type":"sumscore"}) 
        print("feature1 finished")
        return item_feature1


    def featureEngeering_item1(self,item_feature1,data_day):
        ###特征2:热门产品在该类别产品中打分占的比重
        hot_item=self.gethotitem(data_day,threshold=3000)
        category_score=pd.DataFrame(data_day.groupby(['item_category'])['behavior_type'].sum())  
        category_score["item_category"]=category_score.index

        data_day=data_day[data_day["item_id"].isin(hot_item)]
     
        item_score=pd.DataFrame(data_day.groupby(['item_id'])['behavior_type'].sum())
        item_score["item_id"]=item_score.index
        item_score["importance"]=""

        data_day1=pd.DataFrame(data_day,columns=["item_id","item_category"])
        data_day1=data_day1.drop_duplicates()
        item_score1=pd.merge(item_score,data_day1,how="left",on=["item_id"])
        item_score2=pd.merge(item_score1,category_score,how="left",on=["item_category"])
        item_score2["importance"]=item_score2["behavior_type_x"]/item_score2["behavior_type_y"]
        item_score2=pd.DataFrame(item_score2,columns=["item_id","importance"])
        item_feature2=pd.merge(item_feature1,item_score2,how="left",on=["item_id"])
        mid = item_feature2['item_id']
        item_feature2.drop(labels=['item_id'], axis=1,inplace = True)
        item_feature2.insert(0, 'item_id', mid)
        return item_feature2


    def featureEngeering_item2(self,item_feature2,data_day):
        ##特征3:item是否是前500的热门产品
        hot_item=self.gethotitem(data_day,3000)
        data_day=data_day[data_day["item_id"].isin(hot_item)]
        hot_item_500=self.gethotitem(data_day,500)
    
        if_hot_item=pd.DataFrame(data_day.groupby(['item_id'])['behavior_type'].sum())
        if_hot_item["if_top500"]=""
        
        if_hot_item["item_id"]=if_hot_item.index
    
        for i in range(len(if_hot_item)):
            if if_hot_item.iloc[i,2] in hot_item_500:
                if_hot_item.iloc[i,1]=1
            else:
                if_hot_item.iloc[i,1]=0
        item_feature3=pd.merge(item_feature2,if_hot_item,how="left",on=["item_id"])
        item_feature3.drop(labels=['behavior_type'], axis=1,inplace = True)
        return item_feature3


    def featureEngeering_item4(self,item_feature4,trainUser,time_span,date_exact):
        ###trainUser：总的data
        ##特征5:返回是前面5天热门产品的天数之和
        index=time_span.index(date_exact)
        dic={}
        for i in range(5):
            name='data_'+str(time_span[index-i-1])
            locals()[name]=self.getdaydata(trainUser,time_span[index-i-1])
            name2='hot_item'+str(time_span[index-i-1])
            locals()[name2]=self.gethotitem(locals()[name],3000)
            for i in range(len(locals()[name2])):
                dic[locals()[name2][i]]=dic.get(locals()[name2][i],0)+1
        list_temp=list(dic.keys())
        item_feature5=item_feature4
        item_feature5["past_day_counts"]=""
        for i in range(len(item_feature5)):
            if item_feature5.iloc[i,0] in list_temp:
                item_feature5.loc[i,"past_day_counts"]=dic[item_feature5.iloc[i,0]]
            else:
                item_feature5.loc[i,"past_day_counts"]=0    
        return item_feature5


    def getdaybefore(self,trainUser,date):
        before_data_day=trainUser[trainUser["time_day"] <= date]
        return before_data_day
    
    def featureEngeering_behavior_count(self,data_day):
        # 对用户的浏览，收藏，加购，购买行为进行计数
        behavior_count=pd.DataFrame(data_day.groupby(['user_id','item_id','behavior_type'])['time'].count())  
        
        behavior_count = behavior_count.unstack().fillna(0).astype(int)
        behavior_count.columns = ['click_num','favorite_num','add_num','buy_num']
        behavior_count['convert_ratio'] = behavior_count['buy_num'] / behavior_count['click_num']
        behavior_count = behavior_count.fillna(0)
        return behavior_count
    
    def featureEngeering_time_span(self,date_now,data_day):
        def time2stamp(cmnttime):   #转时间戳函数
            cmnttime = str(cmnttime)
            cmnttime=datetime.strptime(cmnttime,'%Y-%m-%d')
            stamp=int(datetime.timestamp(cmnttime))
            return stamp
        date_now_1 = time2stamp(date_now)
        # 用户最后一次操作到当前时间点的时间间隔
        time_span_now = pd.DataFrame(data_day.groupby(['user_id','item_id'])['time_day'].max())
        time_span_now.columns = ['behavior_interval']
        time_span_now['behavior_interval'] = time_span_now['behavior_interval'].apply(time2stamp)
        time_span_now['behavior_interval'] = time_span_now['behavior_interval'].apply(lambda x:int ((date_now_1 - x)/86400 + 1))
        return time_span_now
    
    def getSingleDay(self,userFeature,itemFeature,label,date):
        merge1=pd.merge(label,userFeature,on=["user_id"])
        merge2=pd.merge(merge1,itemFeature,on=["item_id"])
        merge2["time_day"]=date
        merge2["label_temp"]=merge2["label"]
        merge2=merge2.drop(['label'], axis=1)
        merge2.rename(columns={merge2.columns[-1]: "label" }, inplace=True)
        return merge2
    

    
    
    
    
    
    
    
''' 
modelData=trainUser[trainUser["time"]>"2014-12-15 23"]
##最后三天6952
unique_item_category=list(modelData['item_category'].unique())
unique_item=list(modelData['item_id'].unique())
print("当天商品个数：",len(unique_item))
##考虑是否刷单，是否存在单个item被用户浏览多次的情况
tmp=pd.DataFrame(modelData.groupby(['user_id','item_id'])['item_category'].count())
tmp=tmp.sort_values('item_category',ascending=False)
##根据结果，数据都很正常，不存在刷单情况

##get daily data.
def getdaydata(modelData,begin,end):
    tempdata=modelData[modelData["time"]>=begin]
    return tempdata[tempdata["time"]<=end]

#unique_item=list(data_18['item_id'].unique())
#print("当天商品个数：",len(unique_item))
#predict item
predict_item = pd.read_csv('/Users/junewang/Desktop/recommendation/tianchi_fresh_comp_train_item.csv',
                        keep_default_na=False)
print(predict_item.head())
predict_item_category=list(predict_item['item_category'].unique())

##需要预测的item_category有1054个，在modelData里面出现了731个
print(len(set(predict_item_category)&set(unique_item_category)))

#需要预测的产品里面产品分布极为不均匀，有的很多有的很少
item_distribution=pd.DataFrame(predict_item.groupby('item_category')['item_id'].count())
item_concentrate=len(pd.DataFrame(item_distribution[item_distribution["item_id"]>500]))/len(item_distribution)
print("商品集中度:",item_concentrate)

##对test数据集处理，item浏览次数做排序处理，找出最热门的产品信息

def gethotitem(data_day,threshold):
    testdata_count=pd.DataFrame(data_day.groupby(['item_id'])['behavior_type'].count())
    testdata_count=testdata_count.sort_values('behavior_type',ascending=False)
    hot_item_day=testdata_count[:threshold]
    hot_item_day=hot_item_day[hot_item_day['behavior_type']>-1].index.tolist()
    return hot_item_day

##80%的用户都会对热门的3000件产品进行浏览
def getuserrate(df,hot_item):
    hot_top=df[df.item_id.isin(hot_item)]['user_id'].unique()
    data_user=df["user_id"].unique()
    rate_top=len(hot_top)/len(data_user)
    return rate_top


###每天的商品类型有27万左右，但是百分之81%左右的用户行为都集中在前面3万条热门产品
   
##通过分析可以发现，top3万的热门产品分布在了>80%的用户上，如果把热门商品推荐正确了
##就可以获得很高分数    
    

特征太少，可以新建特征进行预测。
下一步为特征工程的构建工作

####做label值

def featureEngeering_label(data_day,hot_item):
    data_day=data_day[data_day.item_id.isin(hot_item)]
    row=data_day["user_id"].unique()
    data_day_3=pd.DataFrame(data_day,columns=["user_id","item_id","behavior_type"])
    records=[(x,y) for x in row for y in hot_item]
    df_records=pd.DataFrame(records,columns=["user_id","item_id"])
    res=pd.merge(df_records,data_day_3,how="left",on=["user_id","item_id"])
    res=res.fillna(0)
    res=res.rename(columns={"behavior_type":"label"}) 
    return res

#print("第18天商品的种类：",len(data_17["item_category"].unique()))

def featureEngeering_user(data_day):
    user_feature=pd.DataFrame(columns=["user_id"])
    users=data_day["user_id"].unique()
    user_feature["user_id"]=users
    #category=data_day["item_category"].unique()
    
    ###构建第一个feature：用户浏览所有item的总分数
    ###totalScore是用户对所有item的打分
    view_time=pd.DataFrame(data_day.groupby(['user_id'])['behavior_type'].sum())  
    view_time["user_id"]=view_time.index
    view_time = view_time.rename(columns={'behavior_type': 'totalScore'})
    user_feature1=pd.merge(user_feature,view_time,how="left",on=["user_id"])
    
    ###构建第二个feature：用户对热门产品的浏览情况
    ###ifView的特征
    top_300=gethotitem(data_day,300)
    grouped=data_day.groupby(['user_id'])["item_id"]
    dic={}
    for a,b in grouped:
        dic[a]=list(b)
    for key in dic:
        for x in dic[key]:
            if x in top_300:
                dic[key]=1
                break
            dic[key]=0
    feature2=pd.DataFrame(dic,index=[0]).T
    feature2["user_id"]=feature2.index
    feature2=feature2.rename(columns={0: 'ifView'})
    user_feature2=pd.merge(user_feature1,feature2,how="left",on=["user_id"])
    return user_feature2



def gettop_item_score(data_day,hot_item,topitem):
    hot_item=gethotitem(data_day,topitem)
    data_day=data_day[data_day.item_id.isin(hot_item)]
    grouped=pd.DataFrame(data_day.groupby(['user_id'])['behavior_type'].sum()) 
    grouped["user_id"]=grouped.index
    return grouped


def featureEngeering_user2(user_feature2,data_day,hot_item):
    ###构建特征3，4，5，用户对前1000，前3000，前5000商品的
    groped_1=gettop_item_score(data_day,hot_item,1000)
    groped_1=groped_1.rename(columns={'behavior_type': 'allScore1000'})
    groped_2=gettop_item_score(data_day,hot_item,3000)
    groped_2=groped_2.rename(columns={'behavior_type': 'allScore3000'})
    groped_3=gettop_item_score(data_day,hot_item,5000)
    groped_3=groped_3.rename(columns={'behavior_type': 'allScore5000'})
    user_feature3=pd.merge(user_feature2,groped_1,how="left",on=["user_id"])
    user_feature4=pd.merge(user_feature3,groped_2,how="left",on=["user_id"])
    user_feature5=pd.merge(user_feature4,groped_3,how="left",on=["user_id"])
    return user_feature5


def featureEngeering_item0(data_day):
    ###特征1:产品被打分的总和
    hot_item=gethotitem(data_day,threshold=30000)
    data_day=data_day[data_day["item_id"].isin(hot_item)]
    feature1=pd.DataFrame(data_day.groupby(['item_id'])['behavior_type'].sum()) 
    feature1["item_id"]=feature1.index
    item_feature1=feature1.rename(columns={"behavior_type":"sumscore"}) 
    print("feature1 finished")
    return item_feature1


def featureEngeering_item1(data_day,item_feature1):
    hot_item=gethotitem(data_day,threshold=30000)
    category_score=pd.DataFrame(data_day.groupby(['item_category'])['behavior_type'].sum())  
    category_score["item_category"]=category_score.index

    data_day=data_day[data_day["item_id"].isin(hot_item)]
     
    item_score=pd.DataFrame(data_day.groupby(['item_id'])['behavior_type'].sum())
    item_score["item_id"]=item_score.index
    item_score["importance"]=""

    data_day1=pd.DataFrame(data_day,columns=["item_id","item_category"])
    data_day1=data_day1.drop_duplicates()
    item_score1=pd.merge(item_score,data_day1,how="left",on=["item_id"])
    item_score2=pd.merge(item_score1,category_score,how="left",on=["item_category"])
    item_score2["importance"]=item_score2["behavior_type_x"]/item_score2["behavior_type_y"]
    item_score2=pd.DataFrame(item_score2,columns=["item_id","importance"])
    item_feature2=pd.merge(item_feature1,item_score2,how="left",on=["item_id"])
    mid = item_feature2['item_id']
    item_feature2.drop(labels=['item_id'], axis=1,inplace = True)
    item_feature2.insert(0, 'item_id', mid)
    return item_feature2


def featureEngeering_item2(item_feature2,data_day):
    ##特征3:item是否是前500的热门产品
    hot_item=gethotitem(data_day,30000)
    data_day=data_day[data_day["item_id"].isin(hot_item)]
    hot_item_500=gethotitem(data_day,500)
    
    if_hot_item=pd.DataFrame(data_day.groupby(['item_id'])['behavior_type'].sum())
    if_hot_item["if_top500"]=""
    if_hot_item["item_id"]=if_hot_item.index
    
    for i in range(len(if_hot_item)):
        if if_hot_item.iloc[i,2] in hot_item_500:
            if_hot_item.iloc[i,1]=1
        else:
            if_hot_item.iloc[i,1]=0
    item_feature3=pd.merge(item_feature2,if_hot_item,how="left",on=["item_id"])
    item_feature3.drop(labels=['behavior_type'], axis=1,inplace = True)
    return item_feature3


def featureEngeering_item3(item_feature3,data_day):
    ##特征4:是前500就对应它的打分
    hot_item=gethotitem(data_day,30000)
    data_day=data_day[data_day["item_id"].isin(hot_item)]
    if_hot_itemx=pd.DataFrame(data_day.groupby(['item_id'])['behavior_type'].count())
    if_hot_itemx=if_hot_itemx.rename(columns={"behavior_type":"top500Score"})
    if_hot_itemx["item_id"]=if_hot_itemx.index
    
    hot_item_500=gethotitem(data_day,500)
    for i in range(len(if_hot_itemx)):
        if if_hot_itemx.iloc[i,1] not in hot_item_500:
            if_hot_itemx.iloc[i,0]=None
            
    item_feature4=pd.merge(item_feature3,if_hot_itemx,how="left",on=["item_id"])
      
    return item_feature4

    
def featureEngeering_item4(item_feature4,data_next_day):
    ##特征5:是否为下一天的热门产品
    hot_item_next=gethotitem(data_next_day,30000)
    item_feature5=item_feature4
    item_feature5["if_next_day"]=""
    for i in range(len(item_feature4)):
        if item_feature5.iloc[i,1] in hot_item_next:
            item_feature5.loc[i,"if_next_day"]=1
        else:
            item_feature5.loc[i,"if_next_day"]=0    
    return item_feature5


rate_16_top=getuserrate(data_16,hot_item_16)
rate_17_top=getuserrate(data_17,hot_item_17)
rate_18_top=getuserrate(data_18,hot_item_18)
print("第16天浏览热门商品的用户比例：",rate_16_top)
print("第17天浏览热门商品的用户比例：",rate_17_top)
print("第18天浏览热门商品的用户比例：",rate_18_top)


def getdaytraindata(data_17,data_18):
    hot_item_17=gethotitem(data_17,3000)
    label=featureEngeering_label(data_17,hot_item_17)
    user_feature2=featureEngeering_user(data_17)    
    user_feature5=featureEngeering_user2(user_feature2,data_17,hot_item_17)
    item_feature1=featureEngeering_item0(data_17)
    item_feature2=featureEngeering_item1(data_17,item_feature1)
    item_feature3=featureEngeering_item2(item_feature2,data_17)
    item_feature4=featureEngeering_item3(item_feature3,data_17)
    item_feature5=featureEngeering_item4(item_feature4,data_18)
    featured_data1=pd.merge(label,user_feature5,how="left",on=["user_id"])
    featured_data2=pd.merge(featured_data1,item_feature5,how="left",on=["item_id"])
    train=featured_data2
    return train
    
data_16=getdaydata(modelData,"2014-12-15 23","2014-12-16 23")
data_17=getdaydata(modelData,"2014-12-16 23","2014-12-17 23")
data_18=getdaydata(modelData,"2014-12-17 23","2014-12-18 23")
train=getdaytraindata(data_16,data_17)
train.to_csv("train.csv")
print(train.info())

#### model部分
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',  # 多分类的问题
    'num_class': 10,               # 类别数，与 multisoftmax 并用
    'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 12,               # 构建树的深度，越大越容易过拟合
    'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,              # 随机采样训练样本
    'colsample_bytree': 0.7,       # 生成树时进行的列采样
    'min_child_weight': 3,
    'silent': 1,                   # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.007,                  # 如同学习率
    'seed': 1000,
    'nthread': 4,                  # cpu 线程数
}    

label=pd.DataFrame(train["label"])
trainx=pd.DataFrame(train,columns=["user_id","item_id","totalScore","ifView","allScore1000","allScore3000","allScore5000","sumscore","importance","if_top500","top500Score","if_next_day"])
lbl = preprocessing.LabelEncoder()
trainy = lbl.fit_transform(label)

dtrain = xgb.DMatrix(trainx, trainy)
num_rounds = 500
model = xgb.train(params, dtrain, num_rounds)

dtest=getdaytraindata(data_17,data_18)
dtest=pd.DataFrame(dtest,columns=["user_id","item_id","totalScore","ifView","allScore1000","allScore3000","allScore5000","sumscore","importance","if_top500","top500Score","if_next_day"])
pred = model.predict(dtest)


    def featureEngeering_item3(self,item_feature3,data_day):
        ##特征4:是前500就对应它的打分
        hot_item=self.gethotitem(data_day,3000)
        data_day=data_day[data_day["item_id"].isin(hot_item)]
        if_hot_itemx=pd.DataFrame(data_day.groupby(['item_id'])['behavior_type'].count())
        if_hot_itemx=if_hot_itemx.rename(columns={"behavior_type":"top500Score"})
        if_hot_itemx["item_id"]=if_hot_itemx.index
    
        hot_item_600=self.gethotitem(data_day,600)
        for i in range(len(if_hot_itemx)):
            if if_hot_itemx.iloc[i,1] not in hot_item_600:
                if_hot_itemx.iloc[i,0]=None
            
        item_feature4=pd.merge(item_feature3,if_hot_itemx,how="left",on=["item_id"])
      
        return item_feature4

'''  

    
            
    
    

    
                
                
    
    











 

    
    
    




