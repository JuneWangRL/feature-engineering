#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 12:12:55 2018

@author: junewang
"""

from projectprocess import process
import pandas as pd

trainUser = pd.read_csv('/Users/junewang/Desktop/recommendation/tianchi_fresh_comp_train_user.csv',
                        keep_default_na=False)
print(trainUser.head())
##选取最后三天的数据
##30天内item的数量： 4758484
##30天内user的数量： 20000
print("30天内item的数量：",len(trainUser["item_id"].unique()))
print("30天内user的数量：",len(trainUser["user_id"].unique()))
unique_user_id=trainUser["user_id"].unique()
unique_user_id_2000=unique_user_id[:1000]
trainUser_2000=trainUser[trainUser["user_id"].isin(unique_user_id_2000)]
trainUser_2000["time_day"] = trainUser_2000["time"].map(lambda x:x.split()[0]) 


process=process()
name=locals()
time_span=list(sorted(trainUser_2000["time_day"].unique()))
merge=pd.DataFrame()
for i in range(24):
    i=i+5
    print(time_span[i])
    data_day=process.getdaydata(trainUser_2000,time_span[i])
    #print(len(data_day["user_id"].unique()))
    #print(len(data_day["item_id"].unique()))
    #hot_item_day=process.gethotitem(data_day,3000)
    label=process.featureEngeering_label(data_day)
    user_feature2=process.featureEngeering_user(data_day)
    user_feature3=process.featureEngeering_user2(user_feature2,data_day)
    item_feature1=process.featureEngeering_item0(data_day)
    item_feature2=process.featureEngeering_item1(item_feature1,data_day)
    item_feature3=process.featureEngeering_item2(item_feature2,data_day)
    item_feature5=process.featureEngeering_item4(item_feature3,trainUser_2000,time_span,time_span[i])
    merged_data=process.getSingleDay(user_feature3,item_feature5,label,time_span[i])
    
    get_before_day = process.getdaybefore(trainUser_2000,time_span[i])
    behavior_interval = process.featureEngeering_time_span(time_span[i],get_before_day)
    behavior_count = process.featureEngeering_behavior_count(data_day)
    rs = behavior_count.merge(behavior_interval,on = ['user_id','item_id'])
    rs["index"]=rs.index
    rs[["user_id","item_id"]]=rs["index"].apply(pd.Series)
    rs.drop(labels=['index'], axis=1,inplace = True)
    
    merged=pd.merge(merged_data,rs,how="left",on=["user_id","item_id"])
    #name='data_'+str(time_span[i])[-2:]
    #locals()[name]=merged_data
    merge=pd.concat([merge,merged],axis=0)
    #merged_data.to_csv("/Users/junewang/Desktop/recommendation/data_"+str(time_span[i])+".csv")


merge.to_csv("traindata_v2.csv")

#merge=pd.concat([data_23,data_24],axis=0)



train = pd.read_csv('/Users/junewang/Desktop/recommendation/traindata_v2.csv',
                        keep_default_na=False)











