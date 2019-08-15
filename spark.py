#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 01:07:09 2019

@author: Bing
"""

import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType, LongType, DateType, UserDefinedType
from pyspark.sql.functions import struct
from pyspark.sql import functions as F
import networkx as nx
from graphrole import RecursiveFeatureExtractor, RoleExtractor
from datetime import datetime
from datetime import timedelta
from pyspark.sql.functions import udf, pandas_udf
import pandas as pd
import numpy as np
import math
import itertools
from numpy import linalg as LA

schema = StructType([\
        StructField("model", StringType()),\
        StructField("date", DateType()), \
        StructField("wSize", IntegerType()),\
        StructField("alpha", DoubleType()),\
        StructField("srcid", LongType()),\
        StructField("dstid", LongType())\
    ])

def getRolxFromMysql(s:str, e: str):
    mydb = mysql.connector.connect(user='brao',password='H1_Cont123',host='10.190.128.50',database='borders2')
    mycursor = mydb.cursor()
    sql = f"SELECT User1, User2 From link where CreatedDate <= '{s}' and CreatedDate >= '{e}'"
    mycursor.execute(sql)
    myresult = mycursor.fetchall()
    graph = nx.from_edgelist(myresult)
    # extract features
    feature_extractor = RecursiveFeatureExtractor(graph) # graphrole.features.extract.RecursiveFeatureExtractor
    features = feature_extractor.extract_features() #pandas.core.frame.DataFrame
    # assign node roles
    role_extractor = RoleExtractor(n_roles=10) # graphrole.roles.extract.RoleExtractor
    role_extractor.extract_role_factors(features)
    #node_roles = role_extractor.roles # dict
    node_perc = role_extractor.role_percentage.round(2) # pandas.core.frame.DataFrame
    return (features, node_perc)


def getFeatures(edges):
    graph = nx.from_edgelist(edges)
    # extract features
    feature_extractor = RecursiveFeatureExtractor(graph)  # graphrole.features.extract.RecursiveFeatureExtractor
    features = feature_extractor.extract_features()  # pandas.core.frame.DataFrame
    return features

def getRoles(features):
    role_extractor = RoleExtractor(n_roles=10)  # graphrole.roles.extract.RoleExtractor
    role_extractor.extract_role_factors(features)
    # node_roles = role_extractor.roles # dict
    node_perc = role_extractor.role_percentage.round(2)  # pandas.core.frame.DataFrame
    return node_perc

def getRolx(edges):
    graph = nx.from_edgelist(edges)
    feature_extractor = RecursiveFeatureExtractor(graph)
    features = feature_extractor.extract_features()
    role_extractor = RoleExtractor(n_roles=10)
    role_extractor.extract_role_factors(features)
    node_perc = role_extractor.role_percentage.round(2)
    return (features, node_perc)

def getweights(k, windows):
    alpha = 0.7
    if k == 9:
        return math.pow(alpha, k)
    else:
        return  (1 - alpha) * math.pow(alpha, k)

def gettransition(date,base):
    dates = base['date'].tolist()
    reg = pd.DataFrame()
    windows = 10
    if date - timedelta(windows - 1) in dates:
        for k in range(windows):
            weight = getweights(k,windows)
            t = date - timedelta(k)
            idx = base[base.date == t].index
            if k == 0:
                reg = base.loc[idx,'roles'] * weight
            else:
                reg = reg.add(base.loc[idx,'roles'] * weight)
        return reg
    else:
        return np.nan

def getdbmm(pandas_df):
    base = pandas_df[pandas_df.model == 'base']
    dates = set(pandas_df['date'])
    models = pandas_df['model'].unique()
    for model, date in itertools.product(models, dates):
        idx = pandas_df[pandas_df.model == model and pandas_df.date == date].index
        gt = pandas_df.loc[idx, 'roles']
        if model == 'base':
            transition = gettransition(date, base)
            pandas_df.loc[idx, 'transition'] = transition
            pandas_df.loc[idx, 'prediction'] = transition
        else:
            pandas_df.loc[idx, 'transition'] = gt
            pandas_df.loc[idx, 'prediction'] = gt
    return pandas_df

def getfrobeniousloss(pandas_df):
    base = pandas_df[pandas_df.model == 'base'][['date', 'roles']]
    dbmm = pandas_df[pandas_df.model == 'base'][['date', 'prediction']]
    rim = pandas_df[pandas_df.model == 'PIM'][['date', 'prediction']]
    pim = pandas_df[pandas_df.model == 'RIM'][['date', 'prediction']]
    dates = set(pandas_df['date'])
    res = pd.DataFrame()
    for date in dates:
        predictdate = date + timedelta(1)
        if predictdate in dates:
            base_act = base.loc[base[base.date == predictdate].index, 'roles']
            dbmm_pre = dbmm.loc[dbmm[dbmm.date == date].index, 'prediction']
            rim_pre = rim.loc[rim[rim.date == date].index, 'prediction']
            pim_pre = pim.loc[pim[base.date == date].index, 'prediction']
            base_dbmm = LA.norm(base_act.sub(dbmm_pre).as_matrix,'fro')
            base_rim = LA.norm(base_act.sub(rim_pre).as_matrix,'fro')
            base_pim = LA.norm(base_act.sub(pim_pre).as_matrix,'fro')
            res = res.append(pd.Series({'date':predictdate,'base_dbmm':base_dbmm,'base_rim':base_rim,'base_pim':base_pim}),ignore_index=True)
    return res

if __name__ == " __main__":
    if len(sys.argv) != 2:
        print("Usage: input <file>", file=sys.stderr)
        sys.exit(-1)
    spark = SparkSession\
        .Builder\
        .appName("SparkCyberDynmaciGraph") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    df = spark.read.load("/data/test.csv",format="csv", sep="\t", schema=schema, header="true")

    # https://stackoverflow.com/questions/44067861/pyspark-add-a-new-column-with-a-tuple-created-from-columns?noredirect=1&lq=1
    # https://stackoverflow.com/questions/33681487/how-do-i-add-a-new-column-to-a-spark-dataframe-using-pyspark
    df = df.withColumn("edge",struct(df.srcid,df.dstid))
    df = df.groupBy("model","date","wSize","alpha").agg(F.collect_list("edge").alias("edges")).select("model","date","edges")
    pandas_df = df.toPandas()
    pandas_df['date'] = pd.to_datetime(pandas_df['date'])
    pandas_df = pandas_df[pandas_df.date.notnull()]

    #pandas_df = pandas_df.sort_values(by='date', inplace=True, ascending=False)
    #pandas_df.first()

    #edges = pandas_df.head(1)['edges'][0]
    #features = getFeatures(edges)
    #roles = getRoles(features)
    #reg = getRolx(edges)
    #reg.__fields__
    #edges = pandas_df['edges'].map(getRolx)
    #pandas_df['len'] = pandas_df['edges'].map(len)
    pandas_df['features'] = pandas_df['edges'].map(getFeatures)
    pandas_df['roles'] = pandas_df['features'].map(getRoles)
    pandas_df['transition'] = np.nan
    pandas_df['prediction'] = np.nan

    reg = getdbmm(pandas_df)

    #pandas_df[pandas_df.model == 'base'].unique()
    #pandas_df[pandas_df.date == '2015-06-28']




    #df = df.withColumn("features",getFeatures(df.edges))
    #dfrdd = df.rdd.map(lambda row: row.__getitem__("list[edges]"))
    #df = df.withColumn("features", getFeatures("list[edges]"))
    #spark.conf.set("spark.sql.execution.arrow.enabled", "true")


    #b = df.first().__getitem__("edges")
    #graph = nx.from_edgelist(b)
    #feature_extractor = RecursiveFeatureExtractor(graph)
    #features = feature_extractor.extract_features()
    #role_extractor = RoleExtractor(n_roles=10)
    #role_extractor.extract_role_factors(features)
    #node_perc = role_extractor.role_percentage.round(2)

    spark.stop()