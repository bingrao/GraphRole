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

from __future__ import print_function

import sys
from random import random
from operator import add

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType, LongType, DateType, UserDefinedType
from pyspark.sql.functions import struct
from pyspark.sql import functions as F
import networkx as nx
from graphrole import RecursiveFeatureExtractor, RoleExtractor
from datetime import timedelta
from pyspark.sql.functions import udf
import pandas as pd
import math
from numpy import linalg as LA
from pandas.compat import StringIO


if __name__ == "__main__":
    """
        Usage: pi [partitions]
    """
    spark = SparkSession\
        .builder\
        .appName("PythonPi")\
        .getOrCreate()

    partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    n = 100000 * partitions

    schema = StructType([ \
        StructField("model", StringType()), \
        StructField("date", DateType()), \
        StructField("wSize", IntegerType()), \
        StructField("alpha", DoubleType()), \
        StructField("srcid", LongType()), \
        StructField("dstid", LongType()) \
        ])
    input = '/data/test.csv'
    output = '/home/cilab/command/dynamicgraph/reg_1'

    df = spark.read.load(input, format="csv", sep="\t", schema=schema, header="true").dropna()
    # https://stackoverflow.com/questions/44067861/pyspark-add-a-new-column-with-a-tuple-created-from-columns?noredirect=1&lq=1
    # https://stackoverflow.com/questions/33681487/how-do-i-add-a-new-column-to-a-spark-dataframe-using-pyspark
    df_edges = df.withColumn("edge", struct(df.srcid, df.dstid))
    df_edgeslist = df_edges.groupBy("model", "date", "wSize", "alpha") \
        .agg(F.collect_list("edge").alias("edges")) \
        .select("model", "date", "edges")


    @udf(returnType=StringType())
    def getFeatures(edges):
        graph = nx.from_edgelist(edges)
        # extract features
        feature_extractor = RecursiveFeatureExtractor(graph)  # graphrole.features.extract.RecursiveFeatureExtractor
        features = feature_extractor.extract_features()  # pandas.core.frame.DataFrame
        return features.to_string()


    @udf(returnType=StringType())
    def getRoles(features_str):
        features = pd.read_csv(pd.compat.StringIO(features_str), sep='\s+')
        role_extractor = RoleExtractor()  # graphrole.roles.extract.RoleExtractor n_roles=10
        role_extractor.extract_role_factors(features)
        # node_roles = role_extractor.roles # dict
        node_perc = role_extractor.role_percentage.round(2)  # pandas.core.frame.DataFrame
        return node_perc.to_string()


    @udf(returnType=StringType())
    def getrolesonetime(edges):
        graph = nx.from_edgelist(edges)
        # extract features
        feature_extractor = RecursiveFeatureExtractor(graph)  # graphrole.features.extract.RecursiveFeatureExtractor
        features = feature_extractor.extract_features()  # pandas.core.frame.DataFrame
        # for each subgraph, it may have different roles, so we do not specify the number of roles here
        role_extractor = RoleExtractor()  # graphrole.roles.extract.RoleExtractor
        role_extractor.extract_role_factors(features)
        # node_roles = role_extractor.roles # dict
        node_perc = role_extractor.role_percentage.round(2)  # pandas.core.frame.DataFrame
        return node_perc.to_string()


    #df_edgeslist = df_edgeslist.sample(0.012,10) # debug statement

    #df_features = df_edgeslist.withColumn("features", getFeatures(df_edgeslist.edges))
    #df_roles = df_features.withColumn("roles", getRoles(df_features.features))

    df_roles = df_edgeslist.withColumn("roles", getrolesonetime(df_edgeslist.edges))

    # Broadcast data to all workers to reduce shuffling data
    # Cannot convert data into pd.dataframe directory since a string will be truncated
    # https://github.com/pandas-dev/pandas/issues/9784
    df_broadcast = spark.sparkContext.broadcast(df_roles.select("model", "date", "roles").collect())


    def getweights(k, windows):
        alpha = 0.7
        if k == 9:
            return math.pow(alpha, k)
        else:
            return (1 - alpha) * math.pow(alpha, k)


    def getrolefromlist(model, date, data_list):
        reglist = list(
            filter(lambda row: row.__getitem__('model') == model and row.__getitem__('date') == date, data_list))
        if len(reglist) > 0:
            return reglist[0].__getitem__('roles')
        else:
            return None


    @udf(returnType=StringType())
    def gettransition(model, date):
        data_list = df_broadcast.value
        dates = set(map(lambda row: row.__getitem__('date'), data_list))
        reg = pd.DataFrame()
        windows = 10
        if date - timedelta(windows - 1) in dates:  # make sure include enought data to get transition
            if model == "base":
                for k in range(windows):
                    weight = getweights(k, windows)
                    t = date - timedelta(k)
                    role = pd.read_csv(pd.compat.StringIO(getrolefromlist(model, t, data_list)), sep='\s+')
                    if role is None:  # debug problem
                        print("[Debug] return role is np.nan: " + t + " model: " + model)
                    else:
                        reg = reg.add(role * weight, fill_value=0).fillna(0)
                return reg.to_string()
            else:
                return getrolefromlist(model, date, data_list)
        else:
            return None


    @udf(returnType=StringType())
    def gettransition_debug(model, date):
        data_list = df_broadcast.value
        gs = getrolefromlist(model, date, data_list)
        if model == "base":
            weight = getweights(0, 10)
            role = pd.read_csv(pd.compat.StringIO(gs), sep='\s+') * weight
            return role.to_string()
        else:
            return gs


    # The formal paramter of an UDF must be an object with type of str or colummn
    df_transition = df_roles.withColumn("transition",
                                        gettransition(df_roles.model, df_roles.date)).dropna()


    @udf(returnType=StringType())
    def getprediction(model, role, transition):
        if model == 'base':
            role_df = pd.read_csv(pd.compat.StringIO(role), sep='\s+')  # dataframe
            transition_df = pd.read_csv(pd.compat.StringIO(transition), sep='\s+')  # dataframe
            prediction = role_df.multiply(transition_df).dropna()
            return prediction.to_string()
        else:
            return role


    df_prediction = df_transition.withColumn("prediction",
                                             getprediction(df_transition.model, df_transition.roles,
                                                           df_transition.transition))

    @udf(returnType=DoubleType())
    def getfrobeniousloss(model, date, pred):
        data_list = df_broadcast.value
        dates = set(map(lambda row: row.__getitem__('date'), data_list))
        pre_date = date + timedelta(1)  # predict next day graph
        if pre_date in dates:
            act_roles = pd.read_csv(pd.compat.StringIO(getrolefromlist(model, pre_date, data_list)), sep='\s+')
            pred_roles = pd.read_csv(pd.compat.StringIO(pred), sep='\s+')
            diff = act_roles.sub(pred_roles).dropna()  # only compare common element
            loss = LA.norm(diff, 'fro')
            return float(loss)
        else:
            return None


    df_loss = df_prediction \
        .withColumn('loss', getfrobeniousloss(df_prediction.model, df_prediction.date, df_prediction.prediction)) \
        .dropna().select('model', 'date', 'loss')

    for row in df_loss.collect():
        print("######################: " + str(row) + "\n")

    spark.stop()