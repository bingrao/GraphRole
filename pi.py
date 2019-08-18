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
import networkx as nx
import pandas as pd
import math
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType, LongType, DateType, FloatType
from pyspark.sql.functions import struct
from pyspark.sql import functions as f
from graphrole import RecursiveFeatureExtractor, RoleExtractor
from datetime import timedelta
from pyspark.sql.functions import udf
from numpy import linalg as la
from pandas.compat import StringIO

if __name__ == "__main__":
    """
        Usage: pi [partitions]
    """
    spark = SparkSession \
        .builder \
        .appName("PythonPi") \
        .getOrCreate()
    inputdata = '/data/test.csv'
    output = '/output/dynamic.parquet'

    if len(sys.argv) > 2:
        inputdata = str(sys.argv[1])
        output = str(sys.argv[2])

    print(f"[*dynamgraph*]\tthe input file is {inputdata}, the output file: {output}.\n")

    inputschema = StructType([
        StructField("model", StringType()),
        StructField("date", DateType()),
        StructField("wSize", IntegerType()),
        StructField("alpha", DoubleType()),
        StructField("srcid", LongType()),
        StructField("dstid", LongType())
    ])

    windows = 10
    alpha = 0.7

    # root
    # | -- model: string(nullable=true)
    # | -- date: date(nullable=true)
    # | -- wSize: integer(nullable=true)
    # | -- alpha: double(nullable=true)
    # | -- srcid: long(nullable=true)
    # | -- dstid: long(nullable=true)
    df = spark.read \
        .load(inputdata, format="csv", sep="\t", schema=inputschema, header="true") \
        .dropna()  # Remove unvalid data
    # https://stackoverflow.com/questions/44067861/pyspark-add-a-new-column-with-a-tuple-created-from-columns?noredirect=1&lq=1
    # https://stackoverflow.com/questions/33681487/how-do-i-add-a-new-column-to-a-spark-dataframe-using-pyspark

    # root
    # | -- model: string(nullable=true)
    # | -- date: date(nullable=true)
    # | -- wSize: integer(nullable=true)
    # | -- alpha: double(nullable=true)
    # | -- srcid: long(nullable=true)
    # | -- dstid: long(nullable=true)
    # | -- edge: struct(nullable=false)
    # | | -- srcid: long(nullable=true)
    # | | -- dstid: long(nullable=true)
    df_edges = df.withColumn("edge", struct(df.srcid, df.dstid))

    # root
    # | -- model: string(nullable=true)
    # | -- date: date(nullable=true)
    # | -- edges: array(nullable=true)
    # | | -- element: struct(containsNull=true)
    # | | | -- srcid: long(nullable=true)
    # | | | -- dstid: long(nullable=true)
    df_edgeslist = df_edges.groupBy("model", "date", "wSize", "alpha") \
        .agg(f.collect_list("edge").alias("edges")) \
        .select("model", "date", "edges")

    # df_edgeslist = df_edgeslist.sample(0.3, 10)  # debug statement

    def getdataframefromstring(inputstr):
        return pd.read_csv(pd.compat.StringIO(inputstr), sep='\s+')


    @udf(returnType=StringType())
    def getfeatures(edges):
        graph = nx.from_edgelist(edges)
        # extract features
        feature_extractor = RecursiveFeatureExtractor(graph)  # graphrole.features.extract.RecursiveFeatureExtractor
        features = feature_extractor.extract_features()  # pandas.core.frame.DataFrame
        return features.to_string()


    @udf(returnType=StringType())
    def getroles(features_str):
        features = getdataframefromstring(features_str)
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


    # df_features = df_edgeslist.withColumn("features", getfeatures(df_edgeslist.edges))
    # df_roles = df_features.withColumn("roles", getroles(df_features.features))

    # root
    # | -- model: string(nullable=true)
    # | -- date: date(nullable=true)
    # | -- roles: string(nullable=true)
    df_roles = df_edgeslist \
        .withColumn("roles", getrolesonetime(df_edgeslist.edges)) \
        .select("model", "date", "roles").cache()

    df_roles.write.parquet(output)

    # StructType(List(StructField(model, StringType, true), StructField(date, DateType, true),
    #                 StructField(roles, StringType, true)))
    # df_roles.schema

    # Broadcast data to all workers to reduce shuffling data
    # Cannot convert data into pd.dataframe directory since a string will be truncated
    # https://github.com/pandas-dev/pandas/issues/9784
    base_df_roles = df_roles.filter(df_roles.model == 'base').collect()
    base_df_broadcast = spark.sparkContext.broadcast(base_df_roles)  # Array of Row[model, date, roles]

    def getrolestrfrombaselist(desired_date): # In the base model
        reglist = list(
            filter(lambda rowdata: rowdata.__getitem__('date') == desired_date, base_df_broadcast.value))
        return reglist[0].__getitem__('roles') if len(reglist) > 0 else None

    def getdatesfrombaselist(): # In the base model
        return set(map(lambda rowdata: rowdata.__getitem__('date'), base_df_broadcast.value))

    def getrolestrfromlist(model, date, data_list):
        reglist = list(
            filter(lambda rowdata: rowdata.__getitem__('model') == model and rowdata.__getitem__('date') == date,
                   data_list))
        if len(reglist) > 0:
            return reglist[0].__getitem__('roles')
        else:
            return None

    @udf(returnType=StringType())
    def gettransition(model, date, roles):
        base_dates = getdatesfrombaselist()
        reg = pd.DataFrame()
        if date - timedelta(windows - 1) in base_dates:  # make sure include enought data to get transition
            if model == "base":
                for k in range(windows):
                    weight = math.pow(alpha, k) if k == 9 else (1 - alpha) * math.pow(alpha, k)
                    iterdate = date - timedelta(k)
                    role_str = getrolestrfrombaselist(iterdate)
                    if role_str is None:  # debug problem
                        print(f"[*dynamgraph*]\treturn role is np.nan: {iterdate}, model: {model}")
                    else:
                        role = getdataframefromstring(role_str)  # Dataframe
                        reg = reg.add(role * weight, fill_value=0).fillna(0)
                return None if reg.empty else reg.to_string()
            else:
                return roles
        else:
            return None


    @udf(returnType=StringType())
    def gettransitiondebug(model, date, roles):  # debug and test
        if model == "base":
            gs = getrolestrfrombaselist(date)
            weight = math.pow(alpha, 7)
            role = getdataframefromstring(gs) * weight
            return role.to_string()
        else:
            return roles


    # root
    # | -- model: string(nullable=true)
    # | -- date: date(nullable=true)
    # | -- roles: string(nullable=true)
    # | -- transition: string(nullable=true)
    # The formal paramter of an UDF must be an object with type of str or col in spark
    df_transition = df_roles\
        .withColumn("transition", gettransition(df_roles.model, df_roles.date, df_roles.roles))\
        .dropna()


    @udf(returnType=StringType())
    def getprediction(model, role, transition):
        if model == 'base':
            role_df = getdataframefromstring(role)  # dataframe
            transition_df = getdataframefromstring(transition)  # dataframe
            prediction = role_df.multiply(transition_df, fill_value=0).fillna(0)
            return prediction.to_string()
        else:
            return role


    # root
    # | -- model: string(nullable=true)
    # | -- date: date(nullable=true)
    # | -- roles: string(nullable=true)
    # | -- transition: string(nullable=true)
    # | -- prediction: string(nullable=true)
    df_prediction = df_transition\
        .withColumn("prediction", getprediction(df_transition.model, df_transition.roles, df_transition.transition))


    @udf(returnType=FloatType())
    def getfrobeniousloss(model, date, pred):
        base_dates = getdatesfrombaselist()
        pre_date = date + timedelta(1)  # predict next day graph
        if pre_date in base_dates:
            base_str = getrolestrfrombaselist(pre_date)
            if base_str is None:
                print(f"[*dynamgraph*]\treturn baseline role is np.nan: {date}:{pre_date}, model: {model}")
                return None
            else:
                base_roles = getdataframefromstring(base_str)
                pred_roles = getdataframefromstring(pred)
                diff = base_roles.sub(pred_roles).fillna(0)  # only compare common element, Nan is replace by 0
                loss = la.norm(diff, 'fro')
                # https://stackoverflow.com/questions/38984775/spark-errorexpected-zero-arguments-for-construction-of-classdict-for-numpy-cor
                return float(loss)  # Convert float type in la.norm to the float type of Spark
        else:
            return None


    # root
    # | -- model: string(nullable=true)
    # | -- date: date(nullable=true)
    # | -- roles: string(nullable=true)
    # | -- transition: string(nullable=true)
    # | -- prediction: string(nullable=true)
    # | -- loss: float(nullable=true)
    df_loss = df_prediction \
        .withColumn('loss', getfrobeniousloss(df_prediction.model, df_prediction.date, df_prediction.prediction))\
        .dropna()

    shapeschema = StructType([
        StructField("nrow", IntegerType(), True),
        StructField("ncol", IntegerType(), True)
    ])

    @udf(shapeschema)
    def getshape(df_str):
        df = getdataframefromstring(df_str)
        return df.shape


    # root
    # | -- model: string(nullable=true)
    # | -- date: date(nullable=true)
    # | -- roles: string(nullable=true)
    # | -- transition: string(nullable=true)
    # | -- prediction: string(nullable=true)
    # | -- loss: float(nullable=true)
    # | -- roles_ndim: struct(nullable=true)
    # | | -- nrow: integer(nullable=true)
    # | | -- ncol: integer(nullable=true)
    # | -- trans_ndim: struct(nullable=true)
    # | | -- nrow: integer(nullable=true)
    # | | -- ncol: integer(nullable=true)
    # | -- predi_ndim: struct(nullable=true)
    # | | -- nrow: integer(nullable=true)
    # | | -- ncol: integer(nullable=true)
    df_reg = df_loss \
        .withColumn('roles_ndim', getshape(df_loss.roles)) \
        .withColumn('trans_ndim',getshape(df_loss.transition))\
        .withColumn('predi_ndim',getshape(df_loss.prediction))

    # nrow = df_reg.first().__getitem__('roles_ndim').__getitem__('nrow')

    print(f"[*dynamgraph*]\tmodel\tdate\troles_ndim\ttrans_ndim\tpredi_ndim\tloss\n")

    for row in df_reg.select('model', 'date', 'roles_ndim', 'trans_ndim', 'predi_ndim', 'loss').collect(): # Array[Row]
        model = row.__getitem__('model')
        date = row.__getitem__('date')
        roles_ndim = row.__getitem__('roles_ndim')
        trans_ndim = row.__getitem__('trans_ndim')
        predi_ndim = row.__getitem__('predi_ndim')
        loss = row.__getitem__('loss')
        print(f"[*dynamgraph*]\t{model}\t{date}\t{roles_ndim}\t{trans_ndim}\t{predi_ndim}\t{loss}\n")

    spark.stop()
