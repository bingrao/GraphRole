# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:16:08 2019

@author: Bing
"""

import networkx as nx
from graphrole import RecursiveFeatureExtractor, RoleExtractor
import mysql.connector

from datetime import datetime  
from datetime import timedelta



def getRolx(s:str, e: str):
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

s = datetime(2013,5,29,23,59,59)  # http://www.pressthered.com/adding_dates_and_times_in_python/
e = datetime(2013,5,15,00,00,00)

f,r = getRolx('2013-05-29 23:23:59','2013-05-15 00:00:00')




#f,r = getRolx(str(s),str(e))

df = spark.read.load("/data/test.csv",format="csv", sep="\t", inferSchema="true", header="true")
edge = df.withColumn('edge', (df.srcid,dst.dstid))







windows = 10
alpha = 0.7



