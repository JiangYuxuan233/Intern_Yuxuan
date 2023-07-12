#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import numpy as np
import time
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport
import json
from deepchecks.tabular.suites import full_suite
import io

def pandas_profiling_report(df):
    df_report = ProfileReport(df,explorative=True)
    return df_report
def read_load_dataset(source_data):
    df = pd.read_csv(source_data)
    return df 
def generate_dc_report(df):
    suite = full_suite()
    dc_report = suite.run(df)
    return dc_report
#def get_all_keys(results):
#    all_keys = []
#    for item in results:
#        a = json.dumps(item)
#        #with open(item) as json_file:
#        item_json = json.loads(a)
#        if "name" in item_json:
#            all_keys.append(item_json['name'])
#    return all_keys  
def render_dc_report(df):
    dc_report = generate_dc_report(df)
    json_result = json.loads(dc_report.to_json())
    results = None
    name = None
    if "name" in json_result:
        name = json_result["name"]
    if "results" in json_result:
        results = json_result["results"]
    result_keys = []
    #if results is not None and len(results)>0:
    #    result_keys = get_all_keys(results)
    return json_result#name,results,result_keys
def select_deepchecks_specific_item(results):#(dc_selection,results):
    selected_report = []
    if len(results)>0:
        for item in results:
            a = json.dumps(item)
            item_json = json.loads(a)
            #if "name" in item_json["name"] == dc_selection:
             #   selected_report_item = item_json
              #  break
    return item_json#selected_report_item  
def render_deepchecks_test_result(results):#,dc_selection):
    selected_report_json = json.dumps(select_deepchecks_specific_item(results))#,dc_selection)
    
    if "header" in selected_report_json:
        st.header(selected_report_json["header"])
    if "summary" in selected_report_json:
        st.info(selected_report_json["summary"])
#    if "value" in selected_report_json:
#        st.success(selected_report_json["value"])
#    
#    if "conditions_table" in selected_report_json:
#        conditions_table = json.loads(selected_report_json["conditions_table"])
#        if len(conditions_table)>0:
#            st.table(conditions_table)
#   if "display" in selected_report_json:
#        display_data = selected_report_json["display"]
#        if len(display_data)>0:
#            for disp_item in display_data:
#                if "py/tuple" in disp_item:
#                    item_data = disp_item["py/tuple"]
#                    if len(item_data)== 2:
#                        header_item = item_data[0]
#                        value_item = item_data[1]
#                if header_item is not None and value_item is not None:
#                    if header_item in ["str","html"]:
#                        st.write(value_item)
#                   elif header_item == "dataframe":
#                        st.dataframe(json.loads(value_item))


def main():
    with st.sidebar.header("Source Data Selection"):
        st.sidebar.write("Select Dataset")
        source_data = st.sidebar.file_uploader("Upload/select source (.csv) data", type = ["csv"])
        
    df = None
    if source_data is not None:
        df = read_load_dataset(source_data)
        st.header("Dataset")
        
    if df is not None:
        user_choices = ['Dataset Sample','Deepchecks Report','Pandas Profiling']
        selected_choices = st.sidebar.selectbox("Please select your choice:",user_choices)
        if selected_choices is not None:
            if selected_choices == "Dataset Sample":
                st.info("Select dataset has "+str(df.shape[0])+" rows and "+str(df.shape[1])+" columns.")
                st.write(df)  
            elif selected_choices == "Deepchecks Report":
                st.write("Deepchecks")
                #name,results,result_keys = render_dc_report(df)
                r = render_dc_report(df)
                #a = render_dc_report(df)
                #st.write("Deepcheck Reports Type:")
                #st.json(results)
                #result_keys = get_all_keys(results)
                #dc_selection = None
                #if len(result_keys)>0:
                #    dc_selection = st.sidebar.selectbox("Select Deepcheck Report Type:", result_keys)
                #if dc_selection is not None:
                #    selected_dc_result = render_deepchecks_test_result(results,dc_selection)
                #    st.json(selected_dc_result)
                #render_deepchecks_test_result(r) 
                #st.json(selected_dc_result)
                if "header" in r:
                     st.header(r["header"])
                if "summary" in r:
                     st.info(r["summary"])
                st.write(r)
                
                   
                    
            elif  selected_choices == "Pandas Profiling":
                df_report = pandas_profiling_report(df)
                st.write("Profiling")
                st_profile_report(df_report)
                
       
    else:
        st.error("Please select your data to started")
        
main()