import streamlit as st
import numpy as np
import scipy
from scipy import stats
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport
import statsmodels.api as sm
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
import plotly.graph_objects as pxx
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split
import sympy as smp
import pandera as pa
from pandera.typing import DataFrame, Series
import seaborn as sns
from pandera import Column, DataFrameSchema
from sklearn.ensemble import RandomForestRegressor
def callback():
    st.session_state['btn_clicked'] = True
def remove_outliers(data, column, z_threshold=3):
    z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
    return data[z_scores < z_threshold]
def remove_duplicates(data):
    return pd.DataFrame.drop_duplicates(data)
def remove_nulls(data):
    return data.dropna()
def pandas_profiling_report(df):
    df_report = ProfileReport(df,explorative=True)
    return df_report
def read_csv(source_data):
    df = pd.read_csv(source_data)
    return df 
def read_excel(source_data):
    df = pd.read_excel(source_data)
    return df
def OLS(df,S1):
    train=df.drop([S1],axis=1)
    train1 = train.loc[df[S1]!=0]
    test = df.loc[df[S1]!=0]
    constant = sm.add_constant(train1)
    model = sm.OLS(list(test[S1]),constant)
    result = model.fit()
    new_constant=sm.add_constant(train)
    pred = result.predict(new_constant)
    return test
def hig(m,c,df,i):
    col1,col2 = st.columns([1,3])
    with col1:
        st.dataframe(m.iloc[:,[-2,-1,-4,-3]])
    w = m["failure_case"]
    if c==[]:
        st.write("There are ",0,"rows in our error result table(left)")
    else:
        #st.write(len(m["check"].unique()))
        for ww in range(len(m["check"].unique())):
            st.write(m["check"].unique()[ww],": There are" ,len(m.loc[m["check"]==m["check"].unique()[ww]]),"wrong rows")
    def highlight(s):
        if s[i] in list(w):
            return ['background-color: yellow'] * len(s)
        #if s.keys==i:
        #    color = 'green' if s else 'white'
        #    return f'background-color: {color}'
        else:
            return ['background-color: white'] * len(s)
    def hh(s):
        if s in list(w) :
            color = "green"
        else:
            color = 'white'
        return f'background-color: {color}'
    with col2:
        openn = df.style.apply(highlight, axis=1)
        st.dataframe(openn.applymap(hh, subset=[i]))
        #st.write(openn)
def main():
    df = None
    with st.sidebar.header("Source Data Selection"):
        selection = ["csv",'excel']
        selected_data = st.sidebar.selectbox("Please select your dataset format:",selection)
        if selected_data is not None:
            if selected_data == "csv":
                st.sidebar.write("Select Dataset")
                source_data = st.sidebar.file_uploader("Upload/select source (.csv) data", type = ["csv"])
                if source_data is not None: 
                    df = pd.read_csv(source_data)       
            elif selected_data == "excel":
                st.sidebar.write("Select Dataset")
                source_data = st.sidebar.file_uploader("Upload/select source (.xlsx) data", type = ["xlsx"])
                if source_data is not None:
                    df = pd.read_excel(source_data)
                   
        
       

        st.header("Dataset")
        
    if df is not None:
        user_choices = ["Data Quality","Data Validation"]
        selected_choices = st.sidebar.selectbox("Please select your choice:",user_choices)
        word = []
        dff = df.copy()
        st.sidebar.write("Would you consider 0 as missing value in the dataset (except classification column)?",key=f"MyKey{13}")
        y =  st.sidebar.checkbox("Yes",key=f"MyKey{3223}")
        n =  st.sidebar.checkbox("No",key=f"MyKey{33333}")
        if y:
            st.sidebar.write("Is there any columns for classification in the dataset?",key=f"MyKey{123}")
            Y = st.sidebar.checkbox("Yes")
            N = st.sidebar.checkbox("No")
            select = df.keys()
            if Y:
                st.sidebar.write("Which columns are classification columns?",key=f"MyKey{133}")
                for i in select:
                    X = st.sidebar.checkbox(i)
                    if X:
                        word.append(i)
                    elif not X:
                        df[i].replace(0,np.nan,inplace=True)
            elif N:
                #st.write("L")
                df.replace(0,np.nan,inplace=True)                  
        elif n:
            df=dff
            st.sidebar.write("Are there any columns for classification in the dataset?",key=f"MyKey{123}")
            Y = st.sidebar.checkbox("Yes")
            N = st.sidebar.checkbox("No")
            select = df.keys()
            if Y:
                st.sidebar.write("Which columns are classification columns?",key=f"MyKey{3433}")
                for i in select:
                    X = st.sidebar.checkbox(i)
                    if X:
                         word.append(i) 
        #st.info("Selected dataset has "+str(df.shape[0])+" rows and "+str(df.shape[1])+" columns.")
        with st.expander("See Data Sample"):
            st.info("Selected dataset has "+str(df.shape[0])+" rows and "+str(df.shape[1])+" columns.")
            st.write(df) 
        if selected_choices is not None:                     
            if  selected_choices == "Data Quality":
                box = ["Overview","Score","Data types","Descriptive statistics","Missing values","Duplicate records",
                     "Correlation", "Outliers","Data distribution","Random Forest"]
                selection = st.selectbox("Data Quality Selection",box,key=f"MyKey{4}") 
                if selection is not None:
                    if selection == "Overview":
                        #df_report = pandas_profiling_report(df)
                        st.write("Profiling")
                        #st_profile_report(df_report)
                    elif selection == "Data types":
                        types = pd.DataFrame(df.dtypes)
                        
                        a = types.astype(str)
                        st.dataframe(a)
                    elif selection == "Descriptive statistics":
                        types = pd.DataFrame(df.describe())
                        www = st.multiselect("Please select any data you want to check",types.keys(),key=f"MyKey{555}")
                        q = None
                        if len(www)>0:
                            q = pd.DataFrame({})
                            for i in www:
                                q[i]=types[i]
                        #if q is not None:
                            st.write(q.T)
                        else:
                            st.table(types.T)
                    elif selection == "Missing values":
                        col1,col2 = st.columns([1,4])
                        types = pd.DataFrame(df.isnull().sum())           
                        a = types.astype(str)
                        fig = plt.figure(figsize=(15,10))
                        g=sns.heatmap(df.isna().transpose(),cmap="YlGnBu",cbar_kws={'label': 'Missing Data'})
                        g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 16)
                        plt.savefig("visualizing_missing_data_with_heatmap_Seaborn_Python.png", dpi=100)
                        plt.yticks(rotation=0) 
                        col2.pyplot(fig)
                        col1.write(a)
                        box = df.keys()
                        se = st.selectbox("Show missing values",box,key=f"MyKey{5}")
                        for i in box:
                            if se == i:
                                st.write(df[pd.isnull(df[i])])
                        if st.button("Remove Null Values"):
                            data = remove_nulls(df)
                            st.write("Data after removing null values")
                            st.write(data)
                    elif selection == "Duplicate records":
                        types = df[df.duplicated()]
                        
                        a = types.astype(str)
                        st.write("The number of duplicated rows is ",len(types))
                        st.write(a)
                        if st.button("Remove Duplicate Values"):
                            df = remove_duplicates(df)
                            st.write("Data after removing duplicate values")
                            st.write(df)
                    elif selection == "Outliers":
                        fig = plt.figure(figsize=(15,20))
                        box = df.select_dtypes(include=['int',"float"])
                        for i in range(len(box.keys())):
                            plt.subplot(len(box.keys()),1,i+1)
                            sns.boxplot(x=df[box.keys()[i]])
                            plt.xlabel(box.keys()[i],fontsize=18)  
                        fig.tight_layout()
                        st.pyplot(fig)
                        remove = st.checkbox("Remove Outlier Values")
                        if remove:
                            selected_column = st.multiselect("Select column", df.keys(),key=f"MyKey{555}")
                            dff = df
                            for i in range(len(selected_column)):  
                                    df = remove_outliers(df, selected_column[i])
                            if st.button("Remove All Outlier"):
                                for i in df.keys():
                                    df = remove_outliers(df, i)
                                st.write(df)
                                st.write(len(df))
                            else:
                                st.write("Data after removing outliers")
                                st.write(df)
                                st.write(len(df))
                    elif selection == "Data distribution":
                        boxs= df.select_dtypes(include=['int',"float"])
                        box = df.keys()
                        se = st.selectbox("Select which column you want to check",box,key=f"MyKey{6}")
                        for i in boxs:
                            if se == i and se not in word:
                                tab1,tab2,tab3 = st.tabs(["Histogram graph","Scatter graph","Line graph"])
                                with tab1:
                                    fig = plt.figure(figsize=(4,3))
                                    #y0,y1 = st.slider("If you want to change the range of the graph, tell me the y axis range",
                                    #                  value=(0, len(df[i])))
                                    x0 = None
                                    agree = st.checkbox("Change graph range",key=f"MyKey{67551}")
                                    if agree:
                                            x0,x1 = st.slider("If you want to change the range of the graph, tell me the x axis range",min(df[i]), max(df[i]),(min(df[i]), max(df[i])),key=f"MyKey{67}")
                                            y0,y1 = st.slider("If you want to change the range of the graph, tell me the y axis range",0,max(df[i].value_counts()*4),(0,max(df[i].value_counts())),key=f"MyKey{700}")
                                    if word != []:
                                        ss = st.selectbox("What classification condition do you want?",word,key=f"MyKey{321}")
                                        for j in word: 
                                            if ss == j: 
                                                sns.histplot(data = df,x=i,binwidth=3,kde=True,hue=j)
                                                if x0 is not None:
                                                    plt.xlim(x0, x1)
                                                    plt.ylim(y0, y1)    
                                                st.pyplot(fig)
                                                
                                    elif word ==[]:
                                        sns.histplot(data = df,x=i,binwidth=3,kde=True)
                                        if x0 is not None:
                                            plt.xlim(x0, x1)
                                            plt.ylim(y0, y1)    
                                        st.pyplot(fig)
                                                      
                
                                with tab2:
                                    fig = plt.figure(figsize=(4,3))
                                    x0 = None
                                    df[" "]=np.arange(len(df))
                                    agree = st.checkbox("Change graph range",key=f"MyKey{6755}")
                                    if agree:
                                            x0,x1 = st.slider("If you want to change the range of the graph, tell me the x axis range",min(df[i]), max(df[i]),(min(df[i]), max(df[i])),key=f"MyKey{699}")
                                            y0,y1 = st.slider("If you want to change the range of the graph, tell me the y axis range",min(df[" "]),max(df[" "]),(min(df[" "]),max(df[" "])),key=f"MyKey{700}")
                                    if word != []:
                                        ss = st.selectbox("What classification condition do you want?",word,key=f"MyKey{1128}")
                                        for j in word: 
                                            if ss == j: 
                                                sns.scatterplot(data = df,x=i,y=" ",hue=j)
                                                if x0 is not None:
                                                    plt.xlim(x0, x1)
                                                    plt.ylim(y0, y1)
                                                st.pyplot(fig)
                                    elif word ==[]:
                                        sns.scatterplot(data = df,x=i,y=" ")
                                        if x0 is not None:
                                            plt.xlim(x0, x1)
                                            plt.ylim(y0, y1)
                                        st.pyplot(fig)
                                with tab3:
                                    fig = plt.figure(figsize=(4,3))
                                    df["range"]=np.arange(len(df))
                                    x0 = None
                                    agree = st.checkbox("Change graph range",key=f"MyKey{65}")
                                    if agree:
                                            
                                            x0,x1 = st.slider("If you want to change the range of the graph, tell me the x axis range",min(df["range"]),max(df["range"]),(min(df["range"]),max(df["range"])),key=f"MyKey{701}")
                                            y0,y1 = st.slider("If you want to change the range of the graph, tell me the y axis range",min(df[i]), max(df[i]),(min(df[i]), max(df[i])),key=f"MyKey{67055}")
                                    if word != []:
                                        ss = st.selectbox("What classification condition do you want?",word,key=f"MyKey{1122}")
                                        for j in word:
                                            if ss == j: 
                                                sns.lineplot(data = df,x="range",y=i,hue=j)
                                                if x0 is not None:
                                                    plt.xlim(x0, x1)
                                                    plt.ylim(y0,y1)
                                                st.pyplot(fig)
                                    elif word ==[]:
                                        sns.lineplot(data = df,x="range",y=i)
                                        if x0 is not None:
                                            plt.xlim(x0, x1)
                                            plt.ylim(y0,y1)
                                        st.pyplot(fig)
                               
                            elif se ==i and se in word:
                                tab1,tab2 = st.tabs(["Pie","  "])
                                with tab1:
                                    fig = plt.figure(figsize=(3,3))
                                    p=df.groupby([i]).size().plot(kind='pie', y='counts',autopct='%1.0f%%')
                                    p.set_ylabel('Counts', size=11)
                                    st.pyplot(fig)
                               
                       
                    elif selection == "Correlation":
                        box = df.keys()
                        sr = st.multiselect("Select the columns you want to compare",box,key=f"MyKey{10}")
                        new = {}
                        new = pd.DataFrame(new)
                        for i in range(len(sr)):
                            new[sr[i]]=df[sr[i]]
                        if new.empty == False:
                            fig,ax = plt.subplots()
                            sns.heatmap(new.corr(),annot = True,ax=ax)
                            st.pyplot(fig)
                        boxs= df.select_dtypes(include=['int',"float"])
                        st.write("Or see the correlation for entire dataset")
                        if st.button("Correlation of entire dataset"):
                            figs,ax = plt.subplots()
                            sns.heatmap(boxs.corr(),annot = True,ax=ax)
                            st.pyplot(figs)
                    elif selection == "Score":
                        x = []
                        box = df.keys()
                        y =len(df[df.isna().any(axis=1)])
                        z = df.duplicated().sum()
                        box = df.keys()
                        for i in box:
                            if df[i].dtypes == "int64" or  df[i].dtypes == "float64":
                                x.append(len(df[(np.abs(stats.zscore(df[i])) >= 3)]))
                        error = sum(x)+y+z
                        
                        a = st.write("number of missing values in the dataset is",y
                                    ) 
                        st.write("number of duplicated rows in the dataset is",z)
                        st.write("number of extreme values in the dataset is", sum(x))
                        st.write("the dataset has",len(df),"rows")
                        st.write("Overall, the score of data is ",round(100*(1-error/len(df))),"percent")
                        st.latex(r'''score = (a*missing+b*extreme+c*duplication)/total''')
                        accuracy = st.number_input("Score_Accuracy")
                        completeness = st.number_input("Score_Completeness")
                      
                        
                    elif selection == "Random Forest":
                        X,y = dff.iloc[:,1:].values,dff.iloc[:,0].values
                        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
                        regressor = RandomForestRegressor(n_estimators=100,
                                  random_state=0)
                        regressor.fit(x_train, y_train)
                        importances = regressor.feature_importances_
                        indices = np.argsort(importances)[::-1]
                        fig = plt.figure(figsize=(4,3))
                        plt.ylabel("Feature importance")
                        plt.bar(range(x_train.shape[1]),importances[indices],align="center")
                        feat_labels = dff.columns[1:]
                        plt.xticks(range(x_train.shape[1]),feat_labels[indices],rotation=60)
                        plt.xlim([-1,x_train.shape[1]])
                        st.pyplot(fig)
            elif selected_choices == "Data Validation":
                b = df.keys()
                se = st.selectbox("Choose columns",b,key=f"MyKey{12}")
                for i in b:  
                    if se is not None :
                        if se == i and i not in word:
                            if df[i].isnull().sum()==len(df[i]):
                                st.error("this column is empty")
                            else:
                                st.write("What type of data is this in the column?")
                                agree = st.checkbox('Int/Float')
                                c = []
                                if agree:
                                    #df[i]=str(df[i])
                                    #df[i] = [float("".join(df[i][1:].split(","))), "$"]
                                    st.write("What is the range of your data?")
                                    minimum = st.number_input('Insert the minimum value')
                                    
                                    maximum = st.number_input('Insert the maximum value')
                                    
                                    if st.button("Error Result"):
                                        schema = pa.DataFrameSchema({
                                      i: pa.Column(float, pa.Check.in_range(minimum,maximum),coerce=True)                                                                                                                       })
                                        try:
                                            schema.validate(df, lazy=True)
                                        except pa.errors.SchemaErrors as err:
                                                m = err.failure_cases
                                                c.append(len(err.failure_cases))
                                                hig(m,c,df,i)
                                        if len(c)==0:
                                              st.write("There are ",0,"rows out of range")
                                       

                                agrees = st.checkbox('Str')
                                if agrees:
                                   # df[i]=df[i].astype(str)
                                    st.write("What is the range of your data?")
                                    minimum = st.number_input('Insert the minimum length of your string')-1
                                    maximum = st.number_input('Insert the maximum length of your string')+1
                                    t = st.text_input("If you want, type the Regular Expression of your data")
                                    if st.button("Error Result",f"MyKey{43}"):
                                        c = []
                                        schema = pa.DataFrameSchema({
                                        i: pa.Column(str, [pa.Check(lambda x: len(x) > minimum, element_wise=True),
                                                         pa.Check(lambda x: len(x) < maximum, element_wise=True),
                                                          pa.Check.str_matches(t)])})
                                        try:
                                            schema.validate(df, lazy=True)
                                        except pa.errors.SchemaErrors as err:
                                            m = err.failure_cases
                                            c.append(err.failure_cases["failure_case"].duplicated().sum())
                                            hig(m,c,df,i)
                                        if len(c)==0:
                                             st.write("There are ",0,"rows out of range")
                                       
                                agreesss = st.checkbox('Date/Time')
                                if agreesss:
                                    st.write("What is the range of your data?")
                                    minimum = st.date_input('Insert the minimum date')
                                    maximum = st.date_input('Insert the maximum date')
                                    df[i] = df[i].astype('datetime64[ns]')
                                    st.write(i)
                                    if st.button("Error Result"):
                                        
                                        df[i]= df[i].astype('datetime64[ns]').dt.tz_localize(None)
                                        c =[]
                                        schema = pa.DataFrameSchema({
                                       i: pa.Column("datetime64[ns]",[pa.Check(lambda x: x > pd.to_datetime(minimum),element_wise=True),
                                                      pa.Check(lambda x: x < pd.to_datetime(maximum),element_wise=True) ]               
                                                             )})  
                                        try:
                                            schema.validate(df, lazy=True)
                                        except pa.errors.SchemaErrors as err:
                                            m = err.failure_cases
                                            c.append(len(err.failure_cases))
                                            hig(m,c,df,i)
                                        if len(c)==0:
                                            st.write("There are ",0,"rows out of range")
                                      
                                st.write("Are all values in the column unique？") 
                                Y = st.checkbox("Yes",key=f"MyKey{32}")
                                #N = st.checkbox("No",key=f"MyKey{42}")
                                if Y:
                                    st.write("In",i,"there are ",len(df[df[i].duplicated()]),"rows with repeated values.")
                                   # def repeat(s):
                                   #      if s[i] in list(df[i][df[i].duplicated()]):
                                   #         return ['background-color: yellow'] * len(s)
                                   #      else:
                                   #         return ['background-color: white'] * len(s)
                                   # st.table(df.style.apply(repeat, axis=1)).head()
                                    st.write(df[df[i].duplicated()])
                        elif se == i and i in word:
                            if df[i].isnull().sum()==len(df[i]):
                                st.error("this column is empty")
                            else: 
                                options = st.radio("What sorting options are in this column？",
                                                         ("number","text"))
                                if options == "number":
                                    c = []
                                    number = st.number_input('How many categories are there？')
                                    bar = np.arange(int(number))
                                    if st.button("Error Result",key=f"MyKey{52}"):
                                            schema = pa.DataFrameSchema({
                                             i: pa.Column(int,pa.Check.isin(bar), coerce=True)                                                                                                                       })
                                            try:
                                                schema.validate(df, lazy=True)
                                            except pa.errors.SchemaErrors as err: 
                                                m = err.failure_cases
                                                c.append(len(err.failure_cases))
                                                hig(m,c,df,i)
                                            if len(c)==0:
                                                st.write("There are ",0,"rows out of range")
                                           
                                elif options == "text":
                                    barr = []
                                    c = []
                                    number = st.number_input('How many categories are there？')
                                    for ii in range(int(number)):
                                        text = st.text_input("insert name of category",key=f"MyKey{ii}")
                                        barr.append(text)
                                    if st.button("Error Result",f"MyKey{42}"):
                                        schema = pa.DataFrameSchema({
                                         i: pa.Column(str,pa.Check.isin(barr), coerce=True)                                                                                                                       })
                                        try:
                                            schema.validate(df, lazy=True)
                                        except pa.errors.SchemaErrors as err:
                                            m = err.failure_cases
                                            c.append(len(err.failure_cases))
                                            hig(m,c,df,i)
                                        if len(c)==0:
                                            st.write("There are ",0,"rows out of range")
                                       
                                            
                                    
         
    
    else:
        st.error("Please select your data to started")

main()