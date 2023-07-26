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
def mento(data,f,i):
    y,x,s,m = smp.symbols("x s m y")
    fs = smp.integrate(f,(x,0,y)).doit()
    Fn = smp.lambdify((y,s,m),fs)
    fn = smp.lambdify((x,s,m),f)
    s=data[i].std()
    m=(data[i]).mean()
    x = np.linspace(min(data[i]),max(data[i]),len(data[i]))
    f = fn(x,s,m)
    F = Fn(x,s,m)
    us = np.random.rand(len(data))
    F_inv = x[np.searchsorted(F[:-1],us)]
    return F_inv
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
    if len(c)==0:
        st.write("There is ",len(c),"rows out of range")
    else:
        st.write("There is ",c[0],"rows out of range")
    def highlight(s):
        if s[i] in list(w):
            return ['background-color: yellow'] * len(s)
        else:
            return ['background-color: white'] * len(s)
    with col2:
        openn = df.style.apply(highlight, axis=1)
        st.write(openn)
def main():
    df = None
    with st.sidebar.header("Source Data Selection"):
        selection = ["csv",'excel']
        selected_data = st.sidebar.selectbox("Please select your dataset fromat:",selection)
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
                   
        
       
    
    #if source_data is not None:
        #df = read_csv(source_data)
        st.header("Dataset")
        
    if df is not None:
        user_choices = ['Dataset Sample',"Data Quality",'Data Prediction',"Data Validation"]
        selected_choices = st.sidebar.selectbox("Please select your choice:",user_choices)
        dff = df.copy()
        st.sidebar.write("Would you consider 0 as missing value?",key=f"MyKey{13}")
        y =  st.sidebar.checkbox("Yes",key=f"MyKey{223}")
        n =  st.sidebar.checkbox("No",key=f"MyKey{333}")
        word = []
        if y:
            st.sidebar.write("Is there any columns for classification in the dataset?",key=f"MyKey{23}")
            Y = st.sidebar.checkbox("Yes")
            N = st.sidebar.checkbox("No")
            select = df.keys()
            if Y:
                st.sidebar.write("Which columns are classification columns?",key=f"MyKey{33}")
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
            st.sidebar.write("Is there any columns for classification in the dataset?",key=f"MyKey{23}")
            Y = st.sidebar.checkbox("Yes")
            N = st.sidebar.checkbox("No")
            select = df.keys()
            if Y:
                st.sidebar.write("Which columns are classification columns?",key=f"MyKey{33}")
                for i in select:
                    X = st.sidebar.checkbox(i)
                    if X:
                        word.append(i)
        if selected_choices is not None:
            if selected_choices == "Dataset Sample":        
                st.info("Select dataset has "+str(df.shape[0])+" rows and "+str(df.shape[1])+" columns.")
                st.write(df) 
                
             
            elif selected_choices == "Data Prediction":
                choices = ['Ordinary Least Squares','Monte Carlo Simulation','interpolation']
                old_val = st.sidebar.selectbox(" ",choices,key=f"MyKey{1}")
                if old_val == "Ordinary Least Squares": 
                    st.markdown("Ordinary Least Squares")
                  
                    select = df.keys()
                    selection = st.selectbox("Please select which column you want to do the prediction",select,key=f"MyKey{2}")
                    if selection is not None:
                        for i in select:
                            box = df.select_dtypes(include=['int',"float"])
                            if selection == i and df.equals(box): 
                                data = OLS(df,i)
                                select_data1 = {i:data,"index":np.arange(len(data)),"color":"OLS"}
                                select_data1 = pd.DataFrame(select_data1)
                                select_data2 = {i:df[i],"index":np.arange(len(data)),"color":"Real"}
                                select_data2 = pd.DataFrame(select_data2)
                                select_data = {"OLS":data,"real":df[i]}
                                select_data = pd.DataFrame(select_data)
                                data1 = [select_data2,select_data1]
                                result = pd.concat(data1)
                                base = alt.Chart(result).mark_rule().encode( 
                                    x=alt.X('index', axis=alt.Axis( )),
                                    y=alt.Y(i,axis=alt.Axis( )),
                                    color = "color").properties(
                                    width=500,
                                    height=400,   
                                    ).interactive()
                                fig = px.scatter(result,x="index",y=i,color="color")
                                score = pearsonr(select_data["OLS"],select_data["real"])
                                tab1,tab2 = st.tabs(["Scatter plot theme", "Histogram theme"])
                                with tab1:
                                    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                                with tab2:
                                    st.altair_chart(base, use_container_width=True)
                                
                                with st.expander("See the OLS and Real data"):
                                    st.write(select_data)
                                st.write("The correlation coefficient between Real and OLS is",score[0])
                                st.write("The corresponding p-value is",score[1])
                            elif selection == i:
                                st.error("OLS only work with dataset which only include int and float data")
                                
                elif old_val == "interpolation":
                    select = df.keys()
                    selection = st.selectbox("Please select which column you want to do the prediction",select,key=f"MyKey{3}")
                    for i in select:
                        box = df.select_dtypes(include=['int',"float"])
                        df.replace(np.nan,0,inplace=True)
                        if selection==i and i in box :
                            df_new = df[1:]
                            df_new = df_new[:-1]
                            train,test=train_test_split(df_new,test_size=0.25,train_size=0.75)
                            bb = df.loc[df.index==0]
                            train = pd.concat([bb,train])
                            train.loc[max(df.index),:]=df.iloc[max(df.index)]
                            x = np.array(train.index)
                            y = np.array(train[i])
                            f = interp1d(x,y,kind="cubic")
                            new_value = []
                            for iii in test.index:
                                new_value.append(f(iii))
                            select_data1 = {i:new_value,"index":np.arange(len(test.index)),"color":"interp1d"}
                            select_data1 = pd.DataFrame(select_data1)
                            select_data2 = {i:list(test[i]),"index":np.arange(len(test.index)),"color":"Real"}
                            select_data2 = pd.DataFrame(select_data2)
                            select_data = {"interp1d":list(map(float, new_value)),"real":list(map(float, test[i]))}
                            select_data = pd.DataFrame(select_data)
                            data1 = [select_data2,select_data1]
                            result = pd.concat(data1)
                            score = pearsonr(select_data["interp1d"],select_data["real"])
                            fig = px.scatter(result,x="index",y=i,color="color")
                            tab1,tab2 = st.tabs(["Scatter plot theme", "Line chart theme"])
                            with tab1:
                                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                            with tab2:
                                st.line_chart(select_data)
                                #st.plotly_chart(fig, use_container_width=True)
                                #st.altair_chart(base, use_container_width=True)
                                
                            with st.expander("See the interpolation and Real data"):
                                st.write(select_data)
                            st.write("The correlation coefficient between Real and interpolation is",score[0])
                            st.write("The corresponding p-value is",score[1])
                            
                elif old_val == "Monte Carlo Simulation":
                    select = df.keys()
                    selection = st.selectbox("Please select which column you want to do the prediction",select,key=f"MyKey{3}")
                    for i in select:
                        df.replace(np.nan,0,inplace=True)
                        box = df.select_dtypes(include=['int',"float"])
                        if selection==i and i in box :
                            y,x,s,m = smp.symbols("x s m y")
                            f = 1/(s*(np.pi*2)**(1/2))*smp.exp(-(x-m)**2/(2*s**2))
                            data = mento(df,f,i)
                            select_data1 = {i:data,"index":np.arange(len(data)),"color":"Monte Carlo"}
                            select_data1 = pd.DataFrame(select_data1)
                            select_data2 = {i:df[i],"index":np.arange(len(data)),"color":"Real"}
                            select_data2 = pd.DataFrame(select_data2)
                            select_data = {"Monte Carlo":data,"real":df[i]}
                            select_data = pd.DataFrame(select_data)
                            data1 = [select_data2,select_data1]
                            result = pd.concat(data1)
                            hist_data = [select_data["Monte Carlo"], select_data["real"]]
                            group_labels = ['Monte Carlo', 'Real']
                            from plotly.figure_factory import create_distplot
                            figs = create_distplot(
                            hist_data, group_labels, bin_size=[0.2, .25, .5])
                            
                            
                            
                            base = alt.Chart(result).mark_rule().encode( 
                            x=alt.X('index', axis=alt.Axis( )),
                            y=alt.Y(i,axis=alt.Axis( )),
                            color = "color").properties(
                                    width=500,
                                    height=400,   
                                    ).interactive()
                            fig = px.scatter(result,x="index",y=i,color="color")
                                #figs = px.bar(result,x="index",y=i,color="color")
                            score = pearsonr(select_data["Monte Carlo"],select_data["real"])
                                #ax.legend("a","b")
                            tab1,tab2 = st.tabs(["Scatter plot theme", "Distrubtion theme"])
                            with tab1:
                                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                            with tab2:
                                st.plotly_chart(figs, use_container_width=True)
                                #st.altair_chart(base, use_container_width=True)
                                
                            with st.expander("See the Monte Carlo and Real data"):
                                st.write(select_data)
                            st.write("The correlation coefficient between Real and Monte Carlo is",score[0])
                            st.write("The corresponding p-value is",score[1])
                            
                
             
            elif  selected_choices == "Data Quality":
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
                        types = pd.DataFrame(df.describe()).T
                        
                        a = types.astype(str)
                        st.table(a)
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
                    elif selection == "Duplicate records":
                        types = df[df.duplicated()]
                        
                        a = types.astype(str)
                        st.write("The number of duplicated rows is ",len(types))
                        st.write(a)
                        
                    elif selection == "Outliers":
                        fig = plt.figure(figsize=(15,20))
                        box = df.select_dtypes(include=['int',"float"])
                        for i in range(len(box.keys())):
                            plt.subplot(len(box.keys()),1,i+1)
                            sns.boxplot(df[box.keys()[i]])
                            plt.xlabel(box.keys()[i],fontsize=18)  
                        fig.tight_layout()
                        st.pyplot(fig)
                    elif selection == "Data distribution":
                        boxs= df.select_dtypes(include=['int',"float"])
                        box = df.keys()
                        se = st.selectbox("Select which column you want to check",box,key=f"MyKey{6}")
                        for i in boxs:
                            if se == i and se not in word:
                                tab1,tab2,tab3 = st.tabs(["Hist_chart","Scatter_chart","Line_chart"])
                                with tab1:
                                    fig = plt.figure(figsize=(4,3))
                                    if word != []:
                                        ss = st.selectbox("What classification condition do you want?",word,key=f"MyKey{7}")
                                        for j in word: 
                                            if ss == j: 
                                                sns.histplot(data = df,x=i,binwidth=3,kde=True,hue=j)
                                                st.pyplot(fig)
                                    elif word ==[]:
                                        sns.histplot(data = df,x=i,binwidth=3,kde=True)
                                        st.pyplot(fig)
                                with tab2:
                                    fig = plt.figure(figsize=(4,3))
                                    df["counts"]=np.arange(len(df))
                                    if word != []:
                                        ss = st.selectbox("What classification condition do you want?",word,key=f"MyKey{8}")
                                        for j in word: 
                                            if ss == j: 
                                                sns.scatterplot(data = df,x="counts",y=i,hue=j)
                                                st.pyplot(fig)
                                    elif word ==[]:
                                        sns.scatterplot(data = df,x= "counts",y=i)
                                        st.pyplot(fig)
                                with tab3:
                                    fig = plt.figure(figsize=(4,3))
                                    df["counts"]=np.arange(len(df))
                                    if word != []:
                                        ss = st.selectbox("What classification condition do you want?",word,key=f"MyKey{9}")
                                        for j in word: 
                                            if ss == j: 
                                                sns.lineplot(data = df,x="counts",y=i,hue=j)
                                                st.pyplot(fig)
                                    elif word ==[]:
                                        sns.lineplot(data = df,x= "counts",y=i)
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
                                st.write("What type of data is this?")
                                agree = st.checkbox('Int/Float')
                                c = []
                                if agree:
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

                                agrees = st.checkbox('Str')
                                if agrees:
                                    st.write("What is the range of your data?")
                                    minimum = st.number_input('Insert the minimum length of your string')-1
                                    maximum = st.number_input('Insert the maximum length of your string')+1
                                    sele = ["numbers", "letters", "symbols"]
                                    o = st.multiselect("What is/are included in your column?",["numbers", "letters", "symbols"])
                                    if o==sele:
                                        if st.button("Error Result"):
                                            c = []
                                            schema = pa.DataFrameSchema({
                                           i: pa.Column(str, [pa.Check(lambda x: len(x) > minimum, element_wise=True),
                                                         pa.Check(lambda x: len(x) < maximum, element_wise=True)])})
                                            try:
                                                schema.validate(df, lazy=True)
                                            except pa.errors.SchemaErrors as err:
                                                m = err.failure_cases
                                                c.append(len(err.failure_cases))
                                                hig(m,c,df,i)
                                    elif o==["numbers","letters"]:
                                        if st.button("Error Result"):
                                            c = []
                                            schema = pa.DataFrameSchema({
                                           i: pa.Column(str, [pa.Check(lambda x: len(x) > minimum, element_wise=True),
                                                         pa.Check(lambda x: len(x) < maximum, element_wise=True),
                                                         pa.Check.str_matches(r'^[a-z0-9-]+$')])})  
                                            try:
                                                schema.validate(df, lazy=True)
                                            except pa.errors.SchemaErrors as err:
                                                m = err.failure_cases
                                                c.append(len(err.failure_cases))
                                                hig(m,c,df,i) 
                                    elif o == ["numbers"]:
                                        if st.button("Error Result"):
                                            c = []
                                            schema = pa.DataFrameSchema({
                                           i: pa.Column(str, [pa.Check(lambda x: len(x) > minimum, element_wise=True),
                                                         pa.Check(lambda x: len(x) < maximum, element_wise=True),
                                                         pa.Check.str_matches(r'^[0-9]+$')])})  
                                            try:
                                                schema.validate(df, lazy=True)
                                            except pa.errors.SchemaErrors as err:
                                                m = err.failure_cases
                                                c.append(len(err.failure_cases))
                                                hig(m,c,df,i)
                                    elif o == ["letters"]:
                                        if st.button("Error Result"):
                                            c = []
                                            schema = pa.DataFrameSchema({
                                           i: pa.Column(str, [pa.Check(lambda x: len(x) > minimum, element_wise=True),
                                                         pa.Check(lambda x: len(x) < maximum, element_wise=True),
                                                         pa.Check.str_matches(r'^[a-z]+$')])})  
                                            try:
                                                schema.validate(df, lazy=True)
                                            except pa.errors.SchemaErrors as err:
                                                    m = err.failure_cases
                                                    c.append(len(err.failure_cases))
                                                    hig(m,c,df,i)
                                   # elif o == ["numbers","symbols"]:
                                   #     if st.button("Error Result"):
                                   #         c = []
                                   #         schema = pa.DataFrameSchema({
                                   #        i: pa.Column(str, [pa.Check(lambda x: len(x) > minimum, element_wise=True),
                                   #                      pa.Check(lambda x: len(x) < maximum, element_wise=True),
                                   #                      pa.Check.str_matches(r'^[0-9]+$')])})  
                                   #         try:
                                   #            schema.validate(df, lazy=True)
                                   #         except pa.errors.SchemaErrors as err:
                                   #                 st.dataframe(err.failure_cases)
                                   #                 c.append(len(err.failure_cases))
                                   #         if len(c)==0:
                                   #             st.write("There is ",len(c),"rows out of range")
                                   #         else:
                                   #             st.write("There is ",c[0],"rows out of range")
                                   # elif o == ["letters","symbols"]:
                                   #     if st.button("Error Result"):
                                   #         c = []
                                   #         schema = pa.DataFrameSchema({
                                   #        i: pa.Column(str, [pa.Check(lambda x: len(x) > minimum, element_wise=True),
                                   #                      pa.Check(lambda x: len(x) < maximum, element_wise=True),
                                   #                      pa.Check.str_matches(r'^[a-z]+$')])})  
                                   #         try:
                                   #             schema.validate(df, lazy=True)
                                   #         except pa.errors.SchemaErrors as err:
                                   #                 st.dataframe(err.failure_cases)
                                   #                 c.append(len(err.failure_cases))
                                   #         if len(c)==0:
                                   #             st.write("There is ",len(c),"rows out of range")
                                   #         else:
                                   #             st.write("There is ",c[0],"rows out of range")
                                #agreess = st.checkbox('Float')
                                #if agreess:
                                #    a = float
                                #    st.write("What is the range of your data?")
                                ##    minimum = st.number_input('Insert the minimum value')
                                #   maximum = st.number_input('Insert the maximum value')
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
                                st.write("Are all values in the column unique？") 
                                Y = st.checkbox("Yes",key=f"MyKey{32}")
                                N = st.checkbox("No",key=f"MyKey{42}")
                                if Y:
                                    st.write("In",i,"there are ",len(df[df[i].duplicated()]),"rows with repeated values.")
                                    def repeat(s):
                                         if s[i] in list(df[i][df[i].duplicated()]):
                                            return ['background-color: yellow'] * len(s)
                                         else:
                                            return ['background-color: white'] * len(s)
                                    fig = plt.figure(figsize=(4,3))
                                    st.table(df.style.apply(repeat, axis=1)).head()
                                    st.pyplot(fig)
                                    #st.write(df[i][df[i].duplicated()])
                                elif N:
                                    st.write(":full_moon_with_face:")
                                t = st.text_input("If you want, type the Regular Expression of your data")
                                if st.button("Error Result"):
                                            c = []
                                            schema = pa.DataFrameSchema({
                                           i: pa.Column(str, 
                                                         pa.Check.str_matches(t))})  
                                            try:
                                                schema.validate(df, lazy=True)
                                            except pa.errors.SchemaErrors as err:
                                                    m = err.failure_cases
                                                    c.append(len(err.failure_cases))
                                                    hig(m,c,df,i)
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
                                    if st.button("Error Result"):
                                            schema = pa.DataFrameSchema({
                                             i: pa.Column(int,pa.Check.isin(bar), coerce=True)                                                                                                                       })
                                            try:
                                                schema.validate(df, lazy=True)
                                            except pa.errors.SchemaErrors as err: 
                                                m = err.failure_cases
                                                c.append(len(err.failure_cases))
                                                hig(m,c,df,i) 
                                elif options == "text":
                                    barr = []
                                    c = []
                                    number = st.number_input('How many categories are there？')
                                    for ii in range(int(number)):
                                        text = st.text_input("insert name of category",key=f"MyKey{ii}")
                                        barr.append(text)
                                    if st.button("Error Result"):
                                        schema = pa.DataFrameSchema({
                                         i: pa.Column(str,pa.Check.isin(barr), coerce=True)                                                                                                                       })
                                        try:
                                            schema.validate(df, lazy=True)
                                        except pa.errors.SchemaErrors as err:
                                            m = err.failure_cases
                                            c.append(len(err.failure_cases))
                                            hig(m,c,df,i)
                                            



                                   
                            
                                    
                                
                            
                            
#                            box = ["Required fields","Data types","String length","Numeric ranges","Categorical #values","Date/time constraints","Pattern matching","Unique values","Dependencies between columns","Custom validation functions"]
#                            sr = st.selectbox("Data Validation Selection",box,key=f"MyKey{11}")
#                            if sr is not None:
#                                if sr == "Required fields":
                                     
         
    
    else:
        st.error("Please select your data to started")

main()