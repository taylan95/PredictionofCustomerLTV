import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
pd.options.display.max_columns = None
from scipy import stats
from scipy.stats import norm, skew
import datetime as dt
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import squarify
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
from xgboost import XGBRegressor
from sklearn import metrics
from math import sqrt
from sklearn.model_selection import cross_val_score
from feature_engine.outliers import Winsorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, RANSACRegressor, ElasticNet, SGDRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam


customerdata = pd.read_csv(r"C:\Users\taylan.polat\Desktop\Python_Codes\Aifinal\marketingcustomeranalytics.csv")
df = customerdata.copy()

figure = plt.figure(figsize=(18,10))
plt.subplot(1,2,1)
sns.distplot(df["Total Claim Amount"] , fit=norm);
(mu, sigma) = norm.fit(df["Total Claim Amount"])
plt.ylabel("Frequency")
plt.title("Claim Distribution")

plt.subplot(1,2,2)
stats.probplot(df["Total Claim Amount"], plot=plt)
plt.show()

locations = round((df[["Location Code","Months Since Last Claim","Months Since Policy Inception",
                           "Monthly Premium Auto"]].groupby(["Location Code"]).sum()),2).reset_index()

locations["Months Since Last Claim"] = 360*locations["Months Since Last Claim"]/locations["Months Since Last Claim"].sum()
locations["Months Since Policy Inception"] = 360*locations["Months Since Policy Inception"]/locations["Months Since Policy Inception"].sum()
locations["Monthly Premium Auto"] = 360*locations["Monthly Premium Auto"]/locations["Monthly Premium Auto"].sum()

fig = go.Figure(go.Barpolar(
    r=round(locations["Months Since Last Claim"],0),
    theta=round(locations["Months Since Policy Inception"],0),
    width=round(locations["Monthly Premium Auto"],0),
    marker_color=["#E4FF87", '#709BFF', '#FFAA70'],
    marker_line_color="black",
    marker_line_width=2,
    opacity=0.8
))

fig.update_layout(
    template=None,
    polar = dict(
        radialaxis = dict(range=[0, 5], showticklabels=False, ticks=''),
        angularaxis = dict(showticklabels=False, ticks='')
    )
)

fig.show()


Edu_amounts = round((df[["Education","Months Since Last Claim","Months Since Policy Inception",
                           "Monthly Premium Auto","Number of Open Complaints","Number of Policies"]].groupby(["Education"]).sum()),2).reset_index()

x=Edu_amounts["Education"]
fig = go.Figure(go.Bar(x=x, y=Edu_amounts["Months Since Last Claim"], name='Claim'))
fig.add_trace(go.Bar(x=x, y=Edu_amounts["Months Since Policy Inception"], name='Inception'))
fig.add_trace(go.Bar(x=x, y=Edu_amounts["Monthly Premium Auto"], name='Premium'))

fig.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'})
fig.show()

fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.boxplot(y='Total Claim Amount',x = 'Policy Type', hue = 'Policy Type',data = df, ax=axarr[0][0])
sns.boxplot(y='Total Claim Amount',x = 'Policy', hue = 'Policy',data = df , ax=axarr[0][1])
sns.boxplot(y='Total Claim Amount',x = 'Coverage', hue = 'Coverage',data = df, ax=axarr[1][0])
sns.boxplot(y='Total Claim Amount',x = 'Response', hue = 'Response',data = df, ax=axarr[1][1])

df["Effective To Date"] = df["Effective To Date"].astype("datetime64[ns]")
StatewithIncome = round(df[["Total Claim Amount","State","Income"]].groupby(["State"]).sum().reset_index(),2)

StatewithIncomeasc = StatewithIncome.sort_values("Income", ascending=False)
fig=px.pie(StatewithIncomeasc, "State", "Income", hover_data=['Total Claim Amount'], title="StatewithIncome")
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()

DatewithIncome = round(df[["Customer Lifetime Value","Effective To Date","Income"]].groupby(["Effective To Date"]).mean().reset_index(),2)

trace0 = go.Bar(
    x = DatewithIncome["Effective To Date"],
    y = DatewithIncome["Customer Lifetime Value"],
    name = "Average CLF")
trace1 = go.Scatter(
    x = DatewithIncome["Effective To Date"],
    y = DatewithIncome["Income"],
    mode = "markers+lines",
    name = "Average Income")

d = [trace0,trace1]
layout = go.Layout(title="Avehome",barmode = "stack")
figure = go.Figure(data=d,layout = layout)
figure.show()

lastday = df["Effective To Date"].max()
rfmTable = df.groupby('Customer').agg({"Effective To Date": lambda x: (df["Effective To Date"].max() - x), # Recency
                                        "Policy Type": lambda x: len(x),      # Frequency
                                        "Total Claim Amount": lambda x: x.sum()}) # Monetary Value

rfmTable.rename(columns={"Effective To Date": "recency", 
                         "Policy Type": "frequency", 
                         "Total Claim Amount": 'monetary_value'}, inplace=True)

rfmTable["recency"] = (rfmTable["recency"] / np.timedelta64(1, 'D')).astype(int)
quantiles = rfmTable.quantile(q=[0.25,0.5,0.75])
rfmSegmentation = rfmTable

def RClass(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4

def FMClass(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1

rfmSegmentation['R_Quartile'] = rfmSegmentation['recency'].apply(RClass, args=('recency',quantiles,))
rfmSegmentation['F_Quartile'] = rfmSegmentation['frequency'].apply(FMClass, args=('frequency',quantiles,))
rfmSegmentation['M_Quartile'] = rfmSegmentation['monetary_value'].apply(FMClass, args=('monetary_value',quantiles,))

rfmSegmentation['RFMClass'] = rfmSegmentation.R_Quartile.map(str) \
                            + rfmSegmentation.F_Quartile.map(str) \
                            + rfmSegmentation.M_Quartile.map(str)
                            
rfmSegmentation["class"] = 0
rfmSegmentation=rfmSegmentation.reset_index()
for i in range(0,len(rfmSegmentation)):
    if int(rfmSegmentation.loc[i,"RFMClass"]) <= 150:
        rfmSegmentation.loc[i,"class"] = "Best Customers"
    elif int(rfmSegmentation.loc[i,"RFMClass"]) <= 250:
        rfmSegmentation.loc[i,"class"] = "Almost Best Customers"
    elif int(rfmSegmentation.loc[i,"RFMClass"]) <= 350:
        rfmSegmentation.loc[i,"class"] = "Almost Worse Customers"
    else:
        rfmSegmentation.loc[i,"class"] = "Worse Customers"
        
rfm_seg = rfmSegmentation[["class","frequency"]].groupby(["class"]).count().reset_index()
fig = px.treemap(rfm_seg,
                 path=["class"],
                 values=round((rfm_seg["frequency"]/sum(rfm_seg["frequency"])),3),color = "frequency")

fig.update_layout(title="Segmentation",
                  width=700, height=500,)

fig.show()

df.drop("Customer",axis = 1, inplace = True)
df.drop("Effective To Date",axis = 1, inplace = True)

wind = Winsorizer(capping_method = 'iqr',
                  tail = 'both',
                  fold = 1.5,
                  variables=['Customer Lifetime Value','Income','Total Claim Amount'])

wind.fit(df)
df = wind.transform(df)

dummylist = []

dummy_variables = ["State","Response","Coverage","Education","EmploymentStatus","Gender","Location Code",
                   "Policy Type","Policy","Renew Offer Type","Sales Channel","Vehicle Class","Vehicle Size","Marital Status"]
for var in dummy_variables:  
    dummylist.append(pd.get_dummies(df[var],prefix = var,prefix_sep="_",drop_first=True))
    dummies_collected = pd.concat(dummylist,axis = 1)
    
df.drop(dummy_variables,axis = 1,inplace = True)
df = pd.concat([df,dummies_collected],axis = 1)

rfc = RandomForestRegressor(n_estimators=200, n_jobs=-1, max_depth=6)
feat_selector = BorutaPy(rfc, n_estimators='auto', verbose=2)

X = df.drop("Total Claim Amount",axis = 1)
y = df["Total Claim Amount"]

sc = StandardScaler()
sc_X = sc.fit_transform(X)
sc_y = sc.fit_transform(np.array(y).reshape(-1,1))

sc_X = np.int64(sc_X)
sc_y = np.int64(sc_y)

feat_selector.fit(sc_X,sc_y)

feat_selected = list(feat_selector.ranking_)

col_names = X.columns
feat_ranked = {"col_names":col_names,
               "feat_rank":feat_selected}
feat_ranked = pd.DataFrame(feat_ranked)
feat_ranked.sort_values("feat_rank",ascending = True)

fin_feat = feat_ranked[feat_ranked["feat_rank"] < 2]
X = X[fin_feat["col_names"]]
sc_X = sc.fit_transform(X)
X = pd.DataFrame(sc_X,columns = fin_feat["col_names"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

class FeatureSelector:

    def __init__(self, X_train):
        self.X_train = X_train

    def get_correlation_matrix(self):
        corr_matrix = self.X_train.corr()
        fig, ax = plt.subplots(figsize=(25, 15))
        ax = sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt=".2f", cmap="summer_r")
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)

    @staticmethod
    def correlation(dataset, threshold):
        col_corr = set()
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    colname = corr_matrix.columns[i]
                    col_corr.add(colname)
                    
        return col_corr

    def get_corr_features_len(self):
        corr_features = self.correlation(self.X_train, 0.7)
        return len(set(corr_features))

    def get_constant_features_len(self):
        constant_features = [
                feat for feat in self.X_train.columns if self.X_train[feat].std() == 0
            ]
        return len(constant_features)

    def get_duplicated_feat_len(self):
        duplicated_feat = []
        for i in range(0, len(self.X_train.columns)):
            if i % 10 == 0:
                print(i)
            col_1 = self.X_train.columns[i]
            for col_2 in self.X_train.columns[i + 1:]:
                if self.X_train[col_1].equals(self.X_train[col_2]):
                    duplicated_feat.append(col_2)

        return len(set(duplicated_feat))

    def get_roc_values(self):
        roc_values = []
        for feature in self.X_train.columns:
            clf = RandomForestClassifier()
            clf.fit(self.X_train[feature].fillna(0).to_frame(), y_train)
            y_scored = clf.predict_proba(X_test[feature].fillna(0).to_frame())
            roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))

        roc_values = pd.Series(roc_values)
        roc_values.index = self.X_train.columns
        roc_values.sort_values(ascending=False)

        roc_values.sort_values(ascending=False).plot.bar(figsize=(20, 8))
        return roc_values
    
feature_selector = FeatureSelector(X_train)
feature_selector.get_correlation_matrix()

feature_selector.get_corr_features_len()
feature_selector.get_constant_features_len()
feature_selector.get_duplicated_feat_len()

corr_feature = feature_selector.correlation(X_train,0.7)

X_train.drop(corr_feature, axis = 1,inplace = True)
X_test.drop(corr_feature, axis = 1,inplace = True)

#Model Results

def cross_val(model):
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

def feat_importance(model,modelname):
    coefs = pd.Series(model.coef_, index = X_train.columns)
    imp_coefs = pd.concat([coefs.sort_values().head(10),
                         coefs.sort_values().tail(10)])
    plt.figure(figsize = (8,10))
    imp_coefs.plot(kind = "barh")
    plt.xlabel(f"{modelname} coefficient", weight='bold')
    plt.title(f"Feature importance in the {modelname} Model", weight='bold')
    plt.show()
    
#Linear Regression

lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train,y_train)

test_pred = lin_reg.predict(X_test)
train_pred = lin_reg.predict(X_train)


results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, test_pred) , cross_val(LinearRegression())]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df

coef_res = pd.DataFrame(lin_reg.coef_,columns = ["Linear Regression"],index = X_train.columns)
feat_importance(lin_reg,"Linear Regression")

#Ransac Regressor

model = RANSACRegressor(base_estimator=LinearRegression(), max_trials=100)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

results_df_2 = pd.DataFrame(data=[["Robust Regression", *evaluate(y_test, test_pred) , cross_val(RANSACRegressor())]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

#Ridge Regression

model = Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

results_df_2 = pd.DataFrame(data=[["Ridge Regression", *evaluate(y_test, test_pred) , cross_val(Ridge())]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

coef_res["Ridge"] = model.coef_
feat_importance(model,"Ridge Regression")

#Lasso Regression

model = Lasso(alpha=0.1, 
              precompute=True, 
               warm_start=True, 
              positive=True, 
              selection='random',
              random_state=42)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

results_df_2 = pd.DataFrame(data=[["Lasso Regression", *evaluate(y_test, test_pred) , cross_val(Lasso())]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

coef_res["Lasso"] = model.coef_
feat_importance(model,"Lasso Regression")

#Elastic Regression

model = ElasticNet(alpha=0.1, l1_ratio=0.9, selection='random', random_state=42)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

results_df_2 = pd.DataFrame(data=[["Elastic Net Regression", *evaluate(y_test, test_pred) , cross_val(ElasticNet())]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

coef_res["ElasticNet"] = model.coef_
feat_importance(model,"ElasticNet Regression")

#SGDRegression

sgd_reg = SGDRegressor(n_iter_no_change=250, penalty=None, eta0=0.0001, max_iter=100000)
sgd_reg.fit(X_train, y_train)

test_pred = sgd_reg.predict(X_test)
train_pred = sgd_reg.predict(X_train)

results_df_2 = pd.DataFrame(data=[["Stochastic Gradient Descent", *evaluate(y_test, test_pred), 0]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

coef_res["SGDRegressor"] = sgd_reg.coef_
feat_importance(sgd_reg,"SGDRegressor")

#Artificial Neural Network

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()

model.add(Dense(X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1))

model.compile(optimizer=Adam(0.00001), loss='mse')

r = model.fit(X_train, y_train,
              validation_data=(X_test,y_test),
              batch_size=1,
              epochs=100)

plt.figure(figsize=(10, 6))

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

results_df_2 = pd.DataFrame(data=[["Artficial Neural Network", *evaluate(y_test, test_pred), 0]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

#Random Forest Regression

rf_reg = RandomForestRegressor(n_estimators=1000)
rf_reg.fit(X_train, y_train)

test_pred = rf_reg.predict(X_test)
train_pred = rf_reg.predict(X_train)

results_df_2 = pd.DataFrame(data=[["Random Forest Regressor", *evaluate(y_test, test_pred), 0]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

#Xgboost Regression

model = XGBRegressor(learning_rate=0.125, n_estimators=100,
                      max_depth=14, min_child_weight=0.01,
                      gamma=0, subsample=0.6,max_leaves = 3,
                      colsample_bytree=1,booster="gbtree",
                      objective='reg:squarederror', nthread=-1,
                      scale_pos_weight=0.2, seed=27,alpha = 1,
                      reg_alpha=0.0000001)

model.fit(X_train, y_train)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

results_df_2 = pd.DataFrame(data=[["Xgboost Regressor", *evaluate(y_test, test_pred), 0]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

#Model Results

model = XGBRegressor(learning_rate=0.125, n_estimators=100,
                      max_depth=14, min_child_weight=0.01,
                      gamma=0, subsample=0.6,max_leaves = 3,
                      colsample_bytree=1,booster="gbtree",
                      objective='reg:squarederror', nthread=-1,
                      scale_pos_weight=0.2, seed=27,alpha = 1,
                      reg_alpha=0.0000001)

model.fit(X_train, y_train)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

results_df_2 = pd.DataFrame(data=[["Xgboost Regressor", *evaluate(y_test, test_pred), 0]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

#Model Results

results_df.set_index('Model', inplace=True)
results_df['R2 Square'].plot(kind='barh', figsize=(12, 8))

results_df

#Convert to CSV Files

results_df.to_csv(r"C:\Users\taylan.polat\Desktop\finalpython\modelresults.csv",decimal = ",",encoding = "utf-8",sep = ";",index=True)
coef_res.to_csv(r"C:\Users\taylan.polat\Desktop\finalpython\importancesresults.csv",decimal = ",",sep = ";",index=True)
customerdata.to_csv(r"C:\Users\taylan.polat\Desktop\finalpython\customerdataconverted.csv",decimal = ",",encoding = "utf-8",sep = ";",index=False)

