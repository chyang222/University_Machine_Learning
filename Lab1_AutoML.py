import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings('ignore')

col = ["Sample code number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", 
       "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]

df = pd.read_csv("/Users/unixking/Desktop/머신러닝/breast-cancer-wisconsin.txt", delimiter= ',', names = col)

df["Bare Nuclei"] = df["Bare Nuclei"].apply(lambda x: np.nan if x.isdigit() == False else x)

df = df.dropna()

df["Class"] = df["Class"].apply(lambda x: 0 if x == 2 else 1)

df = df.astype('int')

#----------- Plot correlation between target and features.----------------

fig, ax = plt.subplots(figsize=(125,125))
sns.heatmap(df.corr(), annot=True, annot_kws={'size':40}, fmt='0.2f', cmap="YlGnBu", linewidths=0.5)
ax.tick_params(axis='both', labelsize=50);
plt.show()

df = df.drop(['Sample code number'], axis = 1)

x = df[['Clump Thickness', 'Uniformity of Cell Size',
       'Uniformity of Cell Shape', 'Marginal Adhesion',
       'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
       'Normal Nucleoli', 'Mitoses']]


y = df[["Class"]]

def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()

#----------------------------single main fuction------------------------
def bestmodel(feature, target, scalers = None, models = None):
    x = feature
    y = target

    if scalers == None:
        scale = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
    else: 
        scale = scalers

    if models == None:
        entro = DT(criterion="entropy")
        gin = DT(criterion="gini")
        lo = LogisticRegression()
        sv = SVC()
        classifier = [entro, gin, lo, sv]
    else: 
        classifier = models
    
    ent_scale, ent_par, ent_acc, ent_fper, ent_tper = [], [], [], [], []
    ent_df = pd.DataFrame()

    gini_scale, gini_par, gini_acc, gini_fper, gini_tper = [], [], [], [], []
    gini_df = pd.DataFrame()
    
    log_scale, log_par, log_acc,log_fper, log_tper = [], [], [], [], []
    log_df = pd.DataFrame()
    
    svc_scale, svc_par, svc_acc, svc_fper, svc_tper = [], [], [], [], []
    svc_df = pd.DataFrame()
    
    ent_param = {'max_depth':[1,2,3], 'min_samples_split':[2,3]}
    gini_param = {'max_depth':[1,2,3], 'min_samples_split':[2,3]}
    log_param = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2',"elasticnet","none"]}
    svc_param = [{'kernel':['linear'],'C': [0.001, 0.01, 0.1, 1, 10],'gamma':[0.01, 0.05, 0.1, 0.5, 1, 3, 7]},
                {'kernel':['poly'],'C': [0.001, 0.01, 0.1, 1, 10],'gamma':[0.01, 0.05, 0.1, 0.5, 1, 3, 7]},
                {'kernel':['sigmoid'],'C': [0.001, 0.01, 0.1, 1, 10],'gamma':[0.01, 0.05, 0.1, 0.5, 1, 3, 7]},
                {'kernel':['rbf'],'C': [0.001, 0.01, 0.1, 1, 10],'gamma':[0.01, 0.05, 0.1, 0.5, 1, 3, 7]}]


    for model in classifier:
        for i in scale:
            scaler = i
            scaler = pd.DataFrame(scaler.fit_transform(x))
            x_train, x_test, y_train, y_test = train_test_split(scaler, y, test_size=0.2, shuffle=False)
            if model == entro:
                ent_grid = GridSearchCV(model, param_grid=ent_param, cv=3)
                ent_grid.fit(x_train, y_train)
                pred = ent_grid.predict(x_test)
                ent_scale.append(i)
                ent_par.append(ent_grid.best_params_)
                ent_acc.append( accuracy_score(y_test,pred) )
                ent_fper.append(roc_curve(y_test, pred)[0])
                ent_tper.append(roc_curve(y_test, pred)[1])
            
            
            elif model == gin:
                gini_grid = GridSearchCV(model, param_grid=gini_param, cv=3)
                gini_grid.fit(x_train, y_train)
                pred = gini_grid.predict(x_test)
                gini_scale.append(i)
                gini_par.append(gini_grid.best_params_)
                gini_acc.append(accuracy_score(y_test,pred))
                pred = gini_grid.predict(x_test)
    
                gini_fper.append(roc_curve(y_test, pred)[0])
                gini_tper.append(roc_curve(y_test, pred)[1])
            
                


            elif model == lo:
                log_grid = GridSearchCV(model, param_grid=log_param, cv=3)
                log_grid.fit(x_train, y_train)
                pred = log_grid.predict(x_test)
                log_scale.append(i)
                log_par.append(log_grid.best_params_)
                log_acc.append(accuracy_score(y_test,pred))
                pred = log_grid.predict(x_test)
                log_fper.append(roc_curve(y_test, pred)[0])
                log_tper.append(roc_curve(y_test, pred)[1])
            


            elif model == sv:
                svc_grid = GridSearchCV(model, param_grid=svc_param, cv=3)
                svc_grid.fit(x_train, y_train)
                pred = svc_grid.predict(x_test)
                svc_scale.append(i)
                svc_par.append(svc_grid.best_params_)
                svc_acc.append(accuracy_score(y_test,pred))
                pred = svc_grid.predict(x_test)
                svc_fper.append(roc_curve(y_test, pred)[0])
                svc_tper.append(roc_curve(y_test, pred)[1])
            

    ent_df["sclae"], ent_df["params"], ent_df["accurancy"], ent_df["fper"], ent_df["tper"] = ent_scale, ent_par, ent_acc, ent_fper, ent_tper

    gini_df["sclae"], gini_df["params"], gini_df["accurancy"], gini_df["fper"], gini_df["tper"] = gini_scale, gini_par, gini_acc, gini_fper, gini_tper

    log_df["sclae"], log_df["params"], log_df["accurancy"], log_df["fper"], log_df["tper"] = log_scale, log_par, log_acc, log_fper, log_tper

    svc_df["sclae"], svc_df["params"], svc_df["accurancy"], svc_df["fper"], svc_df["tper"] = svc_scale, svc_par, svc_acc, svc_fper, svc_tper


    ent_df = ent_df.sort_values(by=["accurancy"], ascending=[False]).iloc[0]
    gini_df = gini_df.sort_values(by=["accurancy"], ascending=[False]).iloc[0]
    log_df = log_df.sort_values(by=["accurancy"], ascending=[False]).iloc[0]
    svc_df = svc_df.sort_values(by=["accurancy"], ascending=[False]).iloc[0]

    print("Decision Tree (Entropy):\n", ent_df, "\n")
    print("Decision Tree (GINI):\n", gini_df, "\n")
    print("Logistic Regression:\n", log_df, "\n")
    print("Support Vector Machine:\n", svc_df, "\n")
    
    plt.figure(figsize=(15, 15))

    fpr = [ent_df[3], gini_df[3], log_df[3], svc_df[3]]
    tpr = [ent_df[4], gini_df[4], log_df[4], svc_df[4]]
    
    kind = ["entropy", "gini", "logistic", "SVM"]
    for i in range(1,5):
        plt.subplot(220 + i)
        plt.plot(fpr[i-1], tpr[i-1], color='red', label='ROC')
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC CURVE of {}'.format(kind[i-1]))
        plt.legend()

    plt.tight_layout()
    plt.show()

    ent_best, gini_best, log_best, svc_best = 0, 0, 0, 0
    ent_k, gini_k, log_k, svc_k = 0, 0, 0, 0

    for k in range(3,11):
        cv1=KFold(n_splits=k)

        ent_grid = GridSearchCV(entro, param_grid=ent_df[1], cv=cv1)
        scaler1 = pd.DataFrame(ent_df[1].fit_transform(x))
        x_train, x_test, y_train, y_test = train_test_split(scaler1, y, test_size=0.2, shuffle=False)
        ent_grid.fit(x_train,y_train)
        compare = ent_grid.best_score_
        if compare > ent_best:
            ent_k = k
            ent_best = compare

        
        gini_grid = GridSearchCV(gin, param_grid=gini_df[1], cv=cv1)
        scaler2 = pd.DataFrame(gini_df[1].fit_transform(x))
        x_train, x_test, y_train, y_test = train_test_split(scaler2, y, test_size=0.2, shuffle=False)
        gini_grid.fit(x_train,y_train)
        compare = gini_grid.best_score_
        if compare > gini_best:
            gini_k = k
            gini_best = compare

        log_grid = GridSearchCV(lo, param_grid=log_df[1], cv=cv1)
        scaler3 = pd.DataFrame(log_df[1].fit_transform(x))
        x_train, x_test, y_train, y_test = train_test_split(scaler3, y, test_size=0.2, shuffle=False)
        log_grid.fit(x_train,y_train)
        compare = log_grid.best_score_
        if compare > log_best:
            log_k = k
            log_best = compare
        #list(d.items())
        svc_grid = GridSearchCV(sv, param_grid=svc_df[1], cv=cv1)
        scaler4 = pd.DataFrame(svc_df[1].fit_transform(x))
        x_train, x_test, y_train, y_test = train_test_split(scaler4, y, test_size=0.2, shuffle=False)
        svc_grid.fit(x_train,y_train)
        compare = svc_grid.best_score_
        if compare > svc_best:
            svc_k = k
            svc_best = compare
        
        print("Decistion model (entropy) best k of k-fold : {}".foramt(ent_k))
        print("Decistion model (gini) best k of k-fold : {}".foramt(gini_k))
        print("Logisctic model best k of k-fold : {}".foramt(log_k))
        print("SVM model best k of k-fold : {}".foramt(svc_k))

    
    ##k-fold k값 for문으로 돌려서 k-fold의 최고값으로 찾아서 최적의 k값 반환하기!
    ##https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=gustn3964&logNo=221431933811