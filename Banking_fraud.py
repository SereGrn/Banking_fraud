# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 13:25:26 2022

@author: seren
"""

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA 
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE 
from dash import Dash, html, dcc, Output, Input, callback, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px 
import plotly.figure_factory as ff
from datetime import date

def score_classifier(dataset,labels,classifier): 
    
    """ 
    Train a classifier by maximizing recall, using a cross-validation by 3Fold.
    Displays the confusion matrix
    @param dataset : Array type, dataset without the target 
    @param labels : Array type, list of target per observation
    @param classifier : Function type, classifier to use 
    
    """
    
    kf = KFold(n_splits=3,random_state=50,shuffle=True)
    confusion_mat = np.zeros((2,2))
    recall = 0
    for training_ids,test_ids in kf.split(dataset):
        training_set = dataset[training_ids]
        training_labels = labels[training_ids]
        test_set = dataset[test_ids]
        test_labels = labels[test_ids]
        classifier.fit(training_set,training_labels)
        predicted_labels = classifier.predict(test_set)
        confusion_mat+=confusion_matrix(test_labels,predicted_labels)
        recall += recall_score(test_labels, predicted_labels)
        
    recall/=3

    print(f"recall : {recall}")
    
    sns.heatmap(confusion_mat, annot=True)
    plt.show()
    print(confusion_mat)
    return recall


df = pd.read_csv('..\autorisations.csv')

# =============================================================================
# Pre processing
# =============================================================================

df.dtypes

df.isna().sum()

# outliers
df_boxplot = df.drop(columns = ['Carte','fraude','Date','Heure','dateheure','CodeRep','MCC','Pays'])
green_diamond = dict(markerfacecolor='g', marker='D')

for i in df_boxplot.columns:
    plt.boxplot(df[i],flierprops = green_diamond)
    plt.title("BoxPlot of " + i)
    plt.show()

# Many extreme values in the data, a robust method for normalization should be used

# One very extreme value for the variable "Montant" : probably due to a typing error 
outliers = df[df['Montant'] == df['Montant'].max() ]

# It seems that the decimal point has been forgotten, the amounts must be divided by 10. But in case of errors, we prefer to simply delete the corresponding observation 
df = df[df['Montant'] != df['Montant'].max() ]


# check correlation 
df_corr = df_boxplot.corr()
sns.heatmap(df_corr)
df_corr = df_corr[df_corr.iloc[:,:] >= 0.8]

# The additional indicators calculated by the creators of the database are highly correlated: a PCA is applied to each group of variables 



df_PCA1 =  df[['FM_Velocity_Condition_3', 'FM_Velocity_Condition_6',
'FM_Velocity_Condition_12', 'FM_Velocity_Condition_24']]
scaler = RobustScaler()
df_PCA1 = scaler.fit_transform(df_PCA1)

pca = PCA()
pca = pca.fit(df_PCA1)
pca.explained_variance_ratio_
# 1st principal component explains 88.97% of the variance 

pca = PCA( 1)
df_pca1 = pd.DataFrame(pca.fit_transform(df_PCA1), columns = ['Velocity'])

##

df_PCA2 =  df[['FM_Sum_3','FM_Sum_6', 'FM_Sum_12', 'FM_Sum_24']]
scaler = RobustScaler() 
df_PCA2 = scaler.fit_transform(df_PCA2)

pca = PCA()
pca = pca.fit(df_PCA2)
pca.explained_variance_ratio_
# 1st principal component explains 87.27% of the variance 

pca = PCA( 1)
df_pca2 = pd.DataFrame(pca.fit_transform(df_PCA2), columns = ['Sum'])

##

df_PCA3 =  df[['FM_Redondance_MCC_3','FM_Redondance_MCC_6', 'FM_Redondance_MCC_12', 'FM_Redondance_MCC_24']]
scaler = RobustScaler() 
df_PCA3 = scaler.fit_transform(df_PCA3)

pca = PCA()
pca = pca.fit(df_PCA3)
pca.explained_variance_ratio_
# 1st principal component explains 90.03% of the variance 

pca = PCA( 1)
df_pca3 = pd.DataFrame(pca.fit_transform(df_PCA3), columns = ['redundancy'])

##

df_PCA4 =  df[[ 'FM_Difference_Pays_3', 'FM_Difference_Pays_6', 'FM_Difference_Pays_12','FM_Difference_Pays_24']]
scaler = RobustScaler() 
df_PCA4 = scaler.fit_transform(df_PCA4)

pca = PCA()
pca = pca.fit(df_PCA4)
pca.explained_variance_ratio_
# 88.43 %

pca = PCA( 1)
df_pca4 = pd.DataFrame(pca.fit_transform(df_PCA4) , columns = ['Difference_country'])

df_pca = pd.concat([df_pca1,df_pca2,df_pca3,df_pca4], axis = 1)


# we delete the date and time variables, which cannot be understood by an algorithm. Instead, we create the variables "Day", "Month" and "Time" from the date. 
# We also create a binary variable that is worth 1 if the transaction is made at night (between 00:00 and 5:00 am), because this can be a sign of bank fraud. 

df['Month'] = df['Date'].str[3:5].astype("int32")
df['Day'] = df['Date'].str[0:2].astype("int32")

df['Time'] = df['Heure'].str[0:2].astype("int32")

df['Night'] = df['Time'].apply(lambda x : 1 if x in range(0,6) else 0)

df.drop(columns = ['Date','Heure','dateheure'], inplace = True)

# The variable "CodeRep" is 0 if the transaction was accepted, and all other codes mean that the transaction was rejected. We then recode this variable by 1 = transaction accepted, 0 = transaction refused. 

df['CodeRep'] = df['CodeRep'].apply(lambda x : 1 if x == 0 else 0)



# check balance of the target 
count_classes = pd.value_counts(df['fraude'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.xticks(range(len(df['fraude'].unique())), df.fraude.unique())
plt.title("Count of the modalities of the variable fraude")
plt.xlabel("1 = fraud, 0 = no fraud")
plt.ylabel("Number of observations");

(df['fraude'].value_counts()/len(df))*100
# 99% of transactions are not fraudulent !! 
# We are in the presence of a very unbalanced database 


df_model = df.drop(columns = ['FM_Difference_Pays_3', 'FM_Difference_Pays_6', 'FM_Difference_Pays_12','FM_Difference_Pays_24',
                    'FM_Redondance_MCC_3','FM_Redondance_MCC_6', 'FM_Redondance_MCC_12', 'FM_Redondance_MCC_24',
                    'FM_Sum_3','FM_Sum_6', 'FM_Sum_12', 'FM_Sum_24',
                    'FM_Velocity_Condition_3', 'FM_Velocity_Condition_6','FM_Velocity_Condition_12', 'FM_Velocity_Condition_24'])

    
# =============================================================================
# Modeling
# =============================================================================

labels = df_model['fraude'].values # labels
paramset = df_model.drop(['fraude','Carte'],axis=1).columns.values
df_vals = df_model.drop(['fraude','Carte'],axis=1)

# normalize dataset
X = pd.DataFrame(RobustScaler().fit_transform(df_vals.values)) # robust outlier standardization

X = pd.concat([X,df_pca], axis = 1)
X = X.values

 

## 1st modeling : testing some classifier with clas weight balanced, mean that  weights are inversely proportional to class frequencies 

# models = [('LR', LogisticRegression(class_weight = 'balanced')),
#           ('RF', RandomForestClassifier(class_weight = 'balanced')),
#           ('SVC', SVC(class_weight = 'balanced')),
#          ]
          

# modele = []
# recall = []
          
# for i, j in models:
#     print(i)
#     recall.append(score_classifier(X,labels,j))
#     modele.append(i)
    
# dict_from_list = dict(zip(modele, recall))
# sorted(dict_from_list.items(), key=lambda t: t[1])

# best = LR 54 % de recall (3959 fraudulent transactions well classified out of 7246)


# LR tuning
model = LogisticRegression()

parametres = {'penalty' : ['l1','l2'], 
              'class_weight' : ['balanced'],
              "C" : np.logspace(-4, 4, 20)
             
             }
                
best_model_LR = GridSearchCV(model,
                            parametres,
                            cv=5,
                            n_jobs=-1, scoring = "recall",
                            verbose=True).fit(X, labels)
best_model_LR.best_params_


score_classifier(X, labels, LogisticRegression( C = best_model_LR.best_params_['C'], class_weight = 'balanced',
                                      penalty = best_model_LR.best_params_['penalty']))


## 2nd modeling : using oversampling method : algorithm SMOTE 

def score_classifier_smote(dataset,labels,classifier): 
    
    """
    Train a classifier by maximizing recall, using a cross-validation by 3Fold, after oversampling training data 
    Displays the confusion matrix
    @param dataset : Array type, dataset without the target 
    @param labels : Array type, list of target per observation
    @param classifier : Function type, classifier to use 
    
   """
    kf = KFold(n_splits=3,random_state=50,shuffle=True)
    confusion_mat = np.zeros((2,2))
    recall = 0
    for training_ids,test_ids in kf.split(dataset):
        
        training_set = dataset[training_ids]
        training_labels = labels[training_ids]
        smo = SMOTE()
        training_set_sm, training_labels_sm = smo.fit_resample(training_set, training_labels)
        training_labels_sm = np.ravel(training_labels_sm)

        test_set = dataset[test_ids]
        test_labels = labels[test_ids]
        classifier.fit(training_set_sm,training_labels_sm)
        predicted_labels = classifier.predict(test_set)
        confusion_mat+=confusion_matrix(test_labels,predicted_labels)
        recall += recall_score(test_labels, predicted_labels)
        
    recall/=3

    print(f"recall : {recall}")
    
    sns.heatmap(confusion_mat, annot=True)
    plt.show()
    print(confusion_mat)
    return recall


models = [('LR', LogisticRegression()), # 0.5326, 993695 150490 3387 3859
          ('KNN', KNeighborsClassifier()),
          ('RF', RandomForestClassifier()),
          ('SVC', SVC()),
          ('XGBM', XGBClassifier()),
          ('GB',GradientBoostingClassifier())]
          

modele = []
recall = []
          
for i, j in models:
    recall.append(score_classifier_smote(X,labels,j))
    modele.append(i)
    
dict_from_list = dict(zip(modele, recall))
sorted(dict_from_list.items(), key=lambda t: t[1])


# LR tuning
model = LogisticRegression()

parametres = {'penalty' : ['l1','l2'], 
              "C" : np.logspace(-4, 4, 20)
             }
        
smo = SMOTE()
X_sm, labels_sm = smo.fit_resample(X, labels)
labels_sm = np.ravel(labels_sm)
        
best_model_smote_LR = GridSearchCV(model,
                            parametres,
                            cv=5,
                            n_jobs=-1, scoring = "recall",
                            verbose=True).fit(X_sm, labels_sm)

best_model_smote_LR.best_params_

score_classifier_smote(X, labels, LogisticRegression( C = best_model_smote_LR.best_params_['C'],
                                      penalty = best_model_smote_LR.best_params_['penalty']))




## 3rd modeling : boosting method 

model = BalancedRandomForestClassifier()

parametres = {'n_estimators' : range(10,100,10),
              'criterion' : ['gini','entropy']
             }

best_model_BRF = GridSearchCV(model,
                            parametres,
                            cv=5,
                            n_jobs=-1, scoring = "recall",
                            verbose=True).fit(X, labels)

best_model_BRF.best_params_

score_classifier(X,labels, BalancedRandomForestClassifier(best_model_BRF.best_params_['n_estimators'], 
                                                          best_model_BRF.best_params_['criterion']) )



# =============================================================================
# API 
# =============================================================================

df = df.loc[0:3000]

Presentation = pd.DataFrame({
    'Feature' : df.columns, 
    'Description' : ["Card ID","Country Code","Transaction status, 1 = accepted, 0 = declined","Merchant Type Code", "Transaction Amount","1 = fraudulent transaction, 0 = non-fraudulent transaction",
                     "Number of transactions in the last 3 hours", "Number of transactions in the last 6 hours","Number of transactions in the last 12 hours","Number of transactions in the last 24 hours",
                     "Cumulative amount of transactions in the last 3 hours","Cumulative amount of transactions in the last 6 hours", "Cumulative amount of transactions in the last 12 hours", 
                     "Cumulative amount of transactions in the last 2 hours", "Number of transactions made with the same type of merchant in the last 3 hours","Number of transactions made at the same type of merchant in the last 6 hours",
                     "Number of transactions with the same type of merchant in the last 12 hours","Number of transactions with the same type of merchant in the last 3 hours",
                     "Number of transactions made in a different country in the last 3 hours", "Number of transactions in a different country in the last 6 hours",
                     "Number of transactions in a different country in the last 12 hours","Number of transactions made in a different country in the last 24 hours",
                     "Month of the transaction", "Day of the transaction", "Time of transaction", "1 = transaction made between 00:00 and 6:00 am, 0 = daytime"
                     ]
    })


count_classes = pd.value_counts(df['fraude'], sort = True)
target_distrib = px.bar(count_classes)

def score_classifier_(dataset,labels,classifier):
    
    kf = KFold(n_splits=3,random_state=50,shuffle=True)
    confusion_mat = np.zeros((2,2))
    precision = 0
    recall = 0
    accuracy = 0
    
    for training_ids,test_ids in kf.split(dataset):
        
        training_set = dataset[training_ids]
        training_labels = labels[training_ids]
        
        test_set = dataset[test_ids]
        test_labels = labels[test_ids]
        
        classifier.fit(training_set,training_labels)
        predicted_labels = classifier.predict(test_set)
        
        confusion_mat += confusion_matrix(test_labels,predicted_labels)
        precision += precision_score(test_labels, predicted_labels)
        accuracy += accuracy_score(test_labels, predicted_labels)
        recall += recall_score(test_labels, predicted_labels)
        
    accuracy /= 3
    recall /= 3
    precision /= 3

    return confusion_mat, precision, accuracy, recall


confusion_mat, precision, accuracy, recall = score_classifier_(X, labels, LogisticRegression( C = best_model_LR.best_params_['C'], class_weight = 'balanced',
                                      penalty = best_model_LR.best_params_['penalty']))


# metrics of the model in a table
Score = pd.DataFrame({
    "Metrics" : ["Accuracy","Recall","Precision"],
    "Score" : [round(accuracy,2),round(recall,2),round(precision,2)]
    })


# confusion mat
confusion_mat = px.imshow(confusion_mat,text_auto=True,
                          labels=dict(x="Predicted label", y="True label"), #, color="Numbers of players"),
                          x = ['0','1'],
                          y = ['0','1'], color_continuous_scale = 'blues')
confusion_mat.update_layout(paper_bgcolor = '#e1fefe') 



# 2nd modeling 
def score_classifier_smote_(dataset,labels,classifier): 
    
    """
    Train a classifier by maximizing recall, using a cross-validation by 3Fold, after oversampling training data 
    Displays the confusion matrix
    @param dataset : Array type, dataset without the target 
    @param labels : Array type, list of target per observation
    @param classifier : Function type, classifier to use 
    
   """
    kf = KFold(n_splits=3,random_state=50,shuffle=True)
    confusion_mat = np.zeros((2,2))
    recall = 0
    precision = 0
    accuracy = 0
    
    for training_ids,test_ids in kf.split(dataset):
        
        training_set = dataset[training_ids]
        training_labels = labels[training_ids]
        
        smo = SMOTE()
        training_set_sm, training_labels_sm = smo.fit_resample(training_set, training_labels)
        training_labels_sm = np.ravel(training_labels_sm)

        test_set = dataset[test_ids]
        test_labels = labels[test_ids]
        classifier.fit(training_set_sm,training_labels_sm)
        predicted_labels = classifier.predict(test_set)

        confusion_mat += confusion_matrix(test_labels,predicted_labels)
        precision += precision_score(test_labels, predicted_labels)
        accuracy += accuracy_score(test_labels, predicted_labels)
        recall += recall_score(test_labels, predicted_labels)
    
    accuracy /= 3
    recall /= 3
    precision /= 3
    
    return confusion_mat, precision, accuracy, recall


confusion_mat2, precision2, accuracy2, recall2 = score_classifier_smote_(X, labels, LogisticRegression( C = best_model_smote_LR.best_params_['C'],
                                      penalty = best_model_smote_LR.best_params_['penalty']))

# metrics of the model in a table
Score2 = pd.DataFrame({
    "Metrics" : ["Accuracy","Recall","Precision"],
    "Score" : [round(accuracy2,2),round(recall2,2),round(precision2,2)]
    })


# confusion mat
confusion_mat2 = px.imshow(confusion_mat2,text_auto=True,
                          labels=dict(x="Predicted label", y="True label"), #, color="Numbers of players"),
                          x = ['0','1'],
                          y = ['0','1'], color_continuous_scale = 'blues')
confusion_mat2.update_layout(paper_bgcolor = '#e1fefe') 


# 3rd modeling 

confusion_mat3, precision3, accuracy3, recall3 = score_classifier_(X, labels, BalancedRandomForestClassifier(best_model_BRF.best_params_['n_estimators'], 
                                                          best_model_BRF.best_params_['criterion']))

# metrics of the model in a table
Score3 = pd.DataFrame({
    "Metrics" : ["Accuracy","Recall","Precision"],
    "Score" : [round(accuracy3,2),round(recall3,2),round(precision3,2)]
    })


# confusion mat
confusion_mat3 = px.imshow(confusion_mat3,text_auto=True,
                          labels=dict(x="Predicted label", y="True label"), #, color="Numbers of players"),
                          x = ['0','1'],
                          y = ['0','1'], color_continuous_scale = 'blues')
confusion_mat3.update_layout(paper_bgcolor = '#e1fefe') 


# =============================================================================
# Creation of the API 
# =============================================================================

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# Style and sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#111111",
}

# style of the application
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "#111111",
}

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


sidebar = html.Div(
    [
        html.H2("Menu", className="display-4",
        style = 
        {'color': colors['text']}),
        
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Presentation of the data", href="/", active="exact"),
                dbc.NavLink("Data visualization", href="/page-1", active="exact"),
                dbc.NavLink("Modeling 1 : penalized classification", href="/page-2", active="exact"),
                dbc.NavLink("Modeling 2 : oversampling", href = "/page-3", active = "exact"),
                dbc.NavLink("Modeling 3 : Boosting Model", href = "/page-4", active = "exact")
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)


content = html.Div(id="page-content", style=CONTENT_STYLE)


app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    dcc.Location(id="url"), 
    sidebar,
    content])

# page 1
page_1_layout = html.Div(style={'backgroundColor': colors['background']},
                         children=[
    
    html.Div(id = "page1"),
    html.H1('Description of the data',
            style={
            'textAlign': 'center',
            'color': colors['text']}),
    html.Br(), 
    html.Br(),
    html.H6("The database is composed of " + str(len(df)) + " observations, corresponding to bank transactions made with " +
            str(df['Carte'].nunique()) + " different cards, in " + str(df['Pays'].nunique()) + " different countries.", 
            style = {
                'color' : colors['text']
                }),

    html.Br(),
    dbc.Row(
        [
            dbc.Col(
                
                html.Div([
                
                html.H6("Here is a short description of each feature obtained on the different transactions : ",
                        style = {
                            'color' : colors['text']
                            }),
                html.Br(),
                html.Div(
                    dash_table.DataTable(Presentation.to_dict('records'),
                                     style_cell={'textAlign': 'left',
                                                 'border': '1px solid grey' ,
                                                 },
                                     style_header = { 'fontWeight': 'bold',
                                                     'color': "#07b4f7", #colors['text'],
                                                     'backgroundColor': colors['background']},
                                     style_data={'whiteSpace': 'normal',
                                                 'width': 'auto',
                                                 'color':"#ebeded", # colors['text'],
                                                 'backgroundColor': colors['background']}
                )  ),
                
                html.Br(), 
                html.H6("There are no missing values in the data",
                        style = {
                            'color' : colors['text']
                            }),
                html.H6("Note that the variables Fm_Velocity_Condition, FM_Sum, FM_redondance and FM_Difference_pays were calculated by the author of the database, and that the variables Month, Day, Time and Night were created for the model.",
                        style = {
                            'color' : colors['text']
                            })
                
                ])
               ),
            
            dbc.Col(
                html.Div([
                    html.Br(), 
                    html.Br(),
                    html.H6("The problem of this project is to predict fraudulent transactions, given their small number.",
                                style = {
                               'color' : colors['text']
                               }),
                    html.H6('We have, in the Modeling sections, tried to find the best classifier by maximizing the recall. We have tested several models with different handling of unbalanced data.',
                            style = {
                                'color' : colors['text']
                                }),
                    html.H6('Above is a graph showing the imbalance in the modalities of the variable we are trying to predict: ',
                            style = {
                                'color' : colors['text']
                                }),
                    html.Br(), 
                    dcc.Graph(
                                id='target_distrib',
                                figure = target_distrib
                            ), 
                    html.Br(), 
                    html.H6('The Data vizualization part allows, before the modeling, to represent the data graphically. ', 
                            style = {
                                'color' : colors['text']
                                })
              
            
            ])
            )
        ]
            )
                

    
    ])


# page 2
page_2_layout = html.Div(style={'backgroundColor': colors['background']}, 
                         children = [
         html.H1("Data visualization",
            style={
            'textAlign': 'center',
            'color': colors['text']}),
    
    html.Br(),
    html.Br(),

   
            dbc.Row([
               
            dbc.Col(
                html.Div(children = [
                    html.Label("Select a variable", style = {'color' : colors['text']}
                               ),
                    html.Br(),
                            dcc.Dropdown(id ="var",
                                options = df_boxplot.columns,
                                value = "Montant", 
                                style = {'backgroundColor' : colors['text']}

                    )
                    ]), width = 3)
           
            ]),
            
            dbc.Row(
                [
            dbc.Col(
                html.Div( children = [
                html.Br(),
                dcc.Graph(id = 'graph1')
                ])), 
                    
            dbc.Col( html.Div( children = [
                html.Br(),
                        dcc.Graph(id = 'graph2'),
                        ])
                        
                        )
                    
                    ]), 
            
            dbc.Row([
             
                
                dbc.Col(html.Div( children = [
                    html.Br(),
                html.Label("Select two variables",
                           style = {'color' : colors['text']}),
                        dcc.Dropdown(id ="var1",
                            options = df_boxplot.columns,
                            value = "Montant",
                            style = {'backgroundColor' : colors['text']}),
                        html.Br(),
                        dcc.Dropdown(id = "var2",
                                     options = df_boxplot.columns,
                                     value = "FM_Sum_6",
                                     style = {'backgroundColor' : colors['text']})
                    ]),width = 3)
                     
                     ]),
                    dbc.Row(
                        [
                    dbc.Col(
                        html.Div( children = [
                        html.Br(),
                        dcc.Graph(id = 'graph3')
                        ]), width = 6), 
                            
                    dbc.Col( html.Div( children = [
                        html.Br(),
                                dcc.Graph(id = 'graph4'),
                                ])
                                , width = 6
                                )
                            
                            ])
    ])
    
                              

@callback(
    Output("graph1","figure"),
    Output("graph2","figure"),
    [
    Input("var","value"),
    ],
    )

def make_graph1(var):
    graph1 = px.box(y = df[var], width=500, height=400)
    graph1.update_layout(title_text="Boxplot of " + var,
                         paper_bgcolor = '#e1fefe', #colors['text'],
                         font_color=colors['background']
                         )
    
    color = ['#0c36e0', '#3ed4fa']
       
    df_0 = df[df['fraude'] == 0]
    df_1 = df[df['fraude'] == 1]
   
    group_data = [df_0[var], df_1[var]]
   
    graph2 = ff.create_distplot(group_data, ['Target = 0','Target = 1'],
                              show_hist=False, show_rug= False,
                              colors = color)
   
    graph2.update_layout(title_text = "Distplot of " + var, 
                         paper_bgcolor = '#e1fefe', #colors['text'],
                         font_color=colors['background']
                         )

    return graph1, graph2


@callback(
    Output('graph3', 'figure'),
    Output('graph4','figure'),
    [
     Input("var1","value"),
     Input("var2","value")
     ]
    )

def make_graph2(var1,var2):
    graph3 = px.scatter(df, x = var1, y = var2) 
    graph3.update_layout(title_text="Scatter plot of variables " + var1 + " and " + var2,
                         paper_bgcolor = '#e1fefe', #colors['text'],
                         font_color=colors['background']
                         )
    
    df_corr = df[[var1,var2]]
    graph4 = px.imshow(df_corr.corr(), text_auto=True, 
                       color_continuous_scale = 'blues')
    
    graph4.update_layout(title_text="Correlation matrix between " + var1 + " and " + var2,
                         paper_bgcolor = '#e1fefe', #colors['text'],
                         font_color=colors['background']
                         )
    
    return graph3,graph4


# page 3
page_3_layout = html.Div(style={'backgroundColor': colors['background']}, 
                         children = [
         html.H1("1st modeling : penalized classification",
            style={
            'textAlign': 'center',
            'color': colors['text']}),
         html.Br(), 
         html.Br(), 
         html.H6("First, we tested several classifiers using a penalized model for each of them. This allows us to impose an additional cost on the model for classification errors made on the minority class.",
                 style = {'color':colors['text']}),
         html.H6("We decided to use a method allowing to associate to each class a weight inversely proportional to its frequency.",
                 style = {'color':colors['text']}),
         html.H6("Among the classifiers tested, we chose the one maximizing the recall, the Logistic Regression. Here are the results of the classification", 
                 style = {'color':colors['text']}), 
         
         dbc.Row([
             dbc.Col(
                 html.Div([
                     html.Br(), 
                     html.Br(),
                     html.H6("Synthesis of the classifier metrics",
                             style = {'color':colors['text']}),
                     html.Br(),
                     html.Div(
                         dash_table.DataTable(Score.to_dict('records'),
                                          style_cell={'textAlign': 'left',
                                                      'border': '1px solid grey' ,
                                                      },
                                          style_header = { 'fontWeight': 'bold',
                                                          'color': "#07b4f7", #colors['text'],
                                                          'backgroundColor': colors['background']},
                                          style_data={'whiteSpace': 'normal',
                                                      'width': 'auto',
                                                      'color':"#ebeded", # colors['text'],
                                                      'backgroundColor': colors['background']}
                     )  ),
                     
                     
                     ])
                 ), 
             
             dbc.Col(
                 html.Div([
                     html.Br(), 
                     html.Br(),
                     html.H6("Confusion matrix : ",
                             style = {'color':colors['text']}),
                     html.Br(), 
                     dcc.Graph(
                                 id='confusion_mat',
                                 figure = confusion_mat
                             )
                     
                     ])
                 )
             ]),
         
         html.Br(),
         html.H4("Prediction",
                 style = {'color':colors['text']}),
         html.H6("Enter the characteristics of a transaction to see if the model predicts it as fraud or not.",
                 style = {'color':colors['text']}),
         html.Br(),
         
         dbc.Row([
             dbc.Col(html.Div([
                 html.H6("Country code :",
                         style = {'color':colors['text']}),
                 dcc.Dropdown(id ="country",
                     options = df_vals['Pays'].unique(),
                     value = "634",
                     style = {'backgroundColor' : colors['text']})
                 
                 ]), width = 3
                 
                 ),
             dbc.Col(html.Div([
                 html.H6("Merchant Code :",
                         style = {'color':colors['text']}),
                 dcc.Dropdown(id ="merchant",
                     options = df_vals['MCC'].unique(),
                     value = "5946",
                     style = {'backgroundColor' : colors['text']})
                 
                 ]), width = 3),
             
             dbc.Col(
                 html.Div([
                     html.H6("Transaction accepted :",
                             style = {'color':colors['text']}),
                     dcc.Dropdown(id ="accepted",
                         options = ['Yes','No'],
                         value = "Yes",
                         style = {'backgroundColor' : colors['text']})
                     
                     ]), width = 3),
             
             dbc.Col(
                 html.Div([
                     html.H6('Time of the transaction :',
                             style = {'color' : colors['text']}),
                     dcc.Dropdown(id = 'hours', 
                                  options = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
                                  value = 8, 
                                  style = {'backgroundColor' : colors['text']})
                     ]), width = 3
                 )
             
             ]),
         
         dbc.Row([
             dbc.Col(html.Div([
                 html.Br(),
                 html.H6('Montant of the transaction', 
                         style = {'color' : colors['text']}),
                 dcc.Input(
                     id = 'montant', type = 'number',
                     style = {'backgroundColor':colors['text']}),
                
                 ]),
                 
                 width = 4),
             dbc.Col(
                 html.Div([
                     html.Br(),
                     html.H6('Date of the transaction :',
                             style = {'color' : colors['text']}),
                     dcc.DatePickerSingle(
                         id='transaction_date',
                         min_date_allowed = date(2020, 1, 1),
                         max_date_allowed = date(2022, 12, 31),
                         initial_visible_month = date(2020, 1, 1),
                         date = date(2020, 1, 1),
                         style = {'backgroundColor' : colors['text']}
                         )
                     ]), width = 4), 
             dbc.Col(width = 4)
             ]),
         
         html.Br(), 
        # html.Div(id = 'prediction', style = {'color' : colors['text']}),
         dcc.Loading(
                    id="loading",
                    children=[html.Div([html.Div(id = 'prediction', style = {'color' : colors['text']})])],
                    type="circle"
                )
         
         
         ])

@callback(
    Output('prediction','children'),
   
    [
      Input('transaction_date','date'),
      Input('country','value'),
      Input('merchant','value'),
      Input('accepted','value'),
      Input('hours','value'),
      Input('montant','value')
      ]
    )


def prediction(date_value,country,merchant,accepted,hours,montant):
    #time.sleep(1)
    if accepted == 'Yes':
        accepted = 1
    else :
        accepted = 0
        
    night = 0
    if hours < 7 :
        night = 1
        
        
    date_object = date.fromisoformat(date_value)
    month = int(date_object.strftime('%m'))
    day = int(date_object.strftime('%d'))   
    
    
    model = LogisticRegression( C = best_model_LR.best_params_['C'], class_weight = 'balanced',
                                          penalty = best_model_LR.best_params_['penalty'])
    model.fit(X,labels)
    new_obs = pd.DataFrame({
        'country' : [country], 
        'coderep' : [accepted],
        'MCC' : [merchant],
        'montant' : [montant],
        'month' : [month],
        'day' : [day],
        'time' : [hours],
        'night' : [night],
        'velocity' : [-2.75573],
        'sum_montant' : [0],
        'redondance' : [-0.375381],
        'diff' : [-0.179571]
        })
    
    predict = model.predict(new_obs.values)
    
    if predict == 1:
        return "Based on the characteristics of this transaction, the model predicts that it is a fraud "
    else:
        return "Based on the characteristics of this transaction, the model predicts that it is not a fraud "
    
    
    
    
# page 4
page_4_layout = html.Div(style={'backgroundColor': colors['background']}, 
                         children = [
         html.H1("2nd modeling : oversampling",
            style={
            'textAlign': 'center',
            'color': colors['text']}),
         html.Br(), 
         html.Br(), 
         html.H6("We then tested a second modeling using this time an algorithm allowing to do a resampling on the data. ",
                 style = {'color':colors['text']}),
         html.H6("More precisely, we used an oversampling method: the SMOTE algorithm. This one allows to create additional synthetic observations belonging to the minority class so that the model can train on this class. ",
                 style = {'color':colors['text']}),
         html.H6("Here again we tested several classifiers, and the best performing one is the logistic regression. Here are the results of the classification", 
                 style = {'color':colors['text']}), 
         
         dbc.Row([
             dbc.Col(
                 html.Div([
                     html.Br(), 
                     html.Br(),
                     html.H6("Synthesis of the classifier metrics",
                             style = {'color':colors['text']}),
                     html.Br(),
                     html.Div(
                         dash_table.DataTable(Score2.to_dict('records'),
                                          style_cell={'textAlign': 'left',
                                                      'border': '1px solid grey' ,
                                                      },
                                          style_header = { 'fontWeight': 'bold',
                                                          'color': "#07b4f7", #colors['text'],
                                                          'backgroundColor': colors['background']},
                                          style_data={'whiteSpace': 'normal',
                                                      'width': 'auto',
                                                      'color':"#ebeded", # colors['text'],
                                                      'backgroundColor': colors['background']}
                     )  ),
                     
                     
                     ])
                 ), 
             
             dbc.Col(
                 html.Div([
                     html.Br(), 
                     html.Br(),
                     html.H6("Confusion matrix : ",
                             style = {'color':colors['text']}),
                     html.Br(), 
                     dcc.Graph(
                                 id='confusion_mat2',
                                 figure = confusion_mat2
                             )
                     
                     ])
                 )
             ]),
         
         html.Br(),
         html.H4("Prediction",
                 style = {'color':colors['text']}),
         html.H6("Enter the characteristics of a transaction to see if the model predicts it as fraud or not.",
                 style = {'color':colors['text']}),
         html.Br(),
         
         dbc.Row([
             dbc.Col(html.Div([
                 html.H6("Country code :",
                         style = {'color':colors['text']}),
                 dcc.Dropdown(id ="country2",
                     options = df_vals['Pays'].unique(),
                     value = "634",
                     style = {'backgroundColor' : colors['text']})
                 
                 ]), width = 3
                 
                 ),
             dbc.Col(html.Div([
                 html.H6("Merchant Code :",
                         style = {'color':colors['text']}),
                 dcc.Dropdown(id ="merchant2",
                     options = df_vals['MCC'].unique(),
                     value = "5946",
                     style = {'backgroundColor' : colors['text']})
                 
                 ]), width = 3),
             
             dbc.Col(
                 html.Div([
                     html.H6("Transaction accepted :",
                             style = {'color':colors['text']}),
                     dcc.Dropdown(id ="accepted2",
                         options = ['Yes','No'],
                         value = "Yes",
                         style = {'backgroundColor' : colors['text']})
                     
                     ]), width = 3),
             
             dbc.Col(
                 html.Div([
                     html.H6('Time of the transaction :',
                             style = {'color' : colors['text']}),
                     dcc.Dropdown(id = 'hours2', 
                                  options = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
                                  value = 8, 
                                  style = {'backgroundColor' : colors['text']})
                     ]), width = 3
                 )
             
             ]),
         
         dbc.Row([
             dbc.Col(html.Div([
                 html.Br(),
                 html.H6('Montant of the transaction', 
                         style = {'color' : colors['text']}),
                 dcc.Input(
                     id = 'montant2', type = 'number',
                     style = {'backgroundColor':colors['text']}),
                
                 ]),
                 
                 width = 4),
             dbc.Col(
                 html.Div([
                     html.Br(),
                     html.H6('Date of the transaction :',
                             style = {'color' : colors['text']}),
                     dcc.DatePickerSingle(
                         id='transaction_date2',
                         min_date_allowed = date(2020, 1, 1),
                         max_date_allowed = date(2022, 12, 31),
                         initial_visible_month = date(2020, 1, 1),
                         date = date(2020, 1, 1),
                         style = {'backgroundColor' : colors['text']}
                         )
                     ]), width = 4), 
             dbc.Col(width = 4)
             ]),
         
         html.Br(), 
        # html.Div(id = 'prediction', style = {'color' : colors['text']}),
         dcc.Loading(
                    id="loading",
                    children=[html.Div([html.Div(id = 'prediction2', style = {'color' : colors['text']})])],
                    type="circle"
                )
         
         
         ])

@callback(
    Output('prediction2','children'),
   
    [
      Input('transaction_date2','date'),
      Input('country2','value'),
      Input('merchant2','value'),
      Input('accepted2','value'),
      Input('hours2','value'),
      Input('montant2','value')
      ]
    )


def prediction_2(date_value,country2,merchant2,accepted2,hours2,montant2):
    #time.sleep(1)
    if accepted2 == 'Yes':
        accepted2 = 1
    else :
        accepted2 = 0
        
    night = 0
    if hours2 < 7 :
        night = 1
        
        
    date_object = date.fromisoformat(date_value)
    month = int(date_object.strftime('%m'))
    day = int(date_object.strftime('%d'))   
    
    
    smo = SMOTE()
    X_sm, labels_sm = smo.fit_resample(X, labels)
    labels_sm = np.ravel(labels_sm)

    
    model = LogisticRegression( C = best_model_smote_LR.best_params_['C'],
                                          penalty = best_model_smote_LR.best_params_['penalty'])
    model.fit(X_sm,labels_sm)
    new_obs = pd.DataFrame({
        'country' : [country2], 
        'coderep' : [accepted2],
        'MCC' : [merchant2],
        'montant' : [montant2],
        'month' : [month],
        'day' : [day],
        'time' : [hours2],
        'night' : [night],
        'velocity' : [-2.75573],
        'sum_montant' : [0],
        'redondance' : [-0.375381],
        'diff' : [-0.179571]
        })
    
    predict = model.predict(new_obs.values)
    
    if predict == 1:
        return "Based on the characteristics of this transaction, the model predicts that it is a fraud "
    else:
        return "Based on the characteristics of this transaction, the model predicts that it is not a fraud "
    
    
    
# page 5
page_5_layout = html.Div(style={'backgroundColor': colors['background']}, 
                         children = [
         html.H1("3rd modeling : boosting",
            style={
            'textAlign': 'center',
            'color': colors['text']}),
         html.Br(), 
         html.Br(), 
         html.H6("Finally, we used a boosting method, which turns out to be the best performing classifier. ",
                 style = {'color':colors['text']}),
         html.H6("It is the balanced random forest model. Here are the results of the classification", 
                 style = {'color':colors['text']}), 
         
         dbc.Row([
             dbc.Col(
                 html.Div([
                     html.Br(), 
                     html.Br(),
                     html.H6("Synthesis of the classifier metrics",
                             style = {'color':colors['text']}),
                     html.Br(),
                     html.Div(
                         dash_table.DataTable(Score3.to_dict('records'),
                                          style_cell={'textAlign': 'left',
                                                      'border': '1px solid grey' ,
                                                      },
                                          style_header = { 'fontWeight': 'bold',
                                                          'color': "#07b4f7", #colors['text'],
                                                          'backgroundColor': colors['background']},
                                          style_data={'whiteSpace': 'normal',
                                                      'width': 'auto',
                                                      'color':"#ebeded", # colors['text'],
                                                      'backgroundColor': colors['background']}
                     )  ),
                     
                     
                     ])
                 ), 
             
             dbc.Col(
                 html.Div([
                     html.Br(), 
                     html.Br(),
                     html.H6("Confusion matrix : ",
                             style = {'color':colors['text']}),
                     html.Br(), 
                     dcc.Graph(
                                 id='confusion_mat3',
                                 figure = confusion_mat3
                                 
                             )
                     
                     ])
                 )
             ]),
         
         html.Br(),
         html.H4("Prediction",
                 style = {'color':colors['text']}),
         html.H6("Enter the characteristics of a transaction to see if the model predicts it as fraud or not.",
                 style = {'color':colors['text']}),
         html.Br(),
         
         dbc.Row([
             dbc.Col(html.Div([
                 html.H6("Country code :",
                         style = {'color':colors['text']}),
                 dcc.Dropdown(id ="country3",
                     options = df_vals['Pays'].unique(),
                     value = "634",
                     style = {'backgroundColor' : colors['text']})
                 
                 ]), width = 3
                 
                 ),
             dbc.Col(html.Div([
                 html.H6("Merchant Code :",
                         style = {'color':colors['text']}),
                 dcc.Dropdown(id ="merchant3",
                     options = df_vals['MCC'].unique(),
                     value = "5946",
                     style = {'backgroundColor' : colors['text']})
                 
                 ]), width = 3),
             
             dbc.Col(
                 html.Div([
                     html.H6("Transaction accepted :",
                             style = {'color':colors['text']}),
                     dcc.Dropdown(id ="accepted3",
                         options = ['Yes','No'],
                         value = "Yes",
                         style = {'backgroundColor' : colors['text']})
                     
                     ]), width = 3),
             
             dbc.Col(
                 html.Div([
                     html.H6('Time of the transaction :',
                             style = {'color' : colors['text']}),
                     dcc.Dropdown(id = 'hours3', 
                                  options = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
                                  value = 8, 
                                  style = {'backgroundColor' : colors['text']})
                     ]), width = 3
                 )
             
             ]),
         
         dbc.Row([
             dbc.Col(html.Div([
                 html.Br(),
                 html.H6('Montant of the transaction', 
                         style = {'color' : colors['text']}),
                 dcc.Input(
                     id = 'montant3', type = 'number',
                     style = {'backgroundColor':colors['text']}),
                
                 ]),
                 
                 width = 4),
             dbc.Col(
                 html.Div([
                     html.Br(),
                     html.H6('Date of the transaction :',
                             style = {'color' : colors['text']}),
                     dcc.DatePickerSingle(
                         id='transaction_date3',
                         min_date_allowed = date(2020, 1, 1),
                         max_date_allowed = date(2022, 12, 31),
                         initial_visible_month = date(2020, 1, 1),
                         date = date(2020, 1, 1),
                         style = {'backgroundColor' : colors['text']}
                         )
                     ]), width = 4), 
             dbc.Col(width = 4)
             ]),
         
         html.Br(), 
        # html.Div(id = 'prediction', style = {'color' : colors['text']}),
         dcc.Loading(
                    id="loading",
                    children=[html.Div([html.Div(id = 'prediction3', style = {'color' : colors['text']})])],
                    type="circle"
                )
         
         
         ])

@callback(
    Output('prediction3','children'),
   
    [
      Input('transaction_date3','date'),
      Input('country3','value'),
      Input('merchant3','value'),
      Input('accepted3','value'),
      Input('hours3','value'),
      Input('montant3','value')
      ]
    )


def prediction_3(date_value,country3,merchant3,accepted3,hours3,montant3):
    #time.sleep(1)
    if accepted3 == 'Yes':
        accepted3 = 1
    else :
        accepted3 = 0
        
    night = 0
    if hours3 < 7 :
        night = 1
        
        
    date_object = date.fromisoformat(date_value)
    month = int(date_object.strftime('%m'))
    day = int(date_object.strftime('%d'))   
    
    

    model = BalancedRandomForestClassifier(best_model_BRF.best_params_['n_estimators'], 
                                                              best_model_BRF.best_params_['criterion'])
    model.fit(X,labels)
    
    new_obs = pd.DataFrame({
        'country' : [country3], 
        'coderep' : [accepted3],
        'MCC' : [merchant3],
        'montant' : [montant3],
        'month' : [month],
        'day' : [day],
        'time' : [hours3],
        'night' : [night],
        'velocity' : [-2.75573],
        'sum_montant' : [0],
        'redondance' : [-0.375381],
        'diff' : [-0.179571]
        })
    
    predict = model.predict(new_obs.values)
    
    if predict == 1:
        return "Based on the characteristics of this transaction, the model predicts that it is a fraud "
    else:
        return "Based on the characteristics of this transaction, the model predicts that it is not a fraud "
    
    
    
        


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return page_1_layout
       
    elif pathname == "/page-1":
        return page_2_layout

    elif pathname == "/page-2":
        return page_3_layout

    elif pathname == "/page-3":
        return page_4_layout 
    
    elif pathname == "/page-4":
        return page_5_layout 
    
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__ == '__main__':
    app.run_server(debug=False)
    
    
    
    

