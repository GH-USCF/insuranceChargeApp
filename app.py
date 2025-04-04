import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objs as go

# Innitialize the Dash app
app = dash.Dash(__name__)

# Load Data and Preprocessing Data
df=pd.read_csv("{Class 8}insurance.csv", engine='python')

#Create single/family status
df['single_family']=np.where(df.children==0, 'single', 'family')

#Create age groups
age_bins=[18,35,50,65]
age_labels=['Young Adult', 'Adult', 'Middle-aged']
df['age_group']=pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

#Create bmi groups
bmi_bins=[0,18.5,25,30,60]
bmi_labels=['Underweight', 'Normal Weight', 'Overweight', 'Obese']
df['bmi_group']=pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels, right=False)

#Create Machine Learning Model
#Encode categorical data
encoder=LabelEncoder()
cat_col=[col for col in df.columns if df[col].dtype in ['O', 'category']]
df_model=df.copy()
for col in cat_col:
    df_model[col]=encoder.fit_transform(df[col])
#Train the model
listTrainCols=['age','bmi', 'sex', 'smoker', 'region', 'children']
X=df_model[listTrainCols]
y=df_model['charges']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=0)
model=LinearRegression()
model.fit(X_train,y_train)

#APP Layout
app.layout=html.Div([
    html.H1("Insurance Charge Analysis Dashboard", style={'textAlign':'center'}),
    html.H2("{Class 7}Now its getting good"),
    
    #Dropdown for plot selection
    dcc.Dropdown(
        id='plot-selector',
        options=[
            {'label': 'Charges by Family Status and Smoking', 'value': 'family_moke'},
            {'label': 'Charges by Number of Children', 'value': 'children'},
            {'label': 'Averge Charges by Age Group', 'value': 'age_group'},
        ],
        value='family_smoke',
        clearable=False
    ),
    
    #Graph display
    dcc.Graph(id='main_graph'),
    
    #Prediction Section
    html.Div([
        html.H2("Insurance Charge Prediction"),
        
        #Input fields
        html.Div([
            html.Label('Age'),
            dcc.Input(id='age-input', type='number', value=20),
            
            html.Label('BMI'),
            dcc.Input(id='bmi-input', type='number', value=25),
            
            html.Label('Children'),
            dcc.Input(id='children-input', type='number', value=0),
            
            html.Label('Sex'),
            dcc.Dropdown(
                id='sex-input', 
                options=[{'label':'Male','value':1},
                         {'label':'Female','value':0}],
                value=1,
            ),
            
            html.Label('Smoker'),
            dcc.Dropdown(
                id='smoker-input', 
                options=[{'label':'Yes','value':1},
                         {'label':'No','value':0}],
                value=1,
            ),
            
            html.Label('Region'),
            dcc.Dropdown(
                id='region-input', 
                options=[{'label':'Northeast','value':0},
                         {'label':'Northwest','value':1},
                         {'label':'Southeast','value':2},
                         {'label':'Southwest','value':3}],
                value=0,
            ),
        ],style={'display':'flex','flexDirection':'column','gap':'10px'} ),
        
        html.Button('Predict', id='prdict-button', n_clicks=0),
        
        html.Div(id='prediction-output')
    ])
    
])

#Call back
@app.callback(
    Output('main_graph', 'figure'),
    Input('plot-selector', 'value')
)
def update_graph(selected_plot):
    if selected_plot=='family_moke':
        fig=px.box(df,y='charges',x='single_family',color='smoker',
                   title='Insurance Charges by Family Status and Smoking Habit')
        
    elif selected_plot=='children':
        fig=px.box(df,y='charges',x='children',color='smoker',
                   title='Insurance Charges by Number of Children')
        
    elif selected_plot=='age_group':
        age_group_charges=df.groupby('age_group')['charges'].mean().reset_index()
        fig=px.bar(age_group_charges,y='age_group',x='charges',
                   title='Averge Insurance Charges by Age Group')
    return fig

@app.callback(
    Output('prediction-output','children'),
    Input('prdict-button', 'n_clicks'),
    [State('age-input','value'),
     State('bmi-input','value'),
     State('children-input','value'),
     State('sex-input','value'),
     State('smoker-input','value'),
     State('region-input','value')
    ]
)

def predict_charge(n_clicks, age, bmi, children, sex, smoker, region):
    if n_clicks>0:
        input_data=np.array([[age, bmi, children, sex, smoker, region]])
        prediction=model.predict(input_data)[0]
        return f'Predicted Insurance Charge: ${prediction:,.2f}'
    return ''

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8096)