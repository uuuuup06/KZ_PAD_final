import pandas as pd
from dash import dcc
from dash import html
import numpy as np
from dash import dash_table, Input, Output
import plotly.express as px
from jupyter_dash import JupyterDash
import plotly.graph_objects as go
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv('houses_parser.csv')
data2 = pd.read_csv('houses_parser_initial.csv')

df2_proper_columns = data.loc[:, data.columns.difference(['Unnamed: 0'])].columns.values;


X = data['area'].values.reshape(-1,1)
y = data['price'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Creating and training a linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# It predicts appartment prices
# based on the area and evaluates the quality of the model using various error metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

X = data[['roomsNumber', 'area', 'year']].values
y = data['price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Creating and training a linear regression model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

#Prints the predicted value and the real value from the test set for one example and
#evaluates the accuracy of the model
print("Predict value " + str(model.predict([X_test[9]])))
print("Real value " + str(y_test[9]))
print("Accuracy --> ", model.score(X_test, y_test)*100)

df_price_without_District = data.drop(columns=['district'])

#Creating and training a linear regression model
#The model describes the dependence of the price (price) on the independent variables (roomsNumber and area)
model_price = smf.ols(formula="price ~ roomsNumber + area", data=df_price_without_District).fit()

#Statistical characteristics of the model.
# This allows us to assess the statistical significance and influence of the signs on the price
print('P Values: ', model_price.pvalues.values)
pValue = model_price.pvalues.values
coef = model_price.pvalues.values
stdError =  model_price.bse.values
print('Coef: ', model_price.params.values)
print("Std Errs", model_price.bse.values)


correlations = df_price_without_District.corr(numeric_only=True)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


statistics = data.describe()

app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
# Create the histogram
histogram_dist = px.histogram(data, x='district', width=600, height=400, title="Distribution of apartments by district")
histogram_year = px.histogram(data, x='year', width=600, height=400, title="Distribution of apartments by year")
histogram_dist_area_rn = px.scatter(data, "area", "district", "roomsNumber" , title = "Districts with Area and Room number")
desc = data.describe()
statistics = data.describe()

# Create the figure
fig = go.Figure(data=[go.Table(
    header=dict(values=['Statistics'] + statistics.columns.tolist(),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']] + [statistics[col].tolist() for col in statistics.columns],
               fill_color='lavender',
               align='left'))
])

# Update the layout
fig.update_layout(
    title='Descriptive Statistics',
    autosize=False,
    width=900,
    height=400
)


fig_corr = px.imshow(correlations)
fig_corr.update_layout(title="Variable Correlations")

app.layout = html.Div([
    html.H1(
        children='Apartment price',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    html.Br(),
    html.Div(children='CSV table view before cleaning data', style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    html.Div([

        html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in data2.columns])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(data2.iloc[i][col]) for col in data2.columns
                ]) for i in range(min(len(data2), 10))
            ])
        ])



    ], style={
        'padding': '10px 50px'
    }),

    html.Div(children='CSV table view after cleaning data', style={
        'textAlign': 'center',
        'color': colors['text']
    }),


    html.Div([

        html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in data.columns])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(data.iloc[i][col]) for col in data.columns
                ]) for i in range(min(len(data), 10))
            ])
        ])



    ], style={
        'padding': '30px 50px'
    }),
    html.H1(children='Data analysis',
        style={'textAlign': 'center','color': colors['text']}),


    html.P(
        children='Here predict apartment prices based on the area and evaluate the quality of the model using various error metrics.'
                 'They show how well the model predicts the target variable (in this case, prices) based on input features (area).',
        style={
            'textAlign': 'center',
            'padding': '0 30'
        }
    ),

    html.Div([
        html.Table([
            html.Tbody([
                html.Tr([
                    html.Td('Mean Absolute Error:'),
                    html.Td(metrics.mean_absolute_error(y_test, y_pred)),
                ]),
                html.Tr([
                    html.Td('Mean Squared Error:'),
                    html.Td(metrics.mean_squared_error(y_test, y_pred)),
                ]),
                html.Tr([
                    html.Td('Root Mean Squared Error:'),
                    html.Td(np.sqrt(metrics.mean_squared_error(y_test, y_pred))),
                ])
            ])
        ], style={'padding': 10}),
    ], style={
        'padding': '10px 50px'}),
    html.Br(),

    html.P(
        children='Here we predict prices based on the roomsNumber, area and year signs, then evaluate the accuracy of the model.'
                'Accuracy measures how well a model matches data.',
        style={
            'textAlign': 'center',
            'padding': '0 30'
        }
    ),
    html.Div([
        html.Table([
            html.Tbody([
                html.Tr([
                    html.Td('Predict value: '),
                    html.Td(str(model.predict([X_test[9]]))),
                ]),
                html.Tr([
                    html.Td('Real value: '),
                    html.Td(str(y_test[9])),
                ]),
                html.Tr([
                    html.Td('Accuracy: '),
                    html.Td(model.score(X_test, y_test) * 100),
                ])
            ])
        ], style={'padding': 10}),
    ], style={
        'padding': '10px 50px'}),
    html.Br(),

    html.P(
        children='Here shows the statistical characteristics of the model, where  the dependent variable is price, and the independent variables are roomsnumber and area.'
                'In this case, the first and second signs (roomsNumber and area) have p-values equal to 0.12 and 0.03 respectively, which means that they may be statistically insignificant.'
                'However, the third sign (year) has a very low p-value of approximately 0.000001278, which indicates its statistical significance.'
                 'The coefficients show the magnitude of the impact of each attribute on the price of real estate.'
                 'Standard errors show how accurate the coefficient estimates are.',
        style={
            'textAlign': 'center',
            'padding': '0 30'
        }
    ),
    html.Div([
        html.Table([
            html.Tbody([
                html.Tr([
                    html.Td('P Values:'),
                    html.Td(str(pValue[0]) + " " + str(pValue[1]) + " " + str(pValue[2]))
                ]),
                html.Tr([
                    html.Td('Coefficient'),
                    html.Td(str(coef[0]) + " " + str(coef[1]) + " " + str(coef[2])),
                ]),
                html.Tr([
                    html.Td('Standard Error:'),
                    html.Td(str(stdError[0]) + " " + str(stdError[1]) + " " + str(stdError[2])),
                ])
            ])
        ], style={'padding': 10})
    ], style={
        'padding': '10px 50px'}),
    html.Br(),




    html.Div(children=[
        html.Div([
            dcc.Graph(figure=histogram_dist)
        ], style={'padding': 10, 'flex': 1}),

        html.Div([
            dcc.Graph(figure=histogram_year)
        ], style={'padding': 10, 'flex': 1})
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    html.P(
        children='The first histogram above shows the distribution of apartments for sale in different Almaty districts. It is clear that the most popular one is "Bostandyk", then "Almaly" and "Auezov".',
        style={
            'textAlign': 'center',
            'padding': '0 30'
        }
    ),
    html.P(
        children='The second histogram above shows the distribution of apartments built in different years of Almaty. It is clear that the construction is actively growing in 2020-2029 than in 2010-2019.',
        style={
            'textAlign': 'center',
            'padding': '0 30'
        }
    ),
    html.Div([
        dcc.Graph(figure=histogram_dist_area_rn)
    ], style={'textAlign': 'center',  'padding': '0 20'}),

    html.P(
        children='Here you can see the maximum number of rooms is in the Turksib district with an area of 199 and in the Medeu district with an area of 215',
        style={
            'textAlign': 'center',
            'padding': '0 30'
        }
    ),

    html.Div([
        dcc.Graph(figure=fig)
    ], style={'width': '60%', 'display': 'inline-block', 'padding': '0 0'}),

    html.P(
        children='The above statistics shows that mean price is 74 millions, mean area is 77 square meters and mean rooms number is 2.4. The cheapest appartment in his sample has 12 square meters and 1 room, it costs about 5.7 millions. While, the most expensive one has 6 rooms, 315 square meters and costs more than 589 millions tenge.',
        style={
            'textAlign': 'center',
            'padding': '0 30'
        }
    ),

    html.Br(),
    html.Div([
        dcc.Graph(figure=fig_corr)
    ], style={'width': '80%', 'display': 'inline-block', 'padding': '0 20'}),

    html.P(
        children='Heatmap which visualizes a matrix of correlations between variables. Here each element shows the degree of connection between two variables',
        style={
            'textAlign': 'center',
            'padding': '0 30'
        }
    ),


    html.Div([


        html.Br(),


        dcc.Dropdown(
            ['area', 'price', 'roomsNumber', 'year'],
            'area',
            id='crossfilter-xaxis-column',
        )

    ],
        style={'width': '30%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'customdata': 'Japan'}]}
        )
    ], style={'width': '80%', 'display': 'inline-block', 'padding': '0 20'}),
])

@app.callback(
    Output('crossfilter-indicator-scatter', 'figure'),
    Input('crossfilter-xaxis-column', 'value'))
def update_graph(xaxis_column_name):
    fig = px.scatter(data, x=xaxis_column_name, y='price', trendline="ols", trendline_color_override="red")

    return fig
if __name__ == '__main__':
    app.run_server(debug=True)