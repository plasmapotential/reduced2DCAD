#reducedCADapp.py


from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


#DASH server
app = Dash(__name__)
#app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])
#Create our own server for downloading files

#CSS stylesheets
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    },
    'bigApp': {
        'max-width': '100%',
        'display': 'flex',
        'flex-direction': 'row',
        'width': '100vw',
        'height': '95vh',
        'vertical-align': 'middle',
        'justify-content': 'center',
    },
    'column': {
        'width': '45%',
        'height': '100%',
        'display': 'flex',
        'flex-direction': 'column',
        'justify-content': 'center',
    },
    'graph': {
#        'display': 'flex',
        'width': '100%',
        'height': '90%',
        'justify-content': 'center',
    },
    'btnRow': {
        'height': '10%',
        'width': '100%',
        'justify-content': 'center',
    },
    'button': {
        'width': '10%',
        'justify-content': 'center',
    },
    'table': {
        'width': '45%',
        'height': '90%',
        'overflowY': 'scroll',
    },
}
def generateLayout(fig, df):
    #generate HTML5 application
    app.layout = html.Div([
        #data storage object
        dcc.Store(id='colorData', storage_type='memory'),
        #graph Div
        html.Div([
            html.Div([
                dcc.Graph(
                    id='polyGraph',
                    figure=fig,
                    style=styles['graph']
                ),
                html.Div([
                    html.Label("Group ID:", style={'margin':'0 10px 0 10px'}),
                    dcc.Input(id="grp", style=styles['button']),
                    ],
                    style=styles['btnRow']
                    ),
                ],
                style=styles['column']
                ),
            html.Div([
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i}
                        for i in df.columns],
                data=df.to_dict('records'),
                export_format="csv",
                style_cell=dict(textAlign='left'),
                style_header=dict(backgroundColor="paleturquoise"),
                style_data=dict(backgroundColor="lavender")
                ),
                ],
                style=styles['table'],
                ),
            ],
            style=styles['bigApp']
            ),
        ],
        )


@app.callback(
    [Output('polyGraph', 'figure'),
     Output('table', 'data'),
     Output('colorData', 'data')],
    Input('polyGraph', 'selectedData'),
    [State('table', 'data'),
     State('grp', 'value'),
     State('colorData', 'data'),
     State('polyGraph', 'figure')]
    )
def color_selected_data(selectedData, tableData, group, colorData, fig):
    """
    colors selected mesh cells based upon group ID
    """
    #selected data is None on page load, dont fire callback
    if selectedData is not None:
        #user must input a group ID
        if group == None:
            print("You must enter a value for group!")
            raise PreventUpdate
        #get mesh elements in selection
        ids = []
        for i,pt in enumerate(selectedData['points']):
            ids.append(pt['curveNumber'])
        #initialize colorData dictionary
        if colorData is None:
            colorData = {}
        #loop thru IDs of selected, assigning color by group
        ids = np.array(np.unique(ids))
        for ID in ids:
            dataDict = fig['data'][ID]
            if ('line' in dataDict) or ('color' in dataDict):
                fig['data'][ID]['line']['color'] = '#9834eb'
                if group == None:
                    group = 0
                #also update the table
                try:
                    tableData[ID]['GroupID'] = group
                except: #contour traces will not have tableData
                    print("Group ID "+str(ID)+" not found in table!")
                if group in colorData:
                    fig['data'][ID]['line']['color'] = colorData[group]
                else:
                    colorData[group] = px.colors.qualitative.Plotly[len(colorData)]
                    fig['data'][ID]['line']['color'] = colorData[group]
    return fig, tableData, colorData
