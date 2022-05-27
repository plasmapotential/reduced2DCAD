#reducedCADapp.py

import os
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

#load HEAT CADclass interface objects
import reducedCADClasses as RC
CAD3D = RC.CAD3D()
CAD2D = RC.CAD2D()
meshes = []

#default figure and table
#fig = go.Figure()
#df = pd.DataFrame({'Rc[m]':[], 'Zc[m]':[], 'L[m]':[], 'W[m]':[], 'AC1[deg]':[], 'AC2[deg]':[], 'GroupID':[]})

#environment variables for tom's development
try:
    runMode = os.environ['runMode']
except:
    runMode = 'dev'

print("Running in environment: "+runMode)

#inputs
if runMode == 'docker':
    #for use in docker container:
    HEAT = '/root/source/HEAT'
    path = '/root/files/'
    STPfile = path + 'VVcompsAdjusted.step'
    STP2D = path + '2Dout.step'
else:
    #for use in tom's dev env
    path = '/home/tom/work/CFS/projects/reducedCAD/'
    STPfile = path + 'vacVes.step'
    #STPfile = path + 'VVcompsAdjusted.step'
    STP2D = path + '2Dout.step'
    HEAT = '/home/tom/source/HEAT/github/source'



#DASH server
app = Dash(__name__, external_stylesheets=[dbc.themes.MATERIA])
load_figure_template('lux')
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
        #'width': '90%',
        'height': '80%',
        'overflowY': 'scroll',
        'overflowX': 'scroll',
    },
    'upload': {
        'width': '60%', 'height': '60px', 'lineHeight': '60px',
        'borderWidth': '1px', 'borderStyle': 'dashed',
        'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px',
        'justify-content': 'center',
        },
}

def generateLayout(fig, df):
    #generate HTML5 application
    app.layout = html.Div([
        #data storage object
        dcc.Store(id='colorData', storage_type='memory'),
        dcc.Store(id='contourTraces', storage_type='memory'),
        dcc.Store(id='meshTraces', storage_type='memory'),
        dcc.Store(id='meshColorTraces', storage_type='memory'),

        #graph Div
        html.Div([
            html.Div([
                dbc.Accordion(
                    [
                    dbc.AccordionItem(
                        [
                            html.Div([
                                html.H6("Environment Settings"),
                                html.Label(children="HEAT Path: "),
                                dcc.Input(id="HEATpath", value=HEAT),
                                html.Label(children="Path to CAD file: "),
                                dcc.Input(id="CADpath", value=path),
                                html.Label(children="Path to save output: "),
                                dcc.Input(id="outPath", value=path),
                                ],
                                style=styles['column']
                                )

                        ],
                    title="Environment",
                    ),
                    dbc.AccordionItem(
                        [
                            CADdiv()
                        ],
                    title="CAD",
                    ),
                    dbc.AccordionItem(
                        [
                            sectionDiv()
                        ],
                    title="Section",
                    ),
                    dbc.AccordionItem(
                        [
                            meshDiv()
                        ],
                    title="Mesh",
                    ),
                    dbc.AccordionItem(
                        [
                                dash_table.DataTable(
                                    id='table',
                                    columns=[{"name": i, "id": i}
                                        for i in df.columns],
                                    data=df.to_dict('records'),
                                    export_format="csv",
                                    style_cell=dict(textAlign='left'),
                                    #style_header=dict(backgroundColor="paleturquoise"),
                                    #style_data=dict(backgroundColor="lavender")
                                    ),
                        ],
                    title="Table",
                    ),
                    ],
                    style=styles['table'],
                    ),

            ],
            style=styles['column'],
            ),
            html.Div([
                dcc.Graph(
                    id='polyGraph',
                    figure=fig,
                    style=styles['graph'],
                    config={'displaylogo':False,},
                ),
                ],
                style=styles['column']
                ),
            ],
            style=styles['bigApp']
            ),
        ],
        )


@app.callback(
    [Output('meshColorTraces', 'data'),
     Output('table', 'data'),
     Output('colorData', 'data')],
    Input('assignID', 'n_clicks'),
    [State('polyGraph', 'selectedData'),
     State('table', 'data'),
     State('grp', 'value'),
     State('colorData', 'data'),
     State('polyGraph', 'figure')]
    )
def color_selected_data(n_clicks, selectedData, tableData, group, colorData, fig):
    """
    colors selected mesh cells based upon group ID
    """
    if n_clicks == None:
        raise PreventUpdate
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



def CADdiv():
    """
    CAD accordian page
    """
    div = html.Div([
            dcc.Upload(
            id='CAD-upload',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select STP file')
            ]),
            style=styles['upload'],
            multiple=False,
            ),
            html.Div(id="hiddenDivCAD"),

        ])
    return div

#Load CAD button connect
@app.callback([Output('hiddenDivCAD', 'children')],
              [Input('CAD-upload', 'filename')],
              [State('HEATpath', 'value'),
               State('CADpath', 'value'),]
               )
def loadCAD(STPfile, HEATpath, CADpath):
    if STPfile is None:
        raise PreventUpdate
    else:
        #Load HEAT environment
        CAD3D.loadHEAT(HEATpath)
        #Load STP file
        CAD3D.loadSTPfile(CADpath + STPfile)

    return [html.Label("Loaded CAD: "+STPfile)]



def sectionDiv():
    """
    div for sectioning CAD
    """
    div = html.Div([
            html.Label(children="Toroidal angle (phi) of section [degrees]: "),
            dcc.Input(id="phi", value=0),
            html.Label(children="Rmax of cutting plane [mm]: "),
            dcc.Input(id="rMax", value=5000),
            html.Label(children="Zmax of cutting plane [mm]: "),
            dcc.Input(id="zMax", value=10000),
            dbc.Button("Section CAD at phi", color="primary", id="loadSection"),
            html.Div(id="hiddenDivSection"),
        ],
        style=styles['column']
        )
    return div

#Section button connect
@app.callback([Output('hiddenDivSection', 'children'),
               Output('contourTraces', 'data')],
              [Input('loadSection', 'n_clicks')],
              [State('rMax', 'value'),
               State('zMax', 'value'),
               State('phi', 'value'),
               State('polyGraph', 'figure')]
               )
def loadSection(n_clicks, rMax, zMax, phi, fig):
    if n_clicks == None or n_clicks < 1:
        raise PreventUpdate
    else:
        #create cross section
        CAD2D.sectionParams(float(rMax),float(zMax),float(phi))
        CAD2D.sectionCAD(CAD3D.CAD)
        CAD2D.buildContourList(CAD3D.CAD)
        traces = CAD2D.getContourTraces()

    return [html.Label("Sectioned CAD"), traces]


def meshDiv():
    """
    mesh div object
    """
    div = html.Div([
                html.H6("Create New Mesh"),
                html.Label("Grid Size [mm]:", style={'margin':'0 10px 0 10px'}),
                dcc.Input(id="gridSize"),
                dbc.Button("Create Grid", color="primary", id="loadGrid"),
                html.Hr(),
                html.H6("Mesh Operations:"),
                html.Label("Group ID:", style={'margin':'0 10px 0 10px'}),
                dcc.Input(id="grp"),
                dbc.Button("Assign ID to selection", color="primary", id="assignID"),
                html.Div(id="hiddenDivMesh"),
                ],
            style=styles['column']
            )
    return div

#create grid button connect
@app.callback([Output('hiddenDivMesh', 'children'),
               Output('meshTraces', 'data')],
              [Input('loadGrid', 'n_clicks')],
              [State('gridSize', 'value'),
               State('meshTraces', 'data')]
               )
def loadGrid(n_clicks, gridSize, meshTraces):
    if n_clicks is None:
        raise PreventUpdate
    else:
        type = 'square'
        name = type + '{:.3f}'.format(float(gridSize))
        #check if this mesh is already in the meshList
        if np.any([m.grid_size==gridSize for m in meshes]) != True:
            mesh = RC.mesh()
            mesh.loadMeshParams(type, float(gridSize))
            mesh.createSquareMesh(CAD2D.contourList, gridSize)
            meshes.append(mesh)

        #mesh overlay
        traces = mesh.getMeshTraces(meshTraces)
    return [html.Label("Loaded Mesh"), traces]




#Update the graph
@app.callback([Output('polyGraph', 'figure')],
              [Input('contourTraces', 'data'),
               Input('meshTraces', 'data')],
               )
def updateGraph(contourTraces, meshTraces):
    #if contourTraces == None and meshTraces == None:
    #    raise PreventUpdate

    fig = go.Figure()
    if contourTraces != None:
        for trace in contourTraces:
            fig.add_trace(trace)
    if meshTraces != None:
        for trace in meshTraces:
            fig.add_trace(trace)

    fig.update_layout(showlegend=False)
    fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)
    return [fig]
