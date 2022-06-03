#reducedCADapp.py
#Description:   Dash python bindings to HTML5/JS/CSS for reduced 2D app
#Engineer:      T Looby
#Date:          20220519

import os
from dash import Dash, dcc, html, dash_table, ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

#load HEAT CADclass interface objects
import reducedCADClasses as RC
CAD3D = RC.CAD3D()
CAD2D = RC.CAD2D()

#globals
meshes = []
solutions = []
gridSizes = []

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
        dcc.Store(id='mainTraces', storage_type='memory'),
        dcc.Store(id='meshColorTraces', storage_type='memory'),
        dcc.Store(id='idData', storage_type='memory'),

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
                            meshCreateDiv()
                        ],
                    title="Create Mesh",
                    ),
                    dbc.AccordionItem(
                        [
                            meshDisplayDiv()
                        ],
                    title="Mesh Display",
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


#@app.callback(
#    [Output('meshColorTraces', 'data'),
#     Output('table', 'data'),
#     Output('colorData', 'data')],
#    Input('assignID', 'n_clicks'),
#    [State('polyGraph', 'selectedData'),
#     State('table', 'data'),
#     State('grp', 'value'),
#     State('colorData', 'data'),
#     State('polyGraph', 'figure')]
#    )
#def color_selected_data(n_clicks, selectedData, tableData, group, colorData, fig):
#    """
#    colors selected mesh cells based upon group ID
#    """
#    if n_clicks == None:
#        raise PreventUpdate
#    #selected data is None on page load, dont fire callback
#    if selectedData is not None:
#        #user must input a group ID
#        if group == None:
#            print("You must enter a value for group!")
#            raise PreventUpdate
#        #get mesh elements in selection
#        ids = []
#        for i,pt in enumerate(selectedData['points']):
#            ids.append(pt['curveNumber'])
#        #initialize colorData dictionary
#        if colorData is None:
#            colorData = {}
#        #loop thru IDs of selected, assigning color by group
#        ids = np.array(np.unique(ids))
#        for ID in ids:
#            dataDict = fig['data'][ID]
#            if ('line' in dataDict) or ('color' in dataDict):
#                fig['data'][ID]['line']['color'] = '#9834eb'
#                if group == None:
#                    group = 0
#                #also update the table
#                try:
#                    tableData[ID]['GroupID'] = group
#                except: #contour traces will not have tableData
#                    print("Group ID "+str(ID)+" not found in table!")
#                if group in colorData:
#                    fig['data'][ID]['line']['color'] = colorData[group]
#                else:
#                    colorData[group] = px.colors.qualitative.Plotly[len(colorData)]
#                    fig['data'][ID]['line']['color'] = colorData[group]
#    return fig, tableData, colorData



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


def meshCreateDiv():
    """
    create mesh div object
    """
    div = html.Div([
                html.H6("Create New Mesh"),
                html.Label("Grid Size [mm]:", style={'margin':'0 10px 0 10px'}),
                dcc.Input(id="gridSize"),
                dbc.Button("Create Grid", color="primary", id="loadGrid"),
                html.Div(id="hiddenDivMesh"),
                html.Hr(),
                html.H6("Mesh Operations:"),
                html.Label("Group ID:", style={'margin':'0 10px 0 10px'}),
                dcc.Input(id="grp"),
                dbc.Button("Assign ID to selection", color="primary", id="assignID"),
                ],
            style=styles['column']
            )
    return div

#create grid button connect
@app.callback([Output('hiddenDivMesh', 'children'),
               Output('meshTraces', 'data'),
               Output('meshToggles', 'options'),
               Output('meshToggles', 'value')],
              [Input('loadGrid', 'n_clicks')],
              [State('gridSize', 'value'),
               State('meshTraces', 'data'),
               State('meshToggles', 'options'),
               State('meshToggles', 'value'),
               State('phi','value'),]
               )
def loadGrid(n_clicks, gridSize, meshTraces, meshToggles, meshToggleVals, phi):
    if n_clicks is None:
        raise PreventUpdate
    else:
        type = 'square'
        name = type + '{:.3f}'.format(float(gridSize))
        #check if this mesh is already in the meshList
        test1 = [m.meshType==type for m in meshes]
        test2 = [m.grid_size==float(gridSize) for m in meshes]
        test3 = [m.phi==float(phi) for m in meshes]
        test = np.logical_and(np.logical_and(test1, test2), test3)

        if np.any(test) == False:
            mesh = RC.mesh()
            mesh.loadMeshParams(type, float(gridSize), float(phi))
            mesh.createSquareMesh(CAD2D.contourList, gridSize)
            meshes.append(mesh)

            #mesh overlay
            #traces = mesh.getMeshTraces(meshTraces)
            if meshTraces == None:
                meshTraces = []
            meshTraces.append(mesh.getMeshTraces())

            #mesh checkboxes
            options = []
            toggleVals = meshToggleVals
            for i,mesh in enumerate(meshes):
                name = mesh.meshType + " {:0.1f}mm at {:0.1f}\u00B0 ".format(mesh.grid_size, mesh.phi)
                options.append({'label': name, 'value': i})

            #append the last index
            if i not in meshToggleVals:
                toggleVals.append(i)



            status = html.Label("Loaded Mesh")
        else:
            traces = meshTraces
            options = meshToggles
            toggleVals = meshToggleVals
            status = html.Label("Mesh Already Loaded")

    return [status, meshTraces, options, toggleVals]


def meshDisplayDiv():
    """
    mesh display options div object
    """
    div = html.Div([
                html.H6("Available Meshes: "),
                dbc.Checklist(
                    options=[

                        ],
                        value=[None],
                        id='meshToggles',
                        switch=True,
                        ),
                html.Br(),
                dbc.Button("Add all to main mesh", color="primary", id="addAll"),
                dbc.Button("Add selection to main mesh", color="primary", id="addSelect"),
                html.Br(),
                html.H6("Main Mesh: "),
                dbc.Checklist(
                    options=[],
                        value=[None],
                        id='mainToggle',
                        switch=True,
                        ),
                ],
            style=styles['column']
            )
    return div


#main mesh
@app.callback([Output('mainTraces', 'data'),
               Output('mainToggle', 'options'),
               Output('mainToggle', 'value'),
               Output('table', 'data')],
              [Input('addAll', 'n_clicks'),
               Input('addSelect', 'n_clicks')],
              [State('meshTraces', 'data'),
               State('mainTraces', 'data'),
               State('polyGraph', 'selectedData'),
               State('idData', 'data'),
               State('outPath', 'value')],
               )
def add2Main(n_clicks_all, n_clicks_select, meshTraces, mainTraces, selected, idData, outPath):
    """
    add to main button callbacks
    """
    global meshes
    global solutions
    global gridSizes

    button_id = ctx.triggered_id
    df = pd.DataFrame({'Rc[m]':[], 'Zc[m]':[], 'L[m]':[], 'W[m]':[], 'AC1[deg]':[], 'AC2[deg]':[], 'GroupID':[]})
    if button_id == None:
        raise PreventUpdate
    elif button_id == 'addAll':
        mainTraces =  [m for sub in meshTraces for m in sub]
        solutions = []
        gridSizes = []
        for m in meshes:
            for s in m.solutions:
                for g in s.geoms:
                    solutions.append(g)
                    gridSizes.append(m.grid_size)
        #get pTable
        pTableOut = meshes[0].shapelyPtables(solutions,
                                             outPath,
                                             gridSizes,
                                             mainOnly=True,
                                             )
        #create pandas df
        df = meshes[0].createDFsFromCSVs(pTableOut)[0]

    elif button_id == 'addSelect':
        if selected is not None:
            if mainTraces == None:
                mainTraces = []
            #get mesh elements in selection
            ids = []
            for i,pt in enumerate(selected['points']):
                id = int(pt['curveNumber'])
                ids.append(id)

            #add mesh elements in selection to main mesh if they are mesh elements
            #and aren't already in the main mesh
            if 'meshIdxs' not in list(idData.keys()):
                print("No meshes displayed.  Doing nothing")
            else:
                #loop thru IDs in selection
                for id in np.unique(ids):
                    #loop thru all meshes
                    for idx,m in enumerate(meshTraces):
                        #map this mesh trace elements back to figure trace elements
                        mappedID = id - idData['meshStarts'][idx]
                        #if this trace is in the mesh
                        if id in idData['meshIdxs'][idx]:
                            solutions.append(meshes[idx].geoms[mappedID])
                            gridSizes.append(meshes[idx].grid_size)

                            if 'mainIdxs' not in list(idData.keys()):
                                mainTraces.append(m[mappedID])
                            else:
                                if id not in idData['mainIdxs']:
                                    mainTraces.append(m[mappedID])

                #get pTable
                pTableOut = meshes[0].shapelyPtables(solutions,
                                                     outPath,
                                                     gridSizes,
                                                     mainOnly=True,
                                                     )
                #create pandas df
                df = meshes[0].createDFsFromCSVs(pTableOut)[0]

    options = [{'label':'Main Mesh', 'value':0}]
    value = [0]

    return [mainTraces, options, value, df.to_dict('records')]


#Update the graph
@app.callback([Output('polyGraph', 'figure'),
               Output('idData', 'data')],
              [Input('contourTraces', 'data'),
               Input('meshTraces', 'data'),
               Input('meshToggles','value'),
               Input('mainTraces', 'data'),
               Input('mainToggle', 'value')],
               )
def updateGraph(contourTraces, meshTraces, toggleVals, mainTraces, mainToggle):
    #if contourTraces == None and meshTraces == None:
    #    raise PreventUpdate
    idData = {}
    idx1 = 0
    fig = go.Figure()
    if contourTraces != None:
        idxs = []
        idx2 = 0
        idData['contourStart'] = idx1
        for i,trace in enumerate(contourTraces):
            fig.add_trace(trace)
            idxs.append(idx1+idx2)
            idx2 += 1
        idx1 = idx1 + idx2
        idData['contourIdxs'] = idxs

    if meshTraces != None:

        idData['meshIdxs'] = []
        #trace index where each independent mesh starts
        idData['meshStarts'] = []
        for i,mesh in enumerate(meshTraces):
            idxs = []
            idData['meshStarts'].append(idx1)
            if i in toggleVals:
                idx2 = 0
                for j,trace in enumerate(mesh):
                    fig.add_trace(trace)
                    idxs.append(idx1+idx2)
                    idx2 += 1
                idx1 = idx1 + idx2
            idData['meshIdxs'].append(idxs)

    if mainTraces != None:
        idxs = []
        idData['mainStart'] = idx1
        for i,trace in enumerate(mainTraces):
            idx2 = 0
            if 0 in mainToggle:
                trace['line']['color'] = '#a122f5'
                fig.add_trace(trace)
                idxs.append(idx1+idx2)
                idx2 += 1
            idx1 = idx1 + idx2
        idData['mainIdxs'] = idxs

    fig.update_layout(showlegend=False)
    fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)
    return [fig, idData]
