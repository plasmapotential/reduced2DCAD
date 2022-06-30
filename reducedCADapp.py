#reducedCADapp.py
#Description:   Dash python bindings to HTML5/JS/CSS for reduced 2D app
#Engineer:      T Looby
#Date:          20220519

import os
import sys
import io
import base64
from dash import Dash, dcc, html, dash_table, ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import time

#load HEAT CADclass interface objects
import reducedCADClasses as RC
CAD3D = RC.CAD3D()
CAD2D = RC.CAD2D()

#globals
meshes = []
centroids = []
solutions = []
gridSizes = []
mainMap = []

#environment variables for tom's development
try:
    runMode = os.environ['runMode']
except:
    runMode = 'dev'

print("Running in environment: "+runMode)

#inputs
if runMode == 'docker':
    #for use in docker container:
    sourcePath = '/root/source/reduced2DCAD'
    HEAT = '/root/source/HEAT'
    filesPath = '/root/files/'
    #STPfile = path + 'VVcompsAdjusted.step'
    #STP2D = path + '2Dout.step'
    FreeCADPath = '/usr/lib/freecad-python3/lib'
else:
    #for use in tom's dev env
    sourcePath = '/home/tom/source/reduced2DCAD/github'
    filesPath = '/home/tom/work/CFS/projects/reducedCAD'
    #STPfile = path + 'vacVes.step'
    #STPfile = path + 'VVcompsAdjusted.step'
    #STP2D = path + '2Dout.step'
    HEAT = '/home/tom/source/HEAT/github/source'
    FreeCADPath = '/usr/lib/freecad-daily/lib'

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
        'width': '80%', 'height': '60px', 'lineHeight': '60px',
        'borderWidth': '1px', 'borderStyle': 'dashed',
        'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px',
        'justify-content': 'center',
        },
}

def generateLayout(fig, df):
    #generate HTML5 application
    app.layout = html.Div([
        #data storage object
        dcc.Store(id='pTableData', storage_type='memory'),
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
                                html.Label(children="Path to Source Code: "),
                                dcc.Input(id="sourcePath", value=sourcePath),
                                html.Label(children="Path to FreeCAD libs: "),
                                dcc.Input(id="freeCADPath", value=FreeCADPath),
                                html.Label(children="Path to save output: "),
                                dcc.Input(id="outPath", value=filesPath),
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
                            mainOpsDiv()
                        ],
                    title="Main Mesh Operations",
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
                                    row_deletable=True,
                                    selected_rows=[],
                                    #style_header=dict(backgroundColor="paleturquoise"),
                                    #style_data=dict(backgroundColor="lavender")
                                    ),
                        ],
                    title="Table",
                    ),
                    ],
                    style=styles['table'],
                    active_item="CAD",
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
              [State('CAD-upload', 'contents'),
               State('HEATpath', 'value'),
               State('sourcePath', 'value'),
               State('freeCADPath', 'value'),]
               )
def loadCAD(STPfile, STPdata, HEATpath, sourcePath, FreeCADPath):
    if STPfile is None:
        raise PreventUpdate
    else:
        content_type, content_string = STPdata.split(',')
        decoded = base64.b64decode(content_string)
        f = '/tmp/loadedCAD.step' #tmp location
        with open(f, 'wb') as file:
            file.write(decoded)



        global runMode
        if runMode == 'local':
            #Load HEAT environment
            CAD3D.loadHEAT(HEATpath)
        else:
            #Load CAD environment only
            CAD3D.loadHEATCADenv(sourcePath, FreeCADPath)

        #Load STP file
        #CAD3D.loadSTPfile(CADpath + STPfile)
        CAD3D.loadSTPfile(f)

    return [html.Label("Loaded CAD: "+STPfile)]



def sectionDiv():
    """
    div for sectioning CAD
    """
    div = html.Div([
            html.Label(children="Toroidal angle (phi) of section [degrees]: "),
            dcc.Input(id="phi", value=0),
            html.Label(children="Rmax of cutting plane [mm] (width=0 to Rmax): "),
            dcc.Input(id="rMax", value=5000),
            html.Label(children="Zrange of cutting plane [mm] (height above z=0): "),
            dcc.Input(id="zMax", value=10000),
            html.Hr(),
            dbc.Checklist(
                options=[{'label':"Discretize Curves?", 'value':1}],
                    value=[1],
                    id='discreteToggle',
                    switch=True,
                    ),
            html.Hr(),
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
               State('polyGraph', 'figure'),
               State('discreteToggle', 'value')]
               )
def loadSection(n_clicks, rMax, zMax, phi, fig, discreteTog):
    if n_clicks == None or n_clicks < 1:
        raise PreventUpdate
    else:
        #create cross section
        CAD2D.sectionParams(float(rMax),float(zMax),float(phi),discreteTog)
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
                html.H6("Load Mesh From File"),
                dcc.Upload(
                    id='pTableUpload',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Load pTable')
                        ]),
                    style=styles['upload'],
                    multiple=False,
                    ),
                ],
            style=styles['column']
            )
    return div


#create grid button connect
@app.callback([Output('hiddenDivMesh', 'children'),
               Output('meshTraces', 'data'),
               Output('meshToggles', 'options'),
               Output('meshToggles', 'value')],
              [Input('loadGrid', 'n_clicks'),
               Input('pTableUpload', 'filename'),
               Input('pTableUpload', 'contents')],
              [State('gridSize', 'value'),
               State('meshTraces', 'data'),
               State('meshToggles', 'options'),
               State('meshToggles', 'value'),
               State('phi','value'),
               State('polyGraph', 'selectedData'),]
               )
def loadGrid(n_clicks, fileName, contents, gridSize, meshTraces, meshToggles, meshToggleVals, phi, selected):
    """
    creates a grid and updates meshData storage object and toggles
    """
    trigger = ctx.triggered_id
    if n_clicks is None and fileName is None:
        raise PreventUpdate
    elif trigger=='pTableUpload':
        mesh = RC.mesh()
        mesh.file = fileName
        mesh.meshType = 'file'
        mesh.grid_size = None
        mesh.phi = None
        mesh.selection = []
        meshes.append(mesh)

        head = ['Rc', 'Zc', 'L', 'W', 'AC1', 'AC2', 'GroupID']
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        try:
            if 'csv' in fileName:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), header=0, names=head)
            elif 'xls' in fileName:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))
        except Exception as e:
            print(e)

        if meshTraces == None:
            meshTraces = []

        meshTraces.append(mesh.tracesFromPtable(df))
        name = fileName

        if meshToggleVals == None or None in meshToggleVals or len(meshToggleVals)==0:
            i=0
            toggleVals = [i]
        else:
            i = max(meshToggleVals)+1
            meshToggleVals.append(i)
            toggleVals = meshToggleVals

        options = []
        #append mesh names for toggle switching
        for i,mesh in enumerate(meshes):
            if mesh.meshType != 'file':
                name = mesh.meshType + " {:0.1f}mm at {:0.1f}\u00B0 ".format(mesh.grid_size, mesh.phi)
            else:
                name = mesh.file
            options.append({'label': name, 'value': i})

        #options.append({'label': name, 'value': i})
        status = html.Label("Loaded Mesh From File")

    #not loading from file
    else:
        t0 = time.time()
        type = 'square'
        name = type + '{:.3f}'.format(float(gridSize))
        #check if this mesh is already in the meshList
        test1 = [m.meshType==type for m in meshes]
        test2 = [m.grid_size==float(gridSize) for m in meshes]
        test3 = [m.phi==float(phi) for m in meshes]
        test = np.logical_and(np.logical_and(test1, test2), test3)

        #if mesh isnt already calculated
        if np.any(test) == False:
            mesh = RC.mesh()

            #get bounds of selection
            if selected != None:
                bounds = selected['range']
            else:
                bounds = None

            mesh.loadMeshParams(type, float(gridSize), float(phi), bounds)
            mesh.createSquareMesh(CAD2D.contourList, gridSize)
            meshes.append(mesh)

            #mesh overlay
            #traces = mesh.getMeshTraces(meshTraces)
            if meshTraces == None:
                meshTraces = []
            meshTraces.append(mesh.getMeshTraces())

            #mesh checkboxes
            options = []

            if None in meshToggleVals:
                toggleVals = []
            else:
                toggleVals = meshToggleVals
            #append mesh names for toggle switching
            for i,mesh in enumerate(meshes):
                if mesh.meshType != 'file':
                    name = mesh.meshType + " {:0.1f}mm at {:0.1f}\u00B0 ".format(mesh.grid_size, mesh.phi)
                else:
                    name = mesh.file
                options.append({'label': name, 'value': i})
            #append the last index
            if i not in meshToggleVals:
                toggleVals.append(i)

            status = html.Label("Loaded Mesh")

        else:
            if selected == None:
                traces = meshTraces
                options = meshToggles
                toggleVals = meshToggleVals
                status = html.Label("Mesh Already Loaded")

            #new selection on existing mesh
            else:
                print("Creating new selection on existing mesh")
                #find mesh
                idx = np.where(test==True)[0][0]
                meshes[idx].bounds = selected['range']

                #meshTraces[idx] = meshTraces[idx] + meshes[idx].getMeshTraces()
                meshTraces[idx] = meshes[idx].getMeshTraces()

                #mesh checkboxes
                options = []
                if None in meshToggleVals:
                    toggleVals = []
                else:
                    toggleVals = meshToggleVals
                #append mesh names for toggle switching
                for i,mesh in enumerate(meshes):
                    if mesh.meshType != 'file':
                        name = mesh.meshType + " {:0.1f}mm at {:0.1f}\u00B0 ".format(mesh.grid_size, mesh.phi)
                    else:
                        name = mesh.file
                    options.append({'label': name, 'value': i})


                #append the index
                if idx not in meshToggleVals:
                    toggleVals.append(idx)

                status = html.Label("Loaded Mesh")

        print("Mesh Execution Time: {:f} seconds".format(time.time()-t0))
        print("Number of mesh elements: {:d}".format(len(meshTraces[-1])))

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
                dbc.Button("Add all existing meshes to main mesh", color="primary", id="addAll"),
                html.Hr(),
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
               Output('table', 'data'),
               Output('colorData', 'data')],
              [Input('addAll', 'n_clicks'),
               Input('addSelect', 'n_clicks'),
               Input('assignID', 'n_clicks'),
               Input('combine', 'n_clicks'),
               Input('table', 'data_previous'),],
              [State('meshTraces', 'data'),
               State('mainTraces', 'data'),
               State('polyGraph', 'selectedData'),
               State('idData', 'data'),
               State('outPath', 'value'),
               State('grp', 'value'),
               State('colorData', 'data'),
               State('table', 'data'),
               ],
               )
def add2Main(n_clicks_all, n_clicks_select, n_clicks_assign, n_clicks_combine, prev_tableData,
             meshTraces, mainTraces, selected,
             idData, outPath, group, colorData, tableData):
    """
    add to main mesh callbacks

    results in additions to the pTables and figures
    """
    global meshes
    global centroids
    global solutions
    global gridSizes
    global mainMap

    button_id = ctx.triggered_id

    df = pd.DataFrame({'Rc[m]':[], 'Zc[m]':[], 'L[m]':[], 'W[m]':[], 'AC1[deg]':[], 'AC2[deg]':[], 'GroupID':[]})
    if button_id == None:
        raise PreventUpdate
    elif button_id == 'addAll':
        mainTraces =  [m for sub in meshTraces for m in sub]
        solutions = []
        gridSizes = []
        centroids = []
        for m in meshes:
            for s in m.solutions:
                for i,g in enumerate(s.geoms):
                    solutions.append(g)
                    centroids.append(np.array(g.centroid))
                    if m.meshType != 'file':
                        gridSizes.append(m.grid_size)
                    else:
                        gridSizes.append(m.grid_size[i])
        #get pTable
        pTableOut = meshes[0].shapelyPtables(centroids,
                                             outPath,
                                             gridSizes,
                                             tableData,
                                             )
        #create pandas df
        df = meshes[0].createDFsFromCSVs(pTableOut)[0]
        tableData = df.to_dict('records')

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
                #loop thru IDs in selection from figure
                for id in np.unique(ids):
                    #loop thru all meshes to find selected mesh elements
                    for idx,m in enumerate(meshTraces):
                        #map mesh trace elements back to figure trace elements
                        mappedID = id - idData['meshStarts'][idx]

                        #check if this trace is in the mesh
                        if id in idData['meshIdxs'][idx]:
                            #if there is a selection on the mesh, we need to map
                            #the mappedID of the figure back to only the selected
                            #elements of the mesh displayed (not all mesh elements)
                            if len(meshes[idx].selection) == 0:
                                selectMap = mappedID #no selection
                            else:
                                #mapping from figure selection to mesh.selection
                                selectMap = meshes[idx].selection[mappedID]

                            #if we havent initialized main mesh yet
                            if 'mainIdxs' not in list(idData.keys()):
                                mainMap.append(id)
                                mainTraces.append(m[mappedID])
                                solutions.append(meshes[idx].geoms[selectMap])
                                centroids.append(np.array(meshes[idx].geoms[selectMap].centroid))
                                if meshes[idx].meshType != 'file':
                                    gridSizes.append(meshes[idx].grid_size)
                                else:
                                    gridSizes.append(meshes[idx].grid_size[selectMap])

                            #mainMesh is already initialized
                            else:
                                mainMap.append(id)
                                mainTraces.append(m[mappedID])
                                solutions.append(meshes[idx].geoms[selectMap])
                                centroids.append(np.array(meshes[idx].geoms[selectMap].centroid))
                                if meshes[idx].meshType != 'file':
                                    gridSizes.append(meshes[idx].grid_size)
                                else:
                                    gridSizes.append(meshes[idx].grid_size[selectMap])
#                                if id not in mainMap:
#                                    mainMap.append(id)
#                                    mainTraces.append(m[mappedID])
#                                    solutions.append(meshes[idx].geoms[selectMap])
#                                    centroids.append(np.array(meshes[idx].geoms[selectMap].centroid))
#                                    if meshes[idx].meshType != 'file':
#                                        gridSizes.append(meshes[idx].grid_size)
#                                    else:
#                                        gridSizes.append(meshes[idx].grid_size[selectMap])
#                                else:
#                                    print('ID in mainMap')
                #get pTable
                pTableOut = meshes[0].shapelyPtables(centroids,
                                                     outPath,
                                                     gridSizes,
                                                     tableData,
                                                     )

                #create pandas df
                df = meshes[0].createDFsFromCSVs(pTableOut)[0]
                tableData = df.to_dict('records')

    #assign group ID to main mesh elements
    elif button_id == 'assignID':
        if n_clicks_assign == None:
            raise PreventUpdate
        #selected data is None on page load, dont fire callback
        if selected is not None:
            #user must input a group ID
            if group == None:
                print("You must enter a value for group!")
                raise PreventUpdate
            #get mesh elements in selection
            ids = []
            for i,pt in enumerate(selected['points']):
                ids.append(pt['curveNumber'])
            #initialize colorData dictionary
            if colorData is None:
                colorData = {}
            #loop thru IDs of selected, assigning color by group
            ids = np.array(np.unique(ids))
            for ID in ids:
                if ID in idData['mainIdxs']:
                    idx = idData['mainIdxs'].index(ID)
                    if group == None:
                        group = 0
                    #also update the table
                    try:
                        tableData[idx]['GroupID'] = group
                    except: #contour traces will not have tableData
                        print("Group ID "+str(idx)+" not found in table!")
                    if group not in colorData:
                        colorData[group] = px.colors.qualitative.Plotly[len(colorData)]

    elif button_id == 'combine':
        if n_clicks_combine == None:
            raise PreventUpdate
        if selected is not None:
            #get mesh elements in selection
            ids = []
            for i,pt in enumerate(selected['points']):
                ids.append(pt['curveNumber'])
            #loop thru IDs of selected, assigning color by group
            ids = np.array(np.unique(ids))
            idxs = []

            #remove mesh elements we are combining
            #reverse so we dont mess up the list index as we go
            for ID in sorted(ids, reverse=True):
                if ID in idData['mainIdxs']:
                    idx = idData['mainIdxs'].index(ID)
                    idxs.append(idx)
                    solutions.pop(idx)
                    centroids.pop(idx)
                    gridSizes.pop(idx)
                    mainTraces.pop(idx)

            #combine elements into single element
            idxs = np.array(sorted(idxs))
            grid, trace, tData, ctrs = meshes[0].combineElements(np.array(tableData)[idxs])
            solutions.append(None)
            centroids.append(ctrs)
            gridSizes.append(grid)
            mainTraces.append(trace)

            #remove elements that were combined from pTable
            for ID in sorted(ids, reverse=True):
                if ID in idData['mainIdxs']:
                    idx = idData['mainIdxs'].index(ID)
                    tableData.pop(idx)
            tableData.append(tData)

    elif button_id == 'table':
        if prev_tableData is None:
            raise PreventUpdate
        else:
            removedRow = [row for row in prev_tableData if row not in tableData]
            removedIdx = prev_tableData.index(removedRow[0])

            #mainIdx = mainMap[removedIdx]
            mainTraces.pop(removedIdx)
            #idData['mainIdxs'].pop(removedIdx)
            solutions.pop(removedIdx)
            centroids.pop(removedIdx)
            gridSizes.pop(removedIdx)
            #mainMap.pop(removedIdx)

    options = [{'label':'Main Mesh', 'value':0}]
    value = [0]

    #initialize colorData dictionary
    if colorData is None:
        #default color
        colorData = {'default':'#d059ff'}
        colorData['selected'] = '#000000'

    return [mainTraces, options, value, tableData, colorData]


##table data
#@app.callback([Output('table', 'data')],
#              [Input('pTableData', 'data')],
#               )
#def updateTable(data):
#    """
#    updates dash table when data storage object changes
#    """
#    if data == None:
#        raise PreventUpdate
#    return data



def mainOpsDiv():
    """
    div for operations on main mesh
    """
    div = html.Div([
            dbc.Button("Combine selected elements", color="primary", id="combine"),
            html.Div(id="hiddenDivMainOps"),
            html.Hr(),
            html.H6("Group elements by ID"),
            html.Label("Group ID:", style={'margin':'0 10px 0 10px'}),
            dcc.Input(id="grp"),
            dbc.Button("Assign ID to selection", color="primary", id="assignID"),

        ],
        style=styles['column']
        )
    return div

#Update the graph
@app.callback([Output('polyGraph', 'figure'),
               Output('idData', 'data')],
              [Input('contourTraces', 'data'),
               Input('meshTraces', 'data'),
               Input('meshToggles','value'),
               Input('mainTraces', 'data'),
               Input('mainToggle', 'value'),
               Input('colorData', 'data'),
               Input('table', 'active_cell'),],
               [State('table', 'data')]
               )
def updateGraph(contourTraces, meshTraces, toggleVals, mainTraces, mainToggle, colorData, activeCell, tableData):
    """
    updates the figure.  the figure contains a single list of all the traces,
    and we need to know if these traces are contours, meshes, or main mesh.
    to do this we use idData dict.

    idData contains a 'start' index for each of those groups (contours, meshes,
    main meshes), and the indices within those groups can be mapped back to
    the figure indices via these 'start' values

    idData also contains an 'Idxs' key:value pair, where the value is a list
    of all of the figure indexes that correspond to this group (contours, meshes,
    main meshes)
    """
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
        activeRow = activeCell['row'] if activeCell else None
        idxs = []
        idxCombined = []
        for i,trace in enumerate(mainTraces):
            idx2 = 0
            if 0 in mainToggle:
                #assign color based upon GroupID
                if activeRow!=None and i==activeRow:
                    trace['line']['color'] = colorData['selected']
                    trace['opacity'] = 1.0
                else:
                    if tableData[i]['GroupID'] != 0:
                        trace['line']['color'] = colorData[tableData[i]['GroupID']]
                    else:
                        trace['line']['color'] = colorData['default']
                fig.add_trace(trace)
                idxs.append(idx1+idx2)
                idx2 += 1
            idx1 = idx1 + idx2
        idData['mainIdxs'] = idxs


    fig.update_layout(showlegend=False)
    fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)
    return [fig, idData]
