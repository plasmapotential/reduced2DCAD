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
from dash_bootstrap_templates import ThemeSwitchAIO, ThemeChangerAIO
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
    #FreeCADPath = '/usr/lib/freecad-python3/lib'
    FreeCADPath = '/usr/lib/freecad-daily-python3/lib'
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
    'banner': {
        'display': 'flex',
        'flex-direction': 'column',
        'width': '100vw',
        'height': '5vh',
    },
    'entireWindow': {
        'max-width': '100%',
        'display': 'flex',
        'flex-direction': 'column',
        'width': '100vw',
        'height': '95vh',
        'vertical-align': 'middle',
        'justify-content': 'center',
    },
    'bigApp': {
        'max-width': '100%',
        'display': 'flex',
        'flex-direction': 'row',
        'width': '100%',
        'height': '95%',
        'vertical-align': 'bottom',
        'align-items': 'center',
        'justify-content': 'center',
    },
    'banner': {
        'width': '100%',
        'height': '5%',
        'display': 'flex',
        'flex-direction': 'row',
        'justify-content':'flex-end',
        'vertical-align':'middle',
    },
    'col2': {
        'width': '45%',
        'height': '100%',
        'max-height': '90%',
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
        'vertical-align':'center',
        'align-items':'center',
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
    #generate an id column for dataframe
    df['id'] = df.index

    #generate HTML5 application
    app.layout = html.Div([
        dcc.Location(id='url'),
        #data storage object
        dcc.Store(id='sizeStore', storage_type='memory'),
        dcc.Store(id='pTableData', storage_type='memory'),
        dcc.Store(id='contourTraces', storage_type='memory'),
        dcc.Store(id='meshTraces', storage_type='memory'),
        dcc.Store(id='mainTraces', storage_type='memory'),
        dcc.Store(id='meshColorTraces', storage_type='memory'),
        dcc.Store(id='idData', storage_type='memory'),
        dcc.Store(id='mapData', storage_type='memory'),

        html.Div([themeToggle()], style=styles['banner']),

        #App Div
        html.Div([
            html.Div([
                dbc.Accordion(
                    [
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
                                        for i in df.columns
                                        #omit id column
                                        if i != 'id'
                                        ],
                                    data=df.to_dict('records'),
                                    export_format="csv",
                                    style_cell=dict(textAlign='left'),
                                    row_deletable=True,
                                    #row_selectable='multi',
                                    selected_rows=[],
                                    editable=False,

                                    #style_header=dict(backgroundColor="paleturquoise"),
                                    #style_data=dict(backgroundColor="lavender")
                                    ),
                        ],
                    title="Table",
                    ),
                    dbc.AccordionItem(
                        [
                            html.Div([
                                html.H6("Environment Settings"),
                                html.Label(children="HEAT Path: "),
                                dbc.Input(id="HEATpath", value=HEAT, readonly=True),
                                html.Label(children="Path to Source Code: "),
                                dbc.Input(id="sourcePath", value=sourcePath,  readonly=True),
                                html.Label(children="Path to FreeCAD libs: "),
                                dbc.Input(id="freeCADPath", value=FreeCADPath,  readonly=True),
                                html.Label(children="Path to save output: "),
                                dbc.Input(id="outPath", value=filesPath,  readonly=True),
                                ],
                                style=styles['column']
                                )

                        ],
                    title="Environment (read only)",
                    ),

                    ],
                    style=styles['table'],
                    active_item="CAD",
                    ),

            ],
            style=styles['column'],
            ),
            html.Br(),
            html.Div([
                dcc.Graph(
                    id='polyGraph',
                    figure=fig,
                    style=styles['graph'],
                    config={'displaylogo':False,},
                ),
                ],
                style=styles['col2']
                ),
            ],
            style=styles['bigApp']
            ),
        ],
        style=styles['entireWindow']
        )

def themeToggle():
    """
    returns a div with theme toggling
    """

    themes_list = [
        dbc.themes.COSMO,
        dbc.themes.SLATE
        ]

    themeToggle = html.Div(
        [
            #for toggling between two themes
            #ThemeChangerAIO(aio_id="theme", themes=themes_list, )
            #for switching between multiple themes
            dbc.Row(ThemeChangerAIO(aio_id="theme",
                                    radio_props={"value":dbc.themes.COSMO},
                                    offcanvas_props={'placement':'end'}
                                    ),
                    ),
        ],
        #style=styles['themeToggle'],
    )

    return themeToggle


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
            dbc.Input(id="phi", value=0),
            html.Label(children="Rmax of cutting plane [mm] (width=0 to Rmax): "),
            dbc.Input(id="rMax", value=5000),
            html.Label(children="Zrange of cutting plane [mm] (height above z=0): "),
            dbc.Input(id="zMax", value=10000),
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
        if hasattr(CAD3D, 'CAD') != True:
            print("No CAD loaded.  Load CAD before sectioning.")
            raise PreventUpdate
        #create cross section
        CAD2D.sectionParams(float(rMax),float(zMax),float(phi),discreteTog)
        CAD2D.sectionCAD(CAD3D.CAD)
        CAD2D.buildContourList(CAD3D.CAD)
        traces = CAD2D.rzTraces()

    return [html.Label("Sectioned CAD"), traces]


def meshCreateDiv():
    """
    create mesh div object
    """
    div = html.Div([
                html.H6("Create New Mesh"),
                html.Label("Grid Size [mm]:", style={'margin':'0 10px 0 10px'}),
                dbc.Input(id="gridSize"),
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
               State('phi','value'),]
               )
def loadGrid(n_clicks, fileName, contents, gridSize, meshTraces, meshToggles, meshToggleVal, phi):
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

        head = ['Rc', 'Zc', 'L', 'W', 'AC1', 'AC2', 'NL', 'NW', 'material', 'caf', 'isf']
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

        options = []
        #append mesh names for toggle switching
        for i,mesh in enumerate(meshes):
            if mesh.meshType != 'file':
                name = mesh.meshType + " {:0.1f}mm at {:0.1f}\u00B0 ".format(mesh.grid_size, mesh.phi)
            else:
                name = mesh.file
            options.append({'label': name, 'value': i})

        toggleVals = i

        #options.append({'label': name, 'value': i})
        status = html.Label("Loaded Mesh From File")

    #not loading from file
    else:
        t0 = time.time()
        type = 'square'
        try:
            name = type + '{:.3f}'.format(float(gridSize))
        except:
            print("Could not load Grid Size input.  Check text box before proceeding.")
            raise PreventUpdate
        #check if this mesh is already in the meshList
        test1 = [m.meshType==type for m in meshes]
        test2 = [m.grid_size==float(gridSize) for m in meshes]
        test3 = [m.phi==float(phi) for m in meshes]
        test = np.logical_and(np.logical_and(test1, test2), test3)

        #if mesh isnt already calculated
        if np.any(test) == False:
            mesh = RC.mesh()

            mesh.loadMeshParams(type, float(gridSize), float(phi), None)
            mesh.createSquareMesh(CAD2D.contourList, gridSize)
            meshes.append(mesh)

            #mesh overlay
            if meshTraces == None:
                meshTraces = []
            out = mesh.getMeshTraces()
            meshTraces.append(out)

            #mesh checkboxes
            options = []

            #append mesh names for toggle switching
            for i,mesh in enumerate(meshes):
                if mesh.meshType != 'file':
                    name = mesh.meshType + " {:0.1f}mm at {:0.1f}\u00B0 ".format(mesh.grid_size, mesh.phi)
                else:
                    name = mesh.file
                options.append({'label': name, 'value': i})
            #append the last index
            toggleVals = i

            status = html.Label("Loaded Mesh")

        else:
            traces = meshTraces
            options = meshToggles
            toggleVals = meshToggleVal
            status = html.Label("Mesh Already Loaded")


        print("Mesh Execution Time: {:f} seconds".format(time.time()-t0))
        print("Number of mesh elements: {:d}".format(len(meshTraces[-1])))

    return [status, meshTraces, options, toggleVals]


def meshDisplayDiv():
    """
    mesh display options div object
    """
    div = html.Div([
                html.H6("Available Meshes: "),
                dbc.RadioItems(
                    options=[

                        ],
                        value=None,
                        id='meshToggles',

                        ),
                html.Hr(),
                html.Br(),
                html.H6("Main Mesh: "),
                dbc.Checklist(
                    options=[],
                        value=[],
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
              [Input('addSelect', 'n_clicks'),
               Input('assignID', 'n_clicks'),
               Input('combine', 'n_clicks'),
               Input('table', 'data_previous'),],
              [State('meshTraces', 'data'),
               State('mainTraces', 'data'),
               State('meshToggles', 'value'),
               State('mapData', 'data'),
               State('idData', 'data'),
               State('outPath', 'value'),
               State('table', 'data'),
               State('polyGraph', 'relayoutData'),
               State('matGrp', 'value'),
               State('nlGrp', 'value'),
               State('nwGrp', 'value'),
               State('cafGrp', 'value'),
               State('isfGrp', 'value'),
               ],
               )
def add2Main(n_clicks_select, n_clicks_assign, n_clicks_combine, prev_tableData,
             meshTraces, mainTraces, toggleVal, mapData,
             idData, outPath,tableData, shapeData,
             matGrp,nlGrp,nwGrp,cafGrp,isfGrp
             ):
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

    df = pd.DataFrame({'Rc[m]':[], 'Zc[m]':[], 'L[m]':[], 'W[m]':[], 'AC1[deg]':[], 'AC2[deg]':[], 'NL':[], 'NW':[], 'material':[], 'caf':[], 'isf':[]})
    if button_id == None:
        raise PreventUpdate

    elif button_id == 'addSelect':
        if shapeData is not None:
            if mainTraces == None:
                mainTraces = []

            mesh = meshes[toggleVal]
            mT = meshTraces[toggleVal]

            xLo,xHi,yLo,yHi = mesh.getRectangleBounds(shapeData)

            z = np.zeros((mapData['Ny'],mapData['Nx']))
            for m in mT:
                #test if each element m is inside box
                xM0 = np.min(np.array(m)[:,0])
                xM1 = np.max(np.array(m)[:,0])
                yM0 = np.min(np.array(m)[:,1])
                yM1 = np.max(np.array(m)[:,1])
                if xM0 > xLo and xM1 < xHi and yM0 > yLo and yM1 < yHi:
                    mainTraces.append(m)
                    #get ptable data for this m
                    gridSizes.append([mesh.grid_size, mesh.grid_size])
                    ctrX = (xM1-xM0)/2.0 + xM0
                    ctrY = (yM1-yM0)/2.0 + yM0
                    centroids.append([ctrX,ctrY])

            #get pTable
            pTableOut = meshes[0].shapelyPtables(centroids,
                                                    outPath,
                                                    gridSizes,
                                                    tableData,
                                                    )

            #create pandas df
            df = meshes[0].createDFsFromCSVs(pTableOut)[0]
            df['id'] = df.index
            tableData = df.to_dict('records')
            print("Updated Table and Added to Main")
        else:
            print("No Shape Data!")


    #assign group ID to main mesh elements
    elif button_id == 'assignID':
        if n_clicks_assign == None:
            raise PreventUpdate

        #selected data is None on page load, dont fire callback
        if shapeData is not None:
            #user must input a group ID
            if matGrp==None or nlGrp==None or nwGrp==None or cafGrp==None or isfGrp==None:
                print("No properties entered...")
                raise PreventUpdate

            xLo,xHi,yLo,yHi = meshes[0].getRectangleBounds(shapeData)

            minR = 10000.0
            maxR = 0.0
            minZ = 10000.0
            maxZ = 0.0

            idxs = []
            for i,m in enumerate(mainTraces):
                #test if each element m is inside box
                xM0 = np.min(np.array(m)[:,0])
                xM1 = np.max(np.array(m)[:,0])
                yM0 = np.min(np.array(m)[:,1])
                yM1 = np.max(np.array(m)[:,1])

                if xM0 > xLo and xM1 < xHi and yM0 > yLo and yM1 < yHi:
                    try:
                        tableData[i]['material'] = matGrp
                        tableData[i]['NL'] = nlGrp
                        tableData[i]['NW'] = nwGrp
                        tableData[i]['caf'] = cafGrp
                        tableData[i]['isf'] = isfGrp

                    except: #contour traces will not have tableData
                        print("Could not assign properties!")

    elif button_id == 'combine':
        if n_clicks_combine == None:
            raise PreventUpdate
        if shapeData is not None:
            if mainTraces == None:
                mainTraces = []

            xLo,xHi,yLo,yHi = meshes[0].getRectangleBounds(shapeData)

            minR = 10000.0
            maxR = 0.0
            minZ = 10000.0
            maxZ = 0.0

            idxs = []
            oldTable = []
            for i,m in enumerate(mainTraces):
                #test if each element m is inside box
                xM0 = np.min(np.array(m)[:,0])
                xM1 = np.max(np.array(m)[:,0])
                yM0 = np.min(np.array(m)[:,1])
                yM1 = np.max(np.array(m)[:,1])

                if xM0 > xLo and xM1 < xHi and yM0 > yLo and yM1 < yHi:
                    idxs.append(i)
                    #find bounds of mesh in box
                    if xM0 < minR: minR = xM0
                    if xM1 > maxR: maxR = xM1
                    if yM0 < minZ: minZ = yM0
                    if yM1 > maxZ: maxZ = yM1
            #remove small mesh elements
            for i in reversed(idxs):
                #solutions.pop(i)
                centroids.pop(i)
                gridSizes.pop(i)
                mainTraces.pop(i)
                oldTable.append(tableData[i])
                tableData.pop(i)
            #combine into big element
            grid, trace, tData, ctrs = meshes[0].combineElements(oldTable)
            solutions.append(None)
            centroids.append(ctrs)
            gridSizes.append(grid)
            mainTraces.append(trace)
            tableData.append(tData)


    elif button_id == 'table':
        if prev_tableData is None:
            raise PreventUpdate
        else:
            removedRow = [row for row in prev_tableData if row not in tableData]
            removedIdx = prev_tableData.index(removedRow[0])
            mainTraces.pop(removedIdx)
            centroids.pop(removedIdx)
            gridSizes.pop(removedIdx)

    options = [{'label':'Main Mesh', 'value':0}]
    value = [0]

    return [mainTraces, options, value, tableData]


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
            html.H6("Edit Main Mesh:"),
            dbc.Button("Add selection to main mesh", color="primary", id="addSelect"),
            html.Hr(),
            dbc.Button("Combine selected elements", color="primary", id="combine"),
            html.Div(id="hiddenDivMainOps"),
            html.Hr(),
            html.H6("Change element properties"),
            html.Label("Material:", style={'margin':'0 10px 0 10px'}),
            dbc.Input(id="matGrp"),
            html.Label("NL:", style={'margin':'0 10px 0 10px'}),
            dbc.Input(id="nlGrp"),
            html.Label("NW:", style={'margin':'0 10px 0 10px'}),
            dbc.Input(id="nwGrp"),
            html.Label("caf:", style={'margin':'0 10px 0 10px'}),
            dbc.Input(id="cafGrp"),
            html.Label("isf:", style={'margin':'0 10px 0 10px'}),
            dbc.Input(id="isfGrp"),
            dbc.Button("Assign properties to selection", color="primary", id="assignID"),

        ],
        style=styles['column']
        )
    return div

#Update the graph
@app.callback([Output('polyGraph', 'figure'),
               Output('polyGraph', 'config'),
               Output('idData', 'data'),
               Output('mapData', 'data')],
              [Input('contourTraces', 'data'),
               Input('meshTraces', 'data'),
               Input('meshToggles','value'),
               Input('mainTraces', 'data'),
               Input('mainToggle', 'value'),
               Input('table', 'selected_cells'),],
               [State('table', 'data'),
                State('sizeStore', 'data')]
               )
def updateGraph(contourTraces, meshTraces, toggleVals, mainTraces, mainToggle,
                activeCells, tableData, size):
    """
    updates the figure.  the figure contains a heatmap and a scatter trace

    heatmap is a regular grid defined by the smallest mesh element size, and
    represents the mesh

    scatter trace is the contour plots of the toroidal slices

    """
    idData = {}
    mapData = {}
    idx1 = 0
    minGrid = 10000.0 #default
    xMin = 1000.0
    xMax = 3000.0
    yMin = -2000.0
    yMax = 2000.0
    trigger = ctx.triggered_id
    fig = go.Figure()

    if trigger == None:
        raise PreventUpdate

    #trigger conditions
    meshTriggers = ['meshTraces', 'meshToggles']
    mainTriggers = ['mainTraces', 'mainToggle', 'meshTraces', 'meshToggles', 'table']
    contourTriggers = ['contourTraces','meshToggles','meshTraces', 'mainTraces', 'mainToggle', 'table']

    cs2 = [[0, 'white'], [0.25, 'white'],
            [0.25, 'blue'], [0.5, 'blue'],
            [0.5, 'seagreen'], [0.75, 'seagreen'],
            [0.75, 'red'], [1.0, 'red']]


    #build the grid
    if trigger in meshTriggers or trigger in mainTriggers:
        for i,mesh in enumerate(meshes):
            #find the minimum mesh size in all meshes, to build grid
            if mesh.grid_size < minGrid:
                minGrid = mesh.grid_size
        #calculate which grid cells we have mesh elements in
        Nx = int((xMax-xMin)/minGrid)
        Ny = int((yMax-yMin)/minGrid)
        if Nx==0 or Ny==0: #nothing loaded yet
            raise PreventUpdate
        dx = (xMax - xMin)/Nx
        dy = (yMax - yMin)/Ny
        x = np.linspace(xMin, xMax, Nx+1)
        y = np.linspace(yMin, yMax, Ny+1)
        z = np.zeros((Ny,Nx))

        #mesh traces
        mesh = meshTraces[toggleVals]
        for m in mesh:
            x0 = np.min(np.array(m)[:,0])
            x1 = np.max(np.array(m)[:,0])
            y0 = np.min(np.array(m)[:,1])
            y1 = np.max(np.array(m)[:,1])
            xLo = int((x0-xMin)/dx)
            xHi = int((x1-xMin)/dx)
            yLo = int((y0-yMin)/dy)
            yHi = int((y1-yMin)/dy)
            z[yLo:yHi,xLo:xHi] = 0.35

            mapData['minGrid'] = minGrid
            mapData['Nx'] = Nx
            mapData['Ny'] = Ny
            mapData['dx'] = dx
            mapData['dy'] = dy
            mapData['z'] = z

        if trigger in mainTriggers and len(mainToggle)>0:
            #build mainmesh traces
            for i,mesh in enumerate(mainTraces):
                x0 = np.min(np.array(mesh)[:,0])
                x1 = np.max(np.array(mesh)[:,0])
                y0 = np.min(np.array(mesh)[:,1])
                y1 = np.max(np.array(mesh)[:,1])
                xLo = int((x0-xMin)/dx)
                xHi = int((x1-xMin)/dx)
                yLo = int((y0-yMin)/dy)
                yHi = int((y1-yMin)/dy)
                z[yLo:yHi,xLo:xHi] = 0.6

        if trigger == 'table':
            if activeCells is not None:
                rows = [int(a['row_id']) for a in activeCells]
                for row in rows:
                    mesh = mainTraces[row]
                    x0 = np.min(np.array(mesh)[:,0])
                    x1 = np.max(np.array(mesh)[:,0])
                    y0 = np.min(np.array(mesh)[:,1])
                    y1 = np.max(np.array(mesh)[:,1])
                    xLo = int((x0-xMin)/dx)
                    xHi = int((x1-xMin)/dx)
                    yLo = int((y0-yMin)/dy)
                    yHi = int((y1-yMin)/dy)
                    z[yLo:yHi,xLo:xHi] = 0.9

        fig.add_trace(go.Heatmap(x=x,y=y,z=z, opacity=0.4, showscale=False,
                              colorscale=cs2, zmin=0.0, zmax=1.0))

    if trigger in contourTriggers:
        for trace in contourTraces:
            fig.add_trace(go.Scatter(x=np.array(trace)[:,0], y=np.array(trace)[:,1], mode='lines'))



    idData = []

    fig.update_layout(showlegend=False)
    fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)
    fig.update_traces()
    fig.update_layout(dragmode="drawrect")
    #this allows zoom to be preserved across button clicks
    fig.update_layout(uirevision='neverUpdate')

    config = {
    "modeBarButtonsToAdd": [
        "drawrect",
        "eraseshape",
    ]}

    return [fig, config, idData, mapData]


app.clientside_callback(
    """
    function(href) {
        var w = window.innerWidth;
        var h = window.innerHeight;
        return {'height': h, 'width': w};
    }
    """,
    Output('sizeStore', 'data'),
    Input('url', 'href')
)
