import reducedCADClasses as RC
import argparse
import os
import plotly.graph_objects as go
import pandas as pd


rMax = 5000
zMax = 10000
phi = 150.0 #degrees
gridSize = 50.0


#

#

#
##create square mesh on top of contours
#mesh = RC.mesh('square' , gridSize)
#polygons, solutions = mesh.createSquareMesh(CAD2D.contourList, gridSize)
#pTableFiles, pTableAll = mesh.shapelyPtables(solutions, path)
#df = mesh.createDFsFromCSVs(pTableAll)[0]
#

##contourList plots
#fig = CAD2D.getContourPlot()




fig = go.Figure()
df = pd.DataFrame({'Rc[m]':[], 'Zc[m]':[], 'L[m]':[], 'W[m]':[], 'AC1[deg]':[], 'AC2[deg]':[], 'GroupID':[]})
#launch the GUI
import reducedCADapp as GUI
GUI.generateLayout(fig, df)

if __name__ == '__main__':
    #parse command line arguments
    parser = argparse.ArgumentParser(description=""" Use this command to launch crossSections2Mesh """)
    parser.add_argument('--a', type=str, help='IP address ', required=False)
    parser.add_argument('--p', type=str, help='port # ', required=False)
    args = parser.parse_args()
    address = vars(args)['a']
    port = vars(args)['p']
    #use default IPv4 address and port unless user provided one
    if address == None:
        address = '127.0.0.1' #default
    if port == None:
        port = 8050 #default
    GUI.app.run_server(
                debug=True,
                dev_tools_ui=True,
                port=port,
                host=address,
                use_reloader=False, #this can be used in local developer mode only
                dev_tools_hot_reload = False, #this can be used in local developer mode only
                )
