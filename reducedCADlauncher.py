import reducedCADClasses as RC
import argparse
#inputs
#for use in docker container:
#HEATpath = '/root/source/HEAT'
#path = '/root/files/'
#STPfile = path + 'VVcompsAdjusted.step'
#STP2D = path + '2Dout.step'

#for use in tom's dev env
path = '/home/tom/work/CFS/projects/reducedCAD/'
STPfile = path + 'VVcompsAdjusted.step'
STP2D = path + '2Dout.step'
HEATpath = '/home/tom/source/HEAT/github/source'

rMax = 5000
zMax = 10000
phi = 32.0 #degrees
gridSize = 50.0

#Load HEAT environment
CAD3D = RC.CAD3D(HEATpath)

#Load STP file
CAD3D.loadSTPfile(STPfile)

#create cross section
CAD2D = RC.CAD2D(rMax,zMax,phi)
CAD2D.sectionCAD(CAD3D.CAD)
CAD2D.buildContourList(CAD3D.CAD)

#create square mesh on top of contours
mesh = RC.mesh('square' , gridSize)
polygons, solutions = mesh.createSquareMesh(CAD2D.contourList, gridSize)
pTableFiles, pTableAll = mesh.shapelyPtables(solutions, path)
df = mesh.createDFsFromCSVs(pTableAll)[0]

#launch the GUI
import reducedCADapp as GUI
#contourList plots
fig = CAD2D.getContourPlot()
#mesh overlay
fig = mesh.addMeshPlots2Fig(fig, solutions)

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
