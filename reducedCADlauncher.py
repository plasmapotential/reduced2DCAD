import reducedCADClasses as RC

#inputs
HEATpath = '/home/tom/source/HEAT/github/source'
STPfile = '/home/tom/work/CFS/projects/reducedCAD/VVcompsAdjusted.step'
STP2D = '/home/tom/work/CFS/projects/reducedCAD/2Dout.step'
path = '/home/tom/work/CFS/projects/reducedCAD/'
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
#fig = mesh.addMeshPlots2Fig(fig, solutions)

GUI.generateLayout(fig, df)

address = '127.0.0.1' #default
port = 8050 #default
GUI.app.run_server(
                debug=True,
                dev_tools_ui=True,
                port=port,
                host=address,
                use_reloader=False, #this can be used in local developer mode only
                dev_tools_hot_reload = False, #this can be used in local developer mode only
                )
