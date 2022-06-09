#reducedCADClasses.py
#Description:
#Engineer:      T Looby
#Date:          20220519

from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
import numpy as np
from shapely.geometry import MultiPoint, MultiPolygon, Polygon, LinearRing
import plotly.graph_objects as go
import os
import sys
import argparse

class CAD3D:

    def __init__(self):
        return

    def loadHEAT(self, HEATpath):
        """
        loads the HEAT environment, loads CADclass object into self
        """
        sys.path.append(HEATpath)

        #load HEAT environment
        import launchHEAT
        launchHEAT.loadEnviron()

        #load HEAT CAD module and STP file
        import CADClass
        self.CAD = CADClass.CAD(os.environ["rootDir"], os.environ["dataPath"])
        return

    def loadSTPfile(self, STPfile):
        self.CAD.STPfile = STPfile
        print(self.CAD.STPfile)
        self.CAD.permute_mask = False
        print("Loading CAD.")
        print("For large CAD files, loading may take a few minutes...")
        self.CAD.loadSTEP()
        return



class CAD2D:

    def __init__(self):
        return


    def sectionParams(self, rMax, zMax, phi):
        """
        initialize 2D cross section object
        """
        self.rMax = rMax
        self.zMax = zMax
        self.phi = phi
        return

    def sectionCAD(self, CAD3D):
        """
        sections CAD using plane of width rMax, zMax at toroidal angle phi

        requires a CAD3D object, which contains the 3D STP model

        slices is all the CAD objects resulting from section/intersection
        """
        #get poloidal cross section at user specified toroidal angle
        print("Number of part objects in CAD: {:d}".format(len(CAD3D.CADparts)))
        self.slices = CAD3D.getPolCrossSection(self.rMax,self.zMax,self.phi)
        print("Number of part objects in section: {:d}".format(len(self.slices)))
        return

    def save2DSTEP(self, file, CAD):
        """
        saves a 2D step file with all the slices

        file is file to save
        CAD is 3D CAD object from HEAT
        """
        CAD.saveSTEP(file, self.slices)
        return

    def saveSliceCSVs(self, csvPath):
        """
        saves a CSV for each slice

        requires a rootPath where all csvs will be saved
        """
        for i,slice in enumerate(self.slices):
            xyz = np.array([ [v.X, v.Y, v.Z] for v in slice.Shape.Vertexes])
            R = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2)
            Z = xyz[:,2]
            rz = np.vstack([R,Z]).T
            f = csvPath + '/slice{:03d}.csv'.format(i)
            head = 'R[mm], Z[mm]'
            np.savetxt(f, rz, delimiter=',', header=head)
        return

    def buildContourList(self, CAD):
        """
        build an ordered list of vertices that comprise a contour
        """
        contourList = []
        edgeList = []
        for slice in self.slices:
            edgeList = CAD.getVertexesFromEdges(slice.Shape.Edges)
            contours = CAD.findContour(edgeList)
            if len(contours) > 0:
                contourList.append(contours)



#            if len(contours)!=0:
#                #make the outermost contour the 1st list element
#                idx = 0
#                xMax = 0
#                yMax = 0
#                zMax = 0
#                for i,c in enumerate(contours):
#                    if max(c[:,0]) > xMax:
#                        xMax = max(c[:,0])
#                        idx = i
#                    elif max(c[:,1]) > yMax:
#                        yMax = max(c[:,1])
#                        idx = i
#                    if max(c[:,2]) > zMax:
#                        zMax = max(c[:,2])
#                        idx = i
#
#                contours.insert(0, contours.pop(idx))
#                #now append with idxOuter as 0 index
#                contourList.append(contours)



        self.contourList = contourList
        return


    def getContourPlot(self, fig=None):
        """
        returns a fig with the contours in contourList
        """
        if fig == None:
            fig = go.Figure()

        for slice in self.contourList:
            for c in slice:
                R = np.sqrt(c[:,0]**2+c[:,1]**2)
                Z = c[:,2]
                fig.add_trace(go.Scatter(x=R, y=Z, mode='lines+markers'))
        return fig

    def getContourTraces(self):
        """
        returns a list of contour traces
        """
        traces = []
        for slice in self.contourList:
            for c in slice:
                R = np.sqrt(c[:,0]**2+c[:,1]**2)
                Z = c[:,2]
                traces.append(go.Scatter(x=R, y=Z, mode='lines+markers'))
        return traces


class mesh:

    def __init__(self):
        return

    def loadMeshParams(self, meshType, gridSize, phi):
        self.meshType = meshType
        self.grid_size = gridSize
        self.phi = phi
        return

    def square(self, x, y, s):
        return Polygon([(x, y), (x+s, y), (x+s, y+s), (x, y+s)])

    def createSquareMesh(self, contourList, grid_size):
        """
        creates a square mesh over the contours in contourList
        using the Shapely library
        """
        polygons = []
        solutions = []
        for c in contourList:
            outerContour = []
            holeContours = []
            #check if this contour has any holes
            cDict = self.findEdgesAndHoles(c)
            N_edges = np.sum([d['type']=='edge' for d in cDict])
            N_holes = np.sum([d['type']=='hole' for d in cDict])

            #loop thru all contours in this c and build meshes accordingly
            #uses hole and edge data from cDict
            for i,d in enumerate(cDict):
                if d['type'] == 'hole':
                    pass
                else:
                    outerContour = c[i]
                    R_out = np.sqrt(outerContour[:,0]**2+outerContour[:,1]**2)
                    Z_out = outerContour[:,2]

                    #find holes for this outer contour
                    holeIdxs = np.where([h['edgeIdx']==i for h in cDict])[0]
                    holes = []
                    for h in holeIdxs:
                        R_hole = np.sqrt(c[h][:,0]**2+c[h][:,1]**2)
                        Z_hole = c[h][:,2]
                        #create a ring of the hole
                        holeRing = LinearRing(np.vstack([R_hole,Z_hole]).T)
                        holes.append(holeRing)

                    #use polygon
                    poly = Polygon(np.vstack([R_out,Z_out]).T, holes)
                    polyCoords = np.array(poly.exterior.coords)
                    ibounds = np.array(poly.bounds)//self.grid_size
                    ibounds[2:4] += 1
                    xmin, ymin, xmax, ymax = ibounds*self.grid_size
                    xrg = np.arange(xmin, xmax, self.grid_size)
                    yrg = np.arange(ymin, ymax, self.grid_size)
                    mp = MultiPolygon([self.square(x, y, self.grid_size) for x in xrg for y in yrg])
                    solution = MultiPolygon(list(filter(poly.intersects, mp)))

                    polygons.append(poly)
                    solutions.append(solution)
        self.polygons = polygons
        self.solutions = solutions
        geoms = []
        for s in self.solutions:
            for g in s.geoms:
                geoms.append(g)
        self.geoms = geoms
        return

    def findEdgesAndHoles(self, contours):
        """
        finds edges and holes for a given list of CAD contours

        use the 2D plane centroids to see if we are outside of the bounding
        box for each of the contours
        """
        print("Finding edges and holes")
        N = len(contours)
        centroids = np.zeros((N,2))
        mins = np.zeros((N,2))
        maxs = np.zeros((N,2))

        Rs = []
        Zs = []
        for c in contours:
            Rs.append(np.sqrt(c[:,0]**2+c[:,1]**2))
            Zs.append(c[:,2])

        for i,c in enumerate(contours):
            centroids[i,0] = np.sum(Rs[i])/len(Rs[i])
            centroids[i,1] = np.sum(Zs[i])/len(Zs[i])
            mins[i,0] = np.min(Rs[i])
            mins[i,1] = np.min(Zs[i])
            maxs[i,0] = np.max(Rs[i])
            maxs[i,1] = np.max(Zs[i])

        #if centroid of one contour is inside mins/maxs of another, its a hole
        #we assume all are edges with no holes to start
        cDicts = [{'type':'edge', 'edgeIdx':None} for x in range(N)]
        for i,ctr in enumerate(centroids):
            for j,c in enumerate(contours):
                rTest = (ctr[0] > mins[j,0]) and (ctr[0] < maxs[j,0])
                zTest = (ctr[1] > mins[j,1]) and (ctr[1] < maxs[j,1])

                if np.all([rTest, zTest]) == False:
                    pass
                else:
                    if i==j:
                        pass
                    else:
                        if maxs[i,0] < maxs[j,0]:
                            #if we found a hole, update pointers and type
                            cDicts[i]['type'] = 'hole'
                            cDicts[i]['edgeIdx'] = j
                        else:
                            pass

        return cDicts


    def shapelyPtables(self, centroids, pTablePath, gridSizes, tableData):
        """
        creates a parallelogram table for input to EFIT/TOKSYS

        solutions are shapely solution objects (list)
        gridSizes are mesh grid sizes for each set of solutions (list)
        pTablePath is path where we save pTables
        """
        count = 0
        pTable = np.zeros((len(centroids), 7))
        for j,ctr in enumerate(centroids):
            #Rc
            pTable[j,0] = ctr[0] *1e-3 #to meters
            #Zc
            pTable[j,1] = ctr[1] *1e-3 #to meters

            if type(gridSizes[j]) != float:
                grid1 = gridSizes[j][0]
                grid2 = gridSizes[j][1]
            else:
                grid1 = gridSizes[j]
                grid2 = gridSizes[j]
            #L
            pTable[j,2] = grid1/2.0 *1e-3 #to meters
            #w
            pTable[j,3] = grid2/2.0 *1e-3 #to meters
            #AC1
            pTable[j,4] = 0.0
            #AC2
            pTable[j,5] = 0.0
            #GroupID
            pTable[j,6] = 0

        #save pTableAll
        pTableOut = pTablePath + 'pTableAll.csv'
        print("Saving Parallelogram Table...")
        head = 'Rc[m], Zc[m], L[m], W[m], AC1[deg], AC2[deg], GroupID'
        try:
            np.savetxt(pTableOut, pTable, delimiter=',',fmt='%.10f', header=head)
        except:
            print("couldn't save pTable")
        return pTableOut

    def createDFsFromCSVs(self, fileList):
        """
        creates a list of dataframes from a list of CSVs
        """
        if type(fileList) != list:
            fileList = [fileList]
        dfs = []
        for f in fileList:
            df = pd.read_csv(f, skiprows=0)
            df.columns = ['Rc[m]', 'Zc[m]', 'L[m]', 'W[m]', 'AC1[deg]', 'AC2[deg]', 'GroupID']
            dfs.append(df)
        return dfs

    def combineElements(self, data):
        """
        combines multiple mesh elements into a single element.
        Only works for square/rectangular elements
        """
        Rc = [x['Rc[m]'] for x in data]
        RminIdxs = [i for i, x in enumerate(Rc) if x == min(Rc)]
        RmaxIdxs = [i for i, x in enumerate(Rc) if x == max(Rc)]

        Zc = np.array([x['Zc[m]'] for x in data])
        ZminIdxs = [i for i, x in enumerate(Zc) if x == min(Zc)]
        ZmaxIdxs = [i for i, x in enumerate(Zc) if x == max(Zc)]

        W = np.array([x['W[m]'] for x in data])
        L = np.array([x['W[m]'] for x in data])
        Wmin = max(W[RminIdxs])
        Lmin = max(L[ZminIdxs])
        Wmax = max(W[RmaxIdxs])
        Lmax = max(L[ZmaxIdxs])

        Rmin = min(Rc)
        Rmax = max(Rc)
        Zmin = min(Zc)
        Zmax = max(Zc)

        Wnew = ((Rmax+Wmax) - (Rmin-Wmin)) / 2.0
        Lnew = ((Zmax+Lmax) - (Zmin-Lmin)) / 2.0
        Rnew = Rmin-Wmin+Wnew
        Znew = Zmin-Lmin+Lnew

        #new grid
        grid = [Wnew, Lnew]

        #build new tableData entry
        tableData={}
        tableData['Rc[m]'] = Rnew
        tableData['Zc[m]'] = Znew
        tableData['W[m]'] = Wnew
        tableData['L[m]'] = Lnew
        tableData['AC1[deg]'] = 0
        tableData['AC2[deg]'] = 0
        tableData['GroupID'] = 0

        #get plotly trace
        xy = np.zeros((5,2))
        xy[0,0] = ( Rnew - Wnew ) *1e3 #m to mm
        xy[0,1] = ( Znew - Lnew ) *1e3 #m to mm
        xy[1,0] = ( Rnew - Wnew ) *1e3 #m to mm
        xy[1,1] = ( Znew + Lnew ) *1e3 #m to mm
        xy[2,0] = ( Rnew + Wnew ) *1e3 #m to mm
        xy[2,1] = ( Znew + Lnew ) *1e3 #m to mm
        xy[3,0] = ( Rnew + Wnew ) *1e3 #m to mm
        xy[3,1] = ( Znew - Lnew ) *1e3 #m to mm
        xy[4,0] = ( Rnew - Wnew ) *1e3 #m to mm
        xy[4,1] = ( Znew - Lnew ) *1e3 #m to mm
        trace = self.XYtrace(xy, opac=0.4)

        return grid, trace, tableData, [Rnew, Znew]


    def addMeshPlots2Fig(self, fig, solutions, opac=0.4):
        """
        adds mesh plots to an existing figure
        """
        for i,sol in enumerate(solutions):
            for j,geom in enumerate(sol.geoms):
                xs, ys = np.array(geom.exterior.xy)
                fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', marker_size=2, fill="toself", opacity=opac, line=dict(color="seagreen"), meta='mesh'))
        return fig

    def XYtrace(self, xy, opac=0.4):
        """
        returns trace of XY points
        """
        trace = go.Scatter(x=xy[:,0], y=xy[:,1], mode='lines+markers', marker_size=2, fill="toself", opacity=opac, line=dict(color="seagreen"), meta='combined')
        return trace

    def getMeshTraces(self, traces = None, opac=0.4):
        """
        returns a list of mesh traces
        """
        if traces == None:
            traces = []
        #generate a random color for this trace
        #c = list(np.random.choice(range(256), size=3))
        #col = 'rgb({:d},{:d},{:d})'.format(c[0],c[1],c[2])
        col = 'seagreen'
        #loop thru all mesh elements and add them to the trace
        for i,sol in enumerate(self.solutions):
            for j,geom in enumerate(sol.geoms):
                xs, ys = np.array(geom.exterior.xy)
                traces.append(go.Scatter(x=xs, y=ys, mode='lines+markers', marker_size=2, fill="toself", opacity=opac, line=dict(color=col), meta='mesh'))
        return traces
