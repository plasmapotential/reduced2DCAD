#reducedCAD.py
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

    def loadMeshParams(self, meshType, gridSize):
        self.meshType = meshType
        self.grid_size = gridSize
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


    def shapelyPtables(self, solutions, pTablePath):
        """
        creates a parallelogram table for input to EFIT/TOKSYS

        solutions are shapely solution objects
        pTablePath is path where we save pTables
        """
        pTableFiles = []
        count = 0
        for i,sol in enumerate(solutions):
            #now create parallelogram table
            pTable = np.zeros((len(sol.geoms), 7))
            for j,geom in enumerate(sol.geoms):
                #Rc
                pTable[j,0] = np.array(geom.centroid)[0] *1e-3 #to meters
                #Zc
                pTable[j,1] = np.array(geom.centroid)[1] *1e-3 #to meters
                #L
                pTable[j,2] = self.grid_size *1e-3 #to meters
                #w
                pTable[j,3] = self.grid_size *1e-3 #to meters
                #AC1
                pTable[j,4] = 0.0
                #AC2
                pTable[j,5] = 0.0
            if i==0:
                pTableAll = pTable
            else:
                pTableAll = np.append(pTableAll, pTable, axis=0)

            #save parallelogram tables
            pTableOut = pTablePath + 'pTable{:03d}.csv'.format(i)
            print("Saving Parallelogram Table...")
            head = 'Rc[m], Zc[m], L[m], W[m], AC1[deg], AC2[deg], GroupID'
            np.savetxt(pTableOut, pTable, delimiter=',',fmt='%.10f', header=head)
            pTableFiles.append(pTableOut)
        #save pTableAll
        pTableOut = pTablePath + 'pTableAll.csv'
        print("Saving Parallelogram Table...")
        head = 'Rc[m], Zc[m], L[m], W[m], AC1[deg], AC2[deg], GroupID'
        try:
            np.savetxt(pTableOut, pTableAll, delimiter=',',fmt='%.10f', header=head)
        except:
            print("couldn't save pTable.  are you sure there is a CAD part at this phi?")
        return pTableFiles, pTableOut

    def createDFsFromCSVs(self, fileList):
        """
        creates a list of dataframes from a list of CSVs
        """
        if type(fileList) != list:
            fileList = [fileList]
        dfs = []
        for f in fileList:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            dfs.append(df)
        return dfs

    def addMeshPlots2Fig(self, fig, solutions, opac=0.4):
        """
        adds mesh plots to an existing figure
        """
        for i,sol in enumerate(solutions):
            for j,geom in enumerate(sol.geoms):
                xs, ys = np.array(geom.exterior.xy)
                fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', marker_size=2, fill="toself", opacity=opac, line=dict(color="seagreen")))
        return fig

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
                traces.append(go.Scatter(x=xs, y=ys, mode='lines+markers', marker_size=2, fill="toself", opacity=opac, line=dict(color=col)))
        return traces
