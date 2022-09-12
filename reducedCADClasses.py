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
import multiprocessing
from functools import partial

class CAD3D:

    def __init__(self):
        return

    def loadHEAT(self, HEATpath):
        """
        loads the HEAT environment, loads CADclass object into self
        Requires the HEAT environment to be installed either locally or via
        docker
        """
        sys.path.append(HEATpath)

        #load HEAT environment
        import launchHEAT
        launchHEAT.loadEnviron()

        #load HEAT CAD module and STP file
        import CADClass
        self.CAD = CADClass.CAD(os.environ["rootDir"], os.environ["dataPath"])
        return

    def loadHEATCADenv(self, CADClassPath, FreeCADPath):
        """
        loads the HEAT CADClass environment
        This can be run with only CADClass.py and toolsClass.py from the HEAT
        source code (ie no full HEAT environment)
        """
        sys.path.append(CADClassPath)
        sys.path.append(FreeCADPath)
        #load HEAT CAD module
        import CADClass
        self.CAD = CADClass.CAD(None, None)
#        self.CAD.loadPath(FreeCADPath)

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


    def sectionParams(self, rMax, zMax, phi, discretizeToggle):
        """
        initialize 2D cross section object
        """
        self.rMax = rMax
        self.zMax = zMax
        self.phi = phi
        #if we discretize curves
        if 1 in discretizeToggle:
            self.discretize = True
        else:
            self.discretize = False
        #if we discretize curves, number of figs after radix
        self.radixFigs = 3
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
            edgeList = CAD.getVertexesFromEdges(slice.Shape.Edges, self.discretize, self.radixFigs)
            contours = CAD.findContour(edgeList)
            if len(contours) > 0:
                contourList.append(contours)

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
                #fig.add_trace(go.Scatter(x=R, y=Z, mode='lines+markers'))
                fig.add_trace(go.Scattergl(x=R, y=Z, mode='lines+markers'))
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
                traces.append(go.Scattergl(x=R, y=Z, mode='lines+markers'))
        return traces

    def rzTraces(self):
        """
        returns a list of contour traces
        """
        traces = []
        for slice in self.contourList:
            for c in slice:
                R = np.sqrt(c[:,0]**2+c[:,1]**2)
                Z = c[:,2]
                traces.append(np.vstack([R,Z]).T)
#                traces.append([R,Z])
        return traces



class mesh:

    def __init__(self):
        return

    def loadMeshParams(self, meshType, gridSize, phi, bounds=None):
        self.meshType = meshType
        self.grid_size = gridSize
        self.phi = phi
        self.bounds = bounds
        self.selection = []
        return

    def square(self, x, y, s):
        return Polygon([(x, y), (x+s, y), (x+s, y+s), (x, y+s)])

    def createSquareMesh(self, contourList, grid_size, parallel=True):
        """
        creates a square mesh over the contours in contourList
        using the Shapely library
        """
        self.polygons = []
        self.solutions = []
        self.geoms = []
        polygons = []
        solutions = []
        #run across multiple cores
        if parallel == True:
            for c in contourList:
                #check if this contour has any holes
                cDict = self.findEdgesAndHoles(c)
                N_edges = np.sum([d['type']=='edge' for d in cDict])
                N_holes = np.sum([d['type']=='hole' for d in cDict])

                N = len(cDict)
                self.cDict = cDict
                self.c = c

                #Prepare multiple cores for mesh builder
                Ncores = multiprocessing.cpu_count() - 2 #reserve 2 cores for overhead
                #in case we run on single core machine
                if Ncores <= 0:
                    Ncores = 1

                print('Initializing mesh builder across {:d} cores'.format(Ncores))
                print('Spawning tasks to multiprocessing workers')
                #Do this try clause to kill any zombie threads that don't terminate
                try:
                    manager = multiprocessing.Manager()
                    pool = multiprocessing.Pool(Ncores)
                    output = np.asarray(pool.map(self.parallelSquares, np.arange(N)))
                    polygons = output[:,0][output[:,0]!=None]
                    solutions = output[:,1][output[:,1]!=None]
                finally:
                    pool.close()
                    pool.join()
                    del pool
                    del manager
                print("Multiprocessing complete")
                self.polygons.append(polygons)
                self.solutions.append(solutions)

#for now we dont need to multiprocess this - use loo below
#                N = len(self.solutions)
#                print('Initializing selection checker across {:d} cores'.format(Ncores))
#                print('Spawning tasks to multiprocessing workers')
#                #Do this try clause to kill any zombie threads that don't terminate
#                try:
#                    manager = multiprocessing.Manager()
#                    self.geom = manager.list()
#                    self.selection = manager.list()
#                    self.selectIdx = manager.Value('i',0)
#                    pool = multiprocessing.Pool(Ncores)
#                    geoms = pool.map(self.parallelSelectionCheck, np.arange(N))
#                finally:
#                    pool.close()
#                    pool.join()
#                    del pool
#                    del manager
#                print("Multiprocessing complete")
#                print(geoms)
#                self.geoms = list(np.array(geoms).flatten())
#                self.selection = self.selection[:]
                geoms = []
                selection = []
                idx = 0
                for s in solutions:
                    for g in s.geoms:
                        self.geoms.append(g)
                        #if there is an active selection, create a list of idxs in the selection
                        #for us to reference later
                        if self.bounds != None:
                            xs, ys = np.array(g.exterior.xy)
                            test1 = np.max(xs) < self.bounds['x'][0]
                            test2 = np.min(xs) > self.bounds['x'][1]
                            test3 = np.max(ys) < self.bounds['y'][0]
                            test4 = np.min(ys) > self.bounds['y'][1]
                            if (test1 or test2 or test3 or test4) == False:
                                selection.append(idx)
                        idx+=1

                #append geoms and selection map to self
                #self.geoms = geoms
                if len(selection) > 0:
                    self.selection = self.selection + selection

            #convert these arrays to 1d lists for use later in mainMesh
            self.polygons = list(np.array(self.polygons).flatten())
            self.solutions = list(np.array(self.solutions).flatten())

        #run across single core
        else:
            for c in contourList:
                #check if this contour has any holes
                cDict = self.findEdgesAndHoles(c)
                N_edges = np.sum([d['type']=='edge' for d in cDict])
                N_holes = np.sum([d['type']=='hole' for d in cDict])

                outerContour = []
                holeContours = []
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

                        #use polygon to find intersections (solutions)
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
                selection = []
                idx = 0
                for s in self.solutions:
                    for g in s.geoms:
                        geoms.append(g)
                        #if there is an active selection, create a list of idxs in the selection
                        #for us to reference later
                        if self.bounds != None:
                            xs, ys = np.array(g.exterior.xy)
                            test1 = np.max(xs) < self.bounds['x'][0]
                            test2 = np.min(xs) > self.bounds['x'][1]
                            test3 = np.max(ys) < self.bounds['y'][0]
                            test4 = np.min(ys) > self.bounds['y'][1]
                            if (test1 or test2 or test3 or test4) == False:
                                selection.append(idx)
                        idx+=1

                #append geoms and selection map to self
                self.geoms = geoms
                if len(selection) > 0:
                    self.selection = self.selection + selection

        return


    def parallelSelectionCheck(self,i):
        geoms = []
        for g in self.solutions[i].geoms:
            geoms.append(g)
            #if there is an active selection, create a list of idxs in the selection
            #for us to reference later
            if self.bounds != None:
                xs, ys = np.array(g.exterior.xy)
                test1 = np.max(xs) < self.bounds['x'][0]
                test2 = np.min(xs) > self.bounds['x'][1]
                test3 = np.max(ys) < self.bounds['y'][0]
                test4 = np.min(ys) > self.bounds['y'][1]
                if (test1 or test2 or test3 or test4) == False:
                    self.selection.append(self.selectIdx.value)
            self.selectIdx.value += 1
        return geoms

    def parallelSquares(self, i):
        #loop thru all contours in this c and build meshes accordingly
        #uses hole and edge data from cDict
        d = self.cDict[i]
        outerContour = []
        holeContours = []
        if d['type'] != 'hole':
            outerContour = self.c[i]
            R_out = np.sqrt(outerContour[:,0]**2+outerContour[:,1]**2)
            Z_out = outerContour[:,2]

            #find holes for this outer contour
            holeIdxs = np.where([h['edgeIdx']==i for h in self.cDict])[0]
            holes = []
            for h in holeIdxs:
                R_hole = np.sqrt(self.c[h][:,0]**2+self.c[h][:,1]**2)
                Z_hole = self.c[h][:,2]
                #create a ring of the hole
                holeRing = LinearRing(np.vstack([R_hole,Z_hole]).T)
                holes.append(holeRing)

            #use polygon to find intersections (solutions)
            poly = Polygon(np.vstack([R_out,Z_out]).T, holes)
            polyCoords = np.array(poly.exterior.coords)
            ibounds = np.array(poly.bounds)//self.grid_size
            ibounds[2:4] += 1
            xmin, ymin, xmax, ymax = ibounds*self.grid_size
            xrg = np.arange(xmin, xmax, self.grid_size)
            yrg = np.arange(ymin, ymax, self.grid_size)
            mp = MultiPolygon([self.square(x, y, self.grid_size) for x in xrg for y in yrg])
            solution = MultiPolygon(list(filter(poly.intersects, mp)))
        else:
            poly=None
            solution=None
        return poly, solution

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

    def parallelEdgeHole(self, i):
        for j,c in enumerate(self.contours):
            rTest = (self.centroids[i][0] > self.mins[j,0]) and (self.centroids[i][0] < self.maxs[j,0])
            zTest = (self.centroids[i][1] > self.mins[j,1]) and (self.centroids[i][1] < self.maxs[j,1])
            if np.all([rTest, zTest]) == False:
                pass
            else:
                if i==j:
                    pass
                else:
                    if self.maxs[i,0] < self.maxs[j,0]:
                        #if we found a hole, update pointers and type
                        #cDicts[i]['type'] = 'hole'
                        #cDicts[i]['edgeIdx'] = j
                        edgeIdx = j #hole
                    else:
                        edgeIdx = None #edge
        return edgeIdx

    def shapelyPtables(self, centroids, pTablePath, gridSizes, tableData):
        """
        creates a parallelogram table for input to EFIT/TOKSYS

        solutions are shapely solution objects (list)
        gridSizes are mesh grid sizes for each set of solutions (list ie [R,Z])
        pTablePath is path where we save pTables
        """
        count = 0
        pTable = np.zeros((len(centroids), 11))
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
            pTable[j,2] = grid1 *1e-3 #to meters
            #w
            pTable[j,3] = grid2 *1e-3 #to meters
            #AC1
            pTable[j,4] = 0.0
            #AC2
            pTable[j,5] = 0.0
            #NL
            pTable[j,6] = 1
            #NW
            pTable[j,7] = 1
            #material
            pTable[j,8] = 0
            #caf
            pTable[j,9] = 1
            #isf
            pTable[j,10] = 1

        #save pTableAll
        pTableOut = pTablePath + 'pTableAll.csv'
        print("Saving Parallelogram Table...")
        head = 'Rc[m], Zc[m], L[m], W[m], AC1[deg], AC2[deg], NL, NW, material, caf, isf'
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
            df = df.round(8)
            df.columns = ['Rc[m]', 'Zc[m]', 'L[m]', 'W[m]', 'AC1[deg]', 'AC2[deg]', 'NL', 'NW', 'material', 'caf', 'isf']
            dfs.append(df)
        return dfs

    def getRectangleBounds(self,shapeData):
        #rectangles
        if 'shapes[0].x0' in shapeData.keys():
            x0 = shapeData['shapes[0].x0']
            x1 = shapeData['shapes[0].x1']
            y0 = shapeData['shapes[0].y0']
            y1 = shapeData['shapes[0].y1']
        else:
            shp = shapeData['shapes'][-1]
            if shp['type'] == 'rect':
                x0 = shp['x0']
                x1 = shp['x1']
                y0 = shp['y0']
                y1 = shp['y1']
        return x0,x1,y0,y1


    def combineElements(self, data):
        """
        combines multiple mesh elements into a single element.
        Only works for square/rectangular elements
        """
        #meter to mm unit conversion
        m2mm = 1000.0

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

        Wnew = ((Rmax+Wmax/2.0) - (Rmin-Wmin/2.0))
        Lnew = ((Zmax+Lmax/2.0) - (Zmin-Lmin/2.0))
        Rnew = Rmin - Wmin/2.0 + Wnew/2.0
        Znew = Zmin - Lmin/2.0 + Lnew/2.0

        #for testing
        #print(Wnew)
        #print(Lnew)
        #print('--')
        #print(Rnew)
        #print(Znew)
        #print('--')
        #print(Rmin)
        #print(Zmin)
        #print('--')
        #print(Wmin)
        #print(Lmin)

        #new grid
        grid = [Wnew*m2mm, Lnew*m2mm]

        #build new tableData entry
        tableData={}
        tableData['Rc[m]'] = round(Rnew, 8)
        tableData['Zc[m]'] = round(Znew, 8)
        tableData['W[m]'] = round(Wnew, 8)
        tableData['L[m]'] = round(Lnew, 8)
        tableData['AC1[deg]'] = 0
        tableData['AC2[deg]'] = 0
        tableData['NL'] = 1
        tableData['NW'] = 1
        tableData['material'] = 0
        tableData['caf'] = 0
        tableData['isf'] = 0

        #get plotly trace
        xy = np.zeros((5,2))
        xy[0,0] = np.round(( Rnew - Wnew/2.0 ) *1e3, 8) #m to mm
        xy[0,1] = np.round(( Znew - Lnew/2.0 ) *1e3, 8) #m to mm
        xy[1,0] = np.round(( Rnew - Wnew/2.0 ) *1e3, 8) #m to mm
        xy[1,1] = np.round(( Znew + Lnew/2.0 ) *1e3, 8) #m to mm
        xy[2,0] = np.round(( Rnew + Wnew/2.0 ) *1e3, 8) #m to mm
        xy[2,1] = np.round(( Znew + Lnew/2.0 ) *1e3, 8) #m to mm
        xy[3,0] = np.round(( Rnew + Wnew/2.0 ) *1e3, 8) #m to mm
        xy[3,1] = np.round(( Znew - Lnew/2.0 ) *1e3, 8) #m to mm
        xy[4,0] = np.round(( Rnew - Wnew/2.0 ) *1e3, 8) #m to mm
        xy[4,1] = np.round(( Znew - Lnew/2.0 ) *1e3, 8) #m to mm

        return grid, xy, tableData, [Rnew*m2mm, Znew*m2mm]


    def addMeshPlots2Fig(self, fig, solutions, opac=0.4):
        """
        adds mesh plots to an existing figure
        """
        for i,sol in enumerate(solutions):
            for j,geom in enumerate(sol.geoms):
                xs, ys = np.array(geom.exterior.xy)
                fig.add_trace(go.Scattergl(x=xs, y=ys, mode='lines+markers', marker_size=2, fill="toself", opacity=opac, line=dict(color="seagreen"), meta='mesh'))
        return fig

    def getMeshTraces(self, parallel=False):
        """
        returns a list of mesh traces
        """
        #Prepare multiple cores for mesh builder
        Ncores = multiprocessing.cpu_count() - 2 #reserve 2 cores for overhead
        #in case we run on single core machine
        if Ncores <= 0:
            Ncores = 1

        #loop thru all mesh elements and add them to the trace
        self.selection = []
        N = len(self.geoms)
        print("Parallel Multiprocessing Run Commencing...")
        #Do this try clause to kill any zombie threads that don't terminate
        try:
            manager = multiprocessing.Manager()
            self.traces = manager.list([None]*N)
            pool = multiprocessing.Pool(Ncores)
            pool.map(self.parallelMeshTraceAdd, np.arange(N))
        finally:
            pool.close()
            pool.join()
            del pool
            del manager

        return np.array(self.traces)

    def parallelMeshTraceAdd(self, i):
        #generate a random color for this trace
        xs, ys = np.array(self.geoms[i].exterior.xy)
        self.traces[i] = np.vstack([xs,ys]).T
        return

    def tracesFromPtable(self, df, opac=0.4):
        """
        generates a list of traces from a dataframe of a pTable csv file
        """
        traces = []
        R = df['Rc'].values
        Z = df['Zc'].values
        L = df['L'].values
        W = df['W'].values
        col = 'seagreen'

        #build a dummy object for reference later in mainMesh functions
        #this dummy object looks like a shapely solution object but it is not
        self.solutions = [solutionClass(len(R))]
        self.solutions[0].geoms = []
        self.geoms = []
        self.grid_size = []
        self.W = []
        self.L = []

        for i in range(len(R)):
            xs = np.array([
                            R[i] - W[i]/2.0,
                            R[i] - W[i]/2.0,
                            R[i] + W[i]/2.0,
                            R[i] + W[i]/2.0,
                            R[i] - W[i]/2.0,
                            ])
            ys = np.array([
                            Z[i] - L[i]/2.0,
                            Z[i] + L[i]/2.0,
                            Z[i] + L[i]/2.0,
                            Z[i] - L[i]/2.0,
                            Z[i] - L[i]/2.0,
                            ])
            #create dummy solutions and geoms accounting for units m=>mm
            self.solutions[0].geoms.append(geomClass(R[i]*1000.0,Z[i]*1000.0))

            #append trace with data scaled from m to mm
            trc = np.round(np.vstack([xs*1000.0,ys*1000.0]).T, 8)
            traces.append(trc)

        minGrid = np.min(np.array([W,L]))
        self.grid_size = minGrid*1000.0
        self.geoms = self.solutions[0].geoms
        return traces

#dummy classes for reading pTables
class solutionClass:
    def __init__(self, Ngeoms):
        self.geoms = []
        return
class geomClass:
    def __init__(self, R, Z):
        self.centroid = [R,Z]
        return
