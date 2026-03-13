import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import scipy

class MCASteps:
    @staticmethod
    def MCAStep1(arr: np.ndarray) -> dict:
        rmin = np.min(arr, axis=0)
        rmax = np.max(arr, axis=0)
        range_list = rmax - rmin

        for index in range(len(rmin)):
            arr[:, index] -= rmin[index]
        for index in range(len(range_list)):
            arr[:, index] /= range_list[index]
            arr[:, index] = np.nan_to_num(arr[:, index])

        complement = 1 - arr
        rows, columns = np.shape(arr)
        FM = np.empty(shape=(rows, columns * 2))
        index = 0
        for column in range(np.shape(FM)[1]):
            if column % 2 == 0:
                FM[:, column] = arr[:, index]
            else:
                FM[:, column] = complement[:, index]
                index += 1
        X = FM.copy()

        FM = np.divide(FM, (np.shape(arr)[0] * np.shape(arr)[1]))

        D_r = np.sum(FM, axis=1)
        D_c = np.sum(FM, axis=0)

        D_r = np.sqrt(np.reciprocal(D_r))
        for n in range(len(D_r)):
            FM[n] *= D_r[n]

        D_c = np.sqrt(np.reciprocal(D_c))
        for i in range(len(D_c)):
            FM[:, i] *= D_c[i]

        return {"Z": FM, "D_r": D_r, "D_c": D_c, "X": X}

    @staticmethod
    def MCAStep2(S: np.ndarray, U: np.ndarray, D_r: np.ndarray, D_c: np.ndarray) -> dict:
        row_coordinates = np.empty(shape=(np.shape(U)))
        for i in range(np.shape(U)[1]):
            row_coordinates[:, i] = D_r * U[:, i]

        S_T = np.transpose(S)
        column_coordinates = np.empty(shape=(np.shape(S_T)))
        for i in range(np.shape(column_coordinates)[1]):
            column_coordinates[:, i] = D_c * S_T[:, i]
        column_coordinates = np.dot(column_coordinates, U)

        rows, columns = np.shape(column_coordinates)
        ret = np.empty(shape=(int(rows / 2), columns))
        index = 0
        for row in range(rows):
            if row % 2 == 0:
                ret[index] = column_coordinates[row]
                index += 1

        return {"cellCoordinates": row_coordinates, "geneCoordinates": ret}

class MCA:
    def __init__(self, cellCoordinates: np.ndarray, geneCoordinates: np.ndarray, X: np.ndarray, genesN: list, cellsN: list, j: int):
        mca_strings = [f"MCA_{i}" for i in range(1, j + 1)]
        self.cellCoordinates = pd.DataFrame(cellCoordinates, index=cellsN, columns=mca_strings)
        self.geneCoordinates = pd.DataFrame(geneCoordinates, index=genesN, columns=mca_strings)
        self.X = X

def RunMCA(arr: pd.DataFrame, j: int = 50, genes: list = None):
    if genes is not None:
        arr = arr.loc[:, genes]

    arr = arr.loc[:, (arr != 0).any(axis=0)]
    arr = arr.loc[:, ~arr.columns.duplicated()]
    cellsN = arr.index
    genesN = arr.columns
    arr = arr.to_numpy()

    MCAPrepRes = MCASteps.MCAStep1(arr)
    U, S, Vh = scipy.sparse.linalg.svds(MCAPrepRes["Z"], k=j, which='LM')
    Vh = np.transpose(Vh)
    S = np.flip(S)

    coordinates = MCASteps.MCAStep2(S=MCAPrepRes["Z"], U=U, D_r=MCAPrepRes["D_r"], D_c=MCAPrepRes["D_c"])
    mca = MCA(cellCoordinates=coordinates["cellCoordinates"], geneCoordinates=coordinates["geneCoordinates"], X=MCAPrepRes["X"], genesN=genesN, cellsN=cellsN, j=j)
    return mca

def GetDistances(cellCoordinates: pd.DataFrame, geneCoordinates: pd.DataFrame, X: np.ndarray = None, barycentric=False, cells_filter: list = None, genes_filter: list = None):
    if genes_filter is not None:
        geneCoordinates = geneCoordinates.loc[genes_filter]
    if cells_filter is not None:
        cellCoordinates = cellCoordinates.loc[cells_filter]

    if not barycentric:
        distances = euclidean_distances(geneCoordinates.to_numpy(), cellCoordinates.to_numpy())
    else:
        g = np.empty(shape=(np.shape(X)[1], np.shape(cellCoordinates)[1]))
        rows, cols = np.shape(g)
        x = np.sum(X, axis=0)
        for k in range(rows):
            for j in range(cols):
                g[k, j] = (1 / x[k]) * np.sum(X[:, k] * cellCoordinates.iloc[:, j])
        g_plus = np.empty(shape=(int(np.shape(X)[1] / 2), np.shape(cellCoordinates)[1]))
        index = 0
        for row in range(0, np.shape(g)[0]):
            if row % 2 == 0:
                g_plus[index] = g[row]
                index += 1
        distances = euclidean_distances(g_plus, cellCoordinates.to_numpy())

    return pd.DataFrame(distances, index=geneCoordinates.index, columns=cellCoordinates.index)
