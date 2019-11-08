import numpy as np

def to_occurrence_matrix(matrix, savepath=None):
    """
    Transform a matrix into a binary matrix where entries are 1 if the original entry was different from 0.
    Parameters
    ----------
    matrix (np.array)
    savepath (string) : path and filename under which to save the file
    Returns
    -------
        The binary matrix or None if a savepath is specified.
    """
    if savepath is None:
        return (matrix > 0)*1
    else:
        np.save(savepath, (matrix>0)*1)

if __name__ == '__main__':

    #a = np.loadtxt(r'adj_coverage_table_5kb_filtered.csv', delimiter=",", skiprows=1, usecols=np.arange(1,19,1))
    #to_occurrence_matrix(a, 'vOTUS_occ')
    #print(a)
    a = np.load(r'D:\Users\Xavier\Documents\Analysis_master\Analysis\clean_analysis\vOTUS_occ.npy')
    print(a.shape)