import numpy as np
import struct

DIPHA_MAGIC = 8067171840
DIPHA_WEIGHTED_BOUNDARY_MATRIX = 0
DIPHA_IMAGE_DATA = 1
DIPHA_PERSISTENCE_DIAGRAM = 2
DIPHA_DISTANCE_MATRIX = 7
DIPHA_SPARSE_DISTANCE_MATRIX = 8

def load(fname, finitization = None):
    ret = []
    
    with open(fname, "rb") as f:
        if struct.unpack('<q', f.read(8))[0] != DIPHA_MAGIC:
            raise IOError("File %s is not a valid DIPHA file." %(fname))
        if struct.unpack('<q', f.read(8))[0] != DIPHA_PERSISTENCE_DIAGRAM:
            raise IOError("File %s is not a valid DIPHA barcode file." %(fname))

        n = struct.unpack('<q', f.read(8))[0]
        
        for i in range(0, n):
            (d, birth, death) = struct.unpack('<qdd', f.read(3*8))
            if d < 0:
                dim = -d-1
            else:
                dim = d

            while dim >= len(ret):
                ret.append([])

            if d < 0:
                if finitization is None:
                    ret[dim].append([birth, np.inf])
                elif finitization >= birth:
                    ret[dim].append([birth, finitization])
            else:
                ret[dim].append([birth, death])

    return [np.array(pd) for pd in ret]


def save(fname, pd):
    with open(fname, "wb") as f:
        f.write(struct.pack("<q", DIPHA_MAGIC))
        f.write(struct.pack("<q", DIPHA_PERSISTENCE_DIAGRAM))

        n = 0
        for x in pd:
            assert(x.shape[1] == 2)
            n += x.shape[0]
        f.write(struct.pack("<q", n))

        for (dim, x) in enumerate(pd):
            for i in range(0, x.shape[0]):
                if np.isfinite(x[i, 1]):
                    f.write(struct.pack("<qdd", dim, x[i, 0], x[i, 1]))
                else:
                    f.write(struct.pack("<qdd", -dim-1, x[i, 0], np.inf))
