import numpy as np
import pandas as pd

import mphot

wavelengths = np.arange(0.3, 3, 0.0001)

mirror = pd.read_csv(
    "./systems/optics/Al_mirror.csv", index_col=0, header=None, names=["mirror"]
)
lens = pd.read_csv(
    "./systems/optics/fused_silica_lens.csv", index_col=0, header=None, names=["lens"]
)
qe = pd.read_csv("./systems/qe/PIRTqe_-60.csv", index_col=0, header=None, names=["qe"])
# filter = pd.read_csv("./filters/zYJ.csv", index_col=0, header=None, names=["filter"])

inp = mphot.interpolate_dfs(wavelengths, mirror, lens, qe)

inp["total"] = inp["mirror"] ** 2 * inp["lens"] ** 2 * inp["qe"]

inp.plot()
