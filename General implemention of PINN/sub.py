import sys, os, math, tempfile, atexit, shutil
from joblib import Parallel, delayed

# Subprocess function
def subprocess_task():
    mpldir = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, mpldir)
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(mpldir, 0o777 & ~umask)
    os.environ['HOME'] = mpldir
    os.environ['MPLCONFIGDIR'] = mpldir
    import matplotlib
    class TexManager(matplotlib.texmanager.TexManager):
        texcache = os.path.join(mpldir, 'tex.cache')
    matplotlib.texmanager.TexManager = TexManager
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['text.usetex'] = True

    # From here on, safe to use matplotlib in parallel

# Main process function
def mainprocess_task(n_threads=32):
    with Parallel(n_jobs=n_threads) as parallel:
        parallel(delayed(subprocess_task)() for i in range(0,256))