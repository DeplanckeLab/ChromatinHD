import sys
import pyximport; pyximport.install(reload_support=True, language_level=3, setup_args=dict(include_dirs=[np.get_include()]))
if "peakfreeatac.loaders.extraction.fragments" in sys.modules:
    del sys.modules['peakfreeatac.loaders.extraction.fragments']
import peakfreeatac.loaders.extraction.fragments
if "peakfreeatac.loaders.motifs" in sys.modules:
    del sys.modules['peakfreeatac.loaders.extraction.motifs']
import peakfreeatac.loaders.extraction.motifs