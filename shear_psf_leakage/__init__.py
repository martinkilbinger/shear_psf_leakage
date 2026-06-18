from importlib.metadata import version
from pathlib import Path
import matplotlib.style

__version__ = version("shear_psf_leakage")

_style_path = Path(__file__).parent / "shear_psf_leakage.mplstyle"
matplotlib.style.core.USER_LIBRARY_PATHS.append(str(_style_path.parent))
matplotlib.style.core.reload_library()
