pip uninstall torch
pip cache purge

pip uninstall torch
pip cache purge
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
