git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -r requirements.txt
pip install -e .
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..