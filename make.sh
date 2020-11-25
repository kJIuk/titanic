python3 -m venv .venv
source ./.venv/bin/activate
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
pip install --upgrade setuptools
pip install /media/kokhanll/4C65702655934747/Distrib/torch-1.4.0a0+7404463-cp36-cp36m-linux_x86_64.whl
pip install /media/kokhanll/4C65702655934747/Distrib/torchvision-0.5.0a0+85b8fbf-cp36-cp36m-linux_x86_64.whl
#pip install -r requirements.txt
