# WhereAmongUs
Where AmongUs is a project to compare reference masks to those generated by SAM
(Meta's segmentation model) and determine areas where a possible overlap occurs.
The Scripts folder contains an OverlayGenerator folder which can be used to generate
overalys on top of input images to place reference masks on top of a specified input
image.

If you can get SAM working you can get this project working. Utilizing the configuration
yaml can also allow you to customize the specific folders where SAM should be referenced
from along with the output and reference images used for mask compariosn.

The ServerTools folder is utilized to create a place to easily send API requests from
a mobile device app in order to take pictures in search of referenced masks.

Refer to the below activated python libaries(I reccomend activating a dev environment) for
further help.

# Activated Python Libraries
certifi            2022.12.7
charset-normalizer 2.1.1
coloredlogs        15.0.1
contourpy          1.0.7
cycler             0.11.0
filelock           3.9.0
flatbuffers        23.3.3
fonttools          4.39.3
humanfriendly      10.0
idna               3.4
Jinja2             3.1.2
kiwisolver         1.4.4
MarkupSafe         2.1.2
matplotlib         3.7.1
mpmath             1.3.0
setuptools         63.2.0
six                1.16.0
sympy              1.11.1
torch              2.0.1+cu117
torchaudio         2.0.2+cu117
torchvision        0.15.2+cu117
typing_extensions  4.5.0
urllib3            1.26.13

