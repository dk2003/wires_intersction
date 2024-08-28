git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything

# 以下不写就报错，_C is not defined
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
which nvcc	/usr/local/cuda/bin/nvcc
export CUDA_HOME=/usr/local/cuda

pip cache purge
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
pip install --upgrade diffusers[torch]
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel

wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

方案1：windows：使用代理（需要梯子）
在你的 Python 代码的开头加上如下代码
import os
os.environ['HTTP_PROXY'] = 'http://proxy_ip_address:port'
os.environ['HTTPS_PROXY'] = 'http://proxy_ip_address:port'
其中 http://proxy_ip_address:port 中的 proxy_ip_address 和 port为开启梯子后
（windows）设置>网络和Internet>代理>手动设置代理>编辑代理服务器
中的代理IP地址和端口

方案2：linux
pip install -U huggingface_hub
vim ~/.bashrc
英文状态下按"i"进入insert模式，在文件模型末尾插入 export HF_ENDPOINT=https://hf-mirror.com，按下Esc退出insert模式，再在英文状态下按":wq"即可保存文件。
source ~/.bashrc

执行以下三行，否则还是  _C 未定义
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.bashrc
source ~/.bashrc
pip install https://github.com/IDEA-Research/GroundingDINO/archive/refs/tags/v0.1.0-alpha2.tar.gz

python grounding_dino_demo.py

# 无hq
export CUDA_VISIBLE_DEVICES=0
python grounded_sam_demo.py \
--config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
--grounded_checkpoint groundingdino_swint_ogc.pth \
--sam_checkpoint sam_vit_h_4b8939.pth \
--input_image assets/demo10.jpg \
--output_dir "outputs" \
--box_threshold 0.3 \
--text_threshold 0.25 \
--text_prompt "wire." \
--device "cuda" \

# hq
export CUDA_VISIBLE_DEVICES=0
python grounded_sam_demo.py \
--config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
--grounded_checkpoint groundingdino_swint_ogc.pth \
--sam_hq_checkpoint sam_hq_vit_h.pth \
--use_sam_hq \
--input_image assets/demo10.jpg \
--output_dir "outputs" \
--box_threshold 0.3 \
--text_threshold 0.25 \
--text_prompt "wire." \
--device "cuda" \