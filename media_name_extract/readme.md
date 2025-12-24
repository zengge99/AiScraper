下载与使用说明
打包下载：将上述 5 个文件保存到同一目录media_name_extract下，可打包为media_name_extract.zip；
上传至 VPS：通过scp/ftp将压缩包上传至 VPS，解压：
bash
运行
unzip media_name_extract.zip
cd media_name_extract
安装依赖：
bash
运行
pip3 install torch==2.0.1 transformers==4.30.2 numpy==1.24.3
训练模型：
bash
运行
python3 main.py train
推理提取名称：
bash
运行
python3 main.py infer --path "你的混乱文件路径"