"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
from setuptools import setup

setup(
    name='dl_profiler',
    version='0.0.1',
    description='Deep Learning Engine for PyTorch',
    url='https://github.com/yuxiaoguo/DLEngine',
    author='Yu-Xiao Guo',
    author_email='yuxiao.guo@outlook.com',
    license='MIT',
    packages=['dl_engine'],
    install_requires=[
        'torch==1.13.1',
        'PyYAML==6.0',
        'tqdm==4.65.0',
        'transformers==4.27.4',
        'tensorboard==2.12.2',
        'configargparse==1.5.3',
        'opencv-python-headless',
        'librosa==0.10.0.post2',
        'gitpython==3.1.31',
        'lightning==2.0.2',
        'moviepy==1.0.3',
        'torchmetrics==0.11.4',
    ],
    zip_safe=False
)
