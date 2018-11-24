# data pipeline

## 数据存放
在根目录下建立data和features文件夹，分别存放原始数据和提好的feature。

注意数据和feature文件的后缀都应包括在.gitignore内，避免上传到github

## similarity的格式

参考 `c_org.py`.  第一部分定义命令行的输入和输出：
```
#!/usr/bin/env python
import argparse
psr = argparse.ArgumentParser("baseline solution")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', help="input")
args = psr.parse_args()

import pandas as pd, itertools as it, h5py, numpy as np
```

中间一部分由我们自行编写，其中 `pair` 的顺序由 `it.combinations(au.groupby('id'),2)` 决定。

最后一部分定义了`hdf5`的输出。

```
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('c_{}'.format(args.field), data=x, compression="gzip", shuffle=True)
```

# 从头开始执行项目的pipeline

## 安装依赖
具体R和python在windows下的安装方式可以百度
### 安装R
以及以下R的库：argparse，rjson，plyr，doMC

附服务器linux下安装步骤
+ 命令行下执行命令    
    ```
    sudo apt-get update
    sudo apt-get install r-base
    sudo apt-get install r-base-dev
    ``` 
+ 命令行下输入R，显示R的欢迎界面，安装成功
+ 退出R，在命令行下输入unset DISPLAY
+ 打开R，输入``install.packages("plyr")``，选择国内的镜像（比如19是北京），即能进行安装

### 安装python 2.7
（注意不要装成python 3，2和3有很大不同）

以及以下python的库：numpy, pandas, h5py, scikit-learn（版本应为0.20.0）

### 安装xgboost
如果只想生成特征，不需要运行分类器，可以跳过此步

附服务器linux下安装步骤，其他环境查看https://xgboost.readthedocs.io/en/latest/build.html
``` 
git clone --recursive https://github.com/dmlc/xgboost
``` 
进入xgboost文件夹，-j后可以加用来执行make命令的核数，比如-j8就是用8个核
``` 
mkdir build
cd build
cmake .. 
make -j
``` 
安装成功后，
```
cd python-package; python setup.py develop --user
```

### 在命令行中安装make和jq
+ linux下一般有make，安装jq执行命令：sudo apt install jq

## 开始在命令行下执行
### 第一步：按字典序排序train_names
执行``make train_names.mk``，
成功会显示
``` 
echo 'train_names:=bin_yu bin_zhao bing_liu bo_jiang bo_zhou c_c_wang c_h_chen c_yang chen_chen chen_liu cheng_zhu dan_wu di_wu dong_zhang f_liu fei_gao feng_zhu gang_chen gang_zhang h_yu hong_zhao hui_gao hui_yu j_lin jia_li jie_sun jie_zhou jie_zhu jin_he jin_xu jing_huang jing_jin jing_tian jing_yu jue_wang kai_chen kun_li kun_zhang l_zhao lan_wang lei_shi lei_wu lei_zhu li_guo li_he li_huang li_jiang li_ma liang_liu lin_zhou liu_yang m_li m_yang min_chen min_yang ming_chen ming_xu ping_zhang qi_li qian_chen qing_li qing_liu qing_yang qing_zhang rui_zhang s_huang s_liu shuai_zhang t_wang w_huang wei_song xi_li xi_liu xi_zhang xia_zhang xiang_gao xiang_wang xiao_liu xiaoping_wu xin_wu xing_liu xue_wang y_feng y_guo y_luo y_shi yan_gao yan_liang yao_zhang yi_chen yi_jiang yong_yu yuanyuan_liu yuanyuan_zhang z_wu z_zhou zhang_lei zhe_zhang zhi_gang_zhang zhi_wang' > train_names.mk
```

### 第二步：生成特征和标签文件
执行
```
make features/train/c_keywords.h5
make features/train/c_org.h5
make features/train/label.h5
```
会逐个显示类似以下命令
```
mkdir -p features/train/c_keywords/
./c_org.py data/train/keywords/bin_yu.csv -o features/train/c_keywords/bin_yu.h5 --field keywords
```
并最终merge到同一个文件，显示
```
./merge.py 所有名字的h5 -o features/train/c_keywords.h5 --field keywords
```
同样地，此处如果不想占用CPU所有核，可以在``make features/train/c_keywords.h5 -j2``，则只使用2个核，-j8就是8个核

如果不想从头生成这些 hdf5 文件，可以从以下地址下载
```
http://dpcg.d.airelinux.org:8000/edit/dedup/features/train/c_keywords.h5
http://dpcg.d.airelinux.org:8000/edit/dedup/features/train/c_org.h5
http://dpcg.d.airelinux.org:8000/edit/dedup/features/train/label.h5
```
