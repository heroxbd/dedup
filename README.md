# data pipeline

## 读入数据
原始json文件存在data数据集下，直接读入data/*.json，根据要处理的域按顺序将信息提取到一个list中

例：提取所有paper的year
```
import json
with open('data/pubs_train.json', 'r') as f:
    data = json.load(f)
year = [d[u'year'] for name in data.keys() for d in data[name]]
```
提取后的''year''是一个长度为152256的列表

## 将数据转化成feature
假设生成的feature是d维，则要求输出格式为(152256, d)大小的numpy数组，并保存到features/your_feature_name.h5中

hdf5使用示例：假设feature为训练好的(152256, d)的numpy数组
```
import h5py
with h5py.File('features/your_feature_name.h5', 'w') as f:
    f['your_feature_name'] = feature
```