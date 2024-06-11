"""配置文件"""
import os

PAGE_SIZE = 8192
DATA_SIZE = 16  # 存储两个双精度浮点数 (d, d) 大小 16B
BUFFER_SIZE = 2 * 1024 * 1024  # 2MB

DATA_NUM = int(PAGE_SIZE / DATA_SIZE)  # 一页的数据量
PAGE_NUM = int(BUFFER_SIZE / PAGE_SIZE)  # 缓冲区页数

# ----------------------------------------------------------------------------

root = os.path.abspath(os.path.dirname(__file__))
root_dir = root.replace('\\utils\\core', '')

test_dir = r'C:\Users\mzq\Desktop\test_data'

# save_dir = r'C:\Users\mzq\Desktop\tom'
save_dir = root_dir
load_dir = root_dir
data_dir = fr'{root_dir}\data'

pass
