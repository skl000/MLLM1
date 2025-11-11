# 导入所需的库
from modelscope.msdatasets import MsDataset  # 用于加载ModelScope平台的数据集
import os  # 提供与操作系统交互的功能，如目录操作等
import pandas as pd  # 用于数据处理和保存为CSV文件

MAX_DATA_NUMBER = 500  # 设置要下载的最大图片数量为500

# 检查目录是否已存在
if not os.path.exists('coco_2014_caption'):  # 如果'coco_2014_caption'目录不存在
    # 从modelscope下载COCO 2014图像描述数据集
    ds = MsDataset.load('modelscope/coco_2014_caption', subset_name='coco_2014_caption', split='train')  # 加载指定数据集的训练部分
    print(len(ds))  # 打印数据集的大小
    # 设置处理的图片数量上限
    total = min(MAX_DATA_NUMBER, len(ds))  # 确保处理的图片数不超过数据集的实际大小或设定的最大值

    # 创建保存图片的目录
    os.makedirs('coco_2014_caption', exist_ok=True)  # 创建目录，如果目录已存在则不报错

    # 初始化存储图片路径和描述的列表
    image_paths = []  # 存储图片路径的列表
    captions = []  # 存储对应描述的列表

    for i in range(total):  # 遍历前total个数据项
        # 获取每个样本的信息
        item = ds[i]  # 获取第i个数据项
        image_id = item['image_id']  # 图片ID
        caption = item['caption']  # 对应的描述
        image = item['image']  # 图片本身
        
        # 保存图片并记录路径
        image_path = os.path.abspath(f'coco_2014_caption/{image_id}.jpg')  # 构造图片保存路径
        image.save(image_path)  # 保存图片
        
        # 将路径和描述添加到列表中
        image_paths.append(image_path)  # 添加图片路径到列表
        captions.append(caption)  # 添加描述到列表
        
        # 每处理50张图片打印一次进度
        if (i + 1) % 50 == 0:
            print(f'Processing {i+1}/{total} images ({(i+1)/total*100:.1f}%)')  # 打印进度信息

    # 将图片路径和描述保存为CSV文件
    df = pd.DataFrame({
        'image_path': image_paths,  # 图片路径列
        'caption': captions  # 描述列
    })
    
    # 将数据保存为CSV文件
    df.to_csv('./coco-2024-dataset.csv', index=False)  # 保存DataFrame为CSV文件
    
    print(f'数据处理完成，共处理了{total}张图片')  # 完成消息

else:
    print('coco_2014_caption目录已存在,跳过数据处理步骤')  # 目录已存在时的提示信息