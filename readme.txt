yolo_sort for rat observation

Rat Behavior Observation System Based on Transfer Learning
DOI: 10.1109/ACCESS.2019.2916339

yolov3部分 代码借鉴于 https://github.com/experiencor/keras-yolo3，针对非最大抑制，迁移学习有改动
sort部分 代码借鉴于 https://github.com/abewley/sort 针对跟踪丢失问题有所修改，能够继续短期丢失目标的跟踪


模型及测试视频百度云盘：


模型放入 model/ 文件下(config.json中可以修改参数)


测试代码：
python3 predict_sort.py -c config.json -i 8.mp4 

