## 实验记录

### 训练轮数和准确率的关系

2轮训练，loss在1.000附近变化

20轮训练，loss逐渐减小至0.655，测试集结果更好了

![image-20201226202725625](/Users/apple/Library/Application Support/typora-user-images/image-20201226202725625.png)

100轮训练，结果差不多，估计是数据集收敛了

![image-20201226231407395](/Users/apple/Library/Application Support/typora-user-images/image-20201226231407395.png)

减小了学习速率至0.000001，训练了20轮，可能是轮数太少了，所以只收敛了一个分类

![image-20201227112523418](/Users/apple/Library/Application Support/typora-user-images/image-20201227112523418.png)

增加lr至十的五次方，训练10轮，仍然没有收敛

![image-20201227114307085](/Users/apple/Library/Application Support/typora-user-images/image-20201227114307085.png)

增加训练轮数为100轮，lr为十的五次方

![image-20201227130121083](/Users/apple/Library/Application Support/typora-user-images/image-20201227130121083.png)

增加batch_size至16

![image-20201227142434564](/Users/apple/Library/Application Support/typora-user-images/image-20201227142434564.png)

使用resnet，2轮训练

![image-20201228162418717](/Users/apple/Library/Application Support/typora-user-images/image-20201228162418717.png)

