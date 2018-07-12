# chartlet_v2
## 简述
通过将目标检测、语义分割中目标需要检测或分割的前景贴图到随机背景而产生训练数据的方法

## 用法
### 1. 目录结构

    ├─背景图片          background
    ├─单个目标的标注图   label
    └─标注图对应的原图   source
**标注图中文件名应与原图中一一对应且对应图片名一致**
**图片命名要以物品分类+'-'+其他组成**，如有两张洗衣液的图片一张牙膏的图片，则分别命名为`laundry-1.png`、`laundry-2.png`、`toothpaste.png`以此类推。**即物品类别要在label.txt（见第三点）中已知，且命名是将类别放在最开始，并以.-隔开**

### 2. 命令行参数
|选项|作用|实例|
| - | :-: | :-: | 
|--src|原图目录|'.\src'|
|--mask|标注后原图目录|'.\label'|
|--bg|背景目录|'.\bg'|
|--label_map|标注分类列表|".\label.txt"|
|--output|输出生成的可用于训练的标注图片目录|'.\output'|
|--num|生成图片数量|40|

### 3. label.txt格式如下：

    desktop
    laundry
    noodles
    sprite
    duck
    pencil_case
    bowl
    cookies
    toothpaste
    towel
    chewing gum

最终标注出来的图片在'.\output\label'目录下，程序生成的贴图在'.\output\src'目录下
在标注图中，**所有为desktop的像素值全部为1，为laundry的像素值为2**，以此类推

### 3. 例子

    python chartlet_v2.py --src ".\src" 
    --mask ".\label" 
    --bg ".\bg" 
    --label_map ".\label.txt"
    --num 40 

生成的标注图：

![gen][gen]

可视化label图片：

![label][label]

[gen]: https://raw.githubusercontent.com/trobr/MarkDownPic/master/ML-DL/chartlet/生成图片.png
[label]: https://raw.githubusercontent.com/trobr/MarkDownPic/master/ML-DL/chartlet/label可视化.png
