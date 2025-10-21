# Environment
+ python3.9.7
+ (CUDA12.1)
+ pytorch==2.4.1 
> torch-2.4.1+cu121-cp312-cp312-win_amd64.whl
+ torchvision==0.19.1
+ torchaudio==2.4.1

[download.pytorch.org/whl/](https://download.pytorch.org/whl/)


> [cuda12.1版本的pytorch环境安装记录-CSDN](https://blog.csdn.net/qq_45902580/article/details/144246640)

## 环境变量
+ `OPENAI_API_KEY`等密钥(配置在项目的`.env`文件中，运行相关demo前请设置环境变量)



# Something
记录深度学习

## pytorchLearn

[ PyTorch 2.4 documentation](https://docs.pytorch.org/docs/2.4/index.html)

- [x]  [PyTorch深度学习快速入门教程【小土堆】_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1hE411t7RN)



修改 torchvision 下载预训练模型的默认路径

windows环境通常默认是

```
C:\Users\<username>\.cache\torch\hub\checkpoints\
```

把模型缓存目录改掉，在cmd中：

```
setx TORCH_HOME "K:\torch_models"
```



## DeepLearning

[《动手学深度学习》2.0.0 documentation](https://zh.d2l.ai/index.html)


## langchain

[LangChain 集成 - 智谱AI开放文档](https://docs.bigmodel.cn/cn/guide/develop/langchain/introduction#langchain)

[ LangChain --- Tutorials  🦜️🔗 ](https://python.langchain.com/docs/tutorials/)