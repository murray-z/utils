# utils
> 存放一些工具函数

# 函数说明
- plot_scatter_fig.py
    - 将高维数据降维，以散点图显示
- measure.py
    - 多分类数据结果评估
- money_format.py
    - 将汉字金额或者阿拉伯金额转化为统一格式
- edit_distance.py
    - 计算字符串编辑距离
    - 其他实现
        - [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy)
        - [python-Levenshtein](https://github.com/miohtama/python-Levenshtein)
- simhash.py
    - 采用‘tf-idf’计算文本关键词权重，采用SimHash计算文本相似度
- lda_topic_words.py
    - 利用lda提取主题关键词
- kenlm_train_lm.py
    - 采用kenlm训练语言模型
    - 参考：[使用kenlm工具训练统计语言模型](https://blog.csdn.net/mingzai624/article/details/79560063)
    
- gunicorn_http_server.py
    - 采用gunicorn提升Http服务处理高并发的能力
    - 使用命令
        - gunicorn -b 10.28.100.164:5001 -k gevent -w 20 guncorn_http_server:app
    - 常用参数
        - -w 设置启动`python app` worker进程的数量
        - -k 运行模式(sync, gevent等等)
        - -b gunicorn 启动绑定的host和port
        - --max-requests 最大处理量, 单woker进程如果处理了超过该数量的请求, 该woker会平滑重启
    - 参考
        - [Flask服务部署](https://www.centos.bz/2017/07/flask-nginx-gunicorn-gevent/) 
  
- trie_tree.py
    - 构建trie树，包含插入、删除、查询、保存 
    
- kashgari.py
    - 深度学习框架，用于文本分类及序列标注
    - [参考文档](https://kashgari-zh.bmio.net/)
    - [git](https://github.com/BrikerMan/Kashgari/) 
  
- sort.py
    - 几种排序方法
    - [参考](https://github.com/amusi/Deep-Learning-Interview-Book/blob/master/docs/%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%E4%B8%8E%E7%AE%97%E6%B3%95.md) 
    