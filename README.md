# simpleKG
个人开源知识图谱构建框架




## 使用例子



* 新词发现

  ```python
  import codecs
  
  from simple_kg.discovery.entropy_pmi_new_word_discovery import Model
  
  # 加载数据集
  document_file = './斗破苍穹.txt'
  doc = codecs.open(document_file, "r", "utf-8").read()
  
  # 加载新词发现模型
  model = Model(max_word_len=5, min_tf=0.00005, min_entropy=1.0, min_pmi=3.0)
  
  # 挖掘新词
  model.extract_phrase(doc)
  ```
