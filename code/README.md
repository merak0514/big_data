* `record.py` 生產`record.csv`，但應該沒用了
* `users.py` 是試圖遍歷然後被我停掉的
* `picture_classification.py`是分類網絡，具體來説：
    - 20%的測試集
    - resnet50
    - adam
    - image generator加了一些基本的旋轉縮放反轉，沒有做複雜的generator
    - early stop，patience=10
    - 訓練完保存最好的模型到`../model.h5` 臨時之舉
    - 加了tensorboard，雖然感覺沒用