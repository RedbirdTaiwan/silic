# SILIC
# 野生動物聲音辨識

## Sound Identification and Labeling Intelligence for Creatures
![SILIC](./model/silic_logo_full.svg)

## 簡介
SILIC(Sound Identification and Labeling Intelligence for Creatures) 是一個由行政院農委會特有生物研究保育中心開發的野生動物聲音辨識 AI 工具
- 本repo fork 自 https://github.com/RedbirdTaiwan/silic ，感謝前人提供這麼好的東西
- 此版本在原版的基礎之上微幅改寫加入輸入整個資料匣，一次辯識功能。
- 支援格式：wav, mp3, wma, m4a, ogg

## 使用方法
- 環境設定 Ta-Chih Chen [SILIC 環境設定 for Window 10 or 11](https://medium.com/@raymond96383/silic-%E7%92%B0%E5%A2%83%E8%A8%AD%E5%AE%9A-for-window10-or-11-f5bb77d4e64f)
- 將要辯識的影音檔放入sample
- 執行silic.ipynb
- 結果會存在result_silic資料匣，並壓縮成result_silic.zip檔，執行資料匣裡的index.html即可看到結果