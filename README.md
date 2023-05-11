# SILIC
# 野生動物聲音辨識

## Sound Identification and Labeling Intelligence for Creatures
![SILIC](./model/silic_logo_full.svg)

## 簡介
SILIC(Sound Identification and Labeling Intelligence for Creatures) 是一個由行政院農委會特有生物研究保育中心開發的野生動物聲音辨識 AI 工具
- 本repo fork 自 [silic](https://github.com/RedbirdTaiwan/silic) 與 [silic-bat](https://github.com/RedbirdTaiwan/silic-bat) ，感謝作者提供這麼好的東西
- 此版本僅在原版的基礎之上微幅改寫。
- 支援格式：wav, mp3, wma, m4a, ogg

## 使用方法
- 環境設定可參考 Ta-Chih Chen 大大的這一篇 [SILIC 環境設定 for Window 10 or 11](https://medium.com/@raymond96383/silic-%E7%92%B0%E5%A2%83%E8%A8%AD%E5%AE%9A-for-window10-or-11-f5bb77d4e64f)
    - 安裝[git](https://git-scm.com/downloads)
    - 安裝[python](https://www.python.org/downloads/)
    - 安裝[FFmpeg](https://www.ffmpeg.org/download.html)
- 執行silic.bat檔


## 更新記錄

### 2023/05/11
- 更新內核為原作者20230503發佈之版本，作者的ui介面太棒了！
- 修正CSV檔直接用excel開啟時亂碼問題
- 整合silic-bat model
- 加入bat檔，讓使用者可以免打Code開始執行

### 2023/03/27
- 資料匣內檔案命名為「8碼日期_6碼時間」，則輸出的CSV檔會加入「錄製日期、開始時間、結束時間」三個欄位
- 可設定檔案路徑

### 2023/03/25
- 輸入整個資料匣進行辯識
