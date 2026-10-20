# 铜环DPI识别上位机

这是一个 Windows 桌面工具，用于导入图片或 PDF，按输入 DPI 进行标定，自动识别铜环/长圆环状特征，并按阈值高亮低于阈值的目标。

## 功能

- 导入 PNG/JPG/BMP/TIF 或 PDF
- 输入图像 DPI
- 输入阈值 DPI
- 自动检测铜环的外接范围
- 计算每个铜环的有效 DPI
- 低于阈值的铜环用红框标记
- 导出标记图片和 CSV 结果

## 使用

1. 先安装 Python 3.10+。
2. 双击 `build_exe.bat`，或先安装依赖后运行 `python app.py`。
3. 导入图片或 PDF。
4. 输入图像 DPI 和阈值 DPI。
5. 点击“开始识别”。

## 打包

运行 `build_exe.bat` 后，生成的 exe 位于 `dist\铜环DPI识别上位机.exe`。
