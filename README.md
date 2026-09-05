# TextureUpscaler · 贴图超分工具（含 Web 界面）

基于 Real-ESRGAN 的二次元贴图 4 倍超分工具，附带一个本地 Web 界面，
可以在浏览器里调参、实时预览、控制超分程度，并做基础图像调整。

## 两种用法

### 方式一：拖拽 exe
把 PNG 贴图（可多选）或整个文件夹直接拖到 `TextureUpscaler.exe` 上，
等窗口显示"全部完成"，结果在图片所在目录的 `upscaled` 文件夹里。

### 方式二：Web
双击 `启动Web界面.bat`，浏览器自动打开 `http://127.0.0.1:5000`。

## 渲染贴图子页面（/pbr）

主界面点"渲染贴图"进入，从单张贴图生成一套 PBR 贴图：
- 法线 Normal（强度可调、可反转高度）
- 高度 Height
- 粗糙度 Roughness
- 金属度 Metallic
- 环境光遮蔽 AO
每张可单独预览和下载。

## 环境要求

- Windows 10/11（x64）
- 任意支持 Vulkan 的 GPU（核显也可跑）
- 无需另装 Python：已内置 `python_embed` 便携 Python 环境，并预装好 flask / pillow / numpy

## 目录结构

```
TextureUpscaler/
├─ TextureUpscaler.exe      拖拽超分主程序
├─ 启动Web界面.bat           一键启动 Web 服务（双击即可）
├─ python_embed/            内置 Python 3.12 + 依赖（flask/pillow/numpy）
├─ realesrgan/               超分引擎与模型
└─ web/
   ├─ app.py                后端服务
   ├─ requirements.txt      Python 依赖清单（仅备用）
   ├─ static/index.html     前端界面（支持日/夜间主题，切换有渐变过渡）
   └─ uploads/ previews/ output/ work/   运行时自动生成的临时目录
```

## 分发说明

整个 `TextureUpscaler` 文件夹打包分发即可，真正绿色免安装。
对方解压后双击 `启动Web界面.bat` 就能用 Web 界面，浏览器自动打开；
不需要安装 Python、不需要联网装依赖（环境已内置）。
不启动 Web 也可以直接把图片拖到 `TextureUpscaler.exe` 上超分。
`web/` 下的 uploads / previews / output / work 为运行时临时目录，可清空，不会随包分发。

## 常见问题

- 端口被占用：默认 5000，如冲突请改 `web/app.py` 末尾的 `port`。
- 超分慢：图越大越慢，4 倍比 2 倍慢，进度条会实时显示。
- 想要命令行手动调参：可直接调用
  `realesrgan\realesrgan-ncnn-vulkan.exe -i 输入 -o 输出 -n 模型名 -s 倍数`