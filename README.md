# Blender 4.2 Cycles GPU Prototype — Willebrord Snellius (1080×1920 Short)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssghtc002-sudo/snellius-blender-gpu-prototype/blob/main/render_on_gpu.ipynb)

1-Click **Free GPU + Blender 4.2 LTS Cycles** prototype for **Scientist #6: Willebrord Snellius** (Snell's Law of Light Refraction: `n1 sin θ1 = n2 sin θ2`).

## How to Run (Free GPU — 2 Minutes Total)
1. Click the **Open in Colab** badge above (or import `render_on_gpu.ipynb` into **Kaggle**).
2. Enable Free GPU:
   - **Colab**: `Runtime` -> `Change runtime type` -> **T4 GPU**
   - **Kaggle**: `Settings` -> `Accelerator` -> **GPU T4 x2**
3. Click **Run All**.
4. Final `1080×1920` H.264 MP4 (`output/snellius_gpu_prototype.mp4`) auto-downloads with real Cycles glass/water refraction (`IOR 1.333 / 1.52`), Depth of Field bokeh, stereo audio, and boxless subtitles!

## Adding `@zackdfilms`-Style Realistic Human Characters
Drop any free rigged `.glb` or `.fbx` human character (from **Mixamo**, **Ready Player Me**, or **Blender MPFB2 / MakeHuman**) into `assets/character.glb` — `src/build_snellius_scene.py` automatically imports, positions, and animates it beside the optical workbench!
