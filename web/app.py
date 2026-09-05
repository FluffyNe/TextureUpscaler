# -*- coding: utf-8 -*-
"""TextureUpscaler Web 前端后端服务。

依赖: flask, pillow（均已安装）
用法: python app.py  ->  浏览器打开 http://127.0.0.1:5000
"""
import os
import io
import uuid
import threading
import subprocess
from flask import Flask, request, jsonify, send_from_directory, send_file, abort
from PIL import Image, ImageEnhance
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
PREVIEW_DIR = os.path.join(BASE_DIR, 'previews')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
WORK_DIR = os.path.join(BASE_DIR, 'work')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
EXPORT_DIR = os.path.join(BASE_DIR, 'export')

for d in (UPLOAD_DIR, PREVIEW_DIR, OUTPUT_DIR, WORK_DIR, MODEL_DIR, EXPORT_DIR):
    os.makedirs(d, exist_ok=True)

# realesrgan 工具位置：web 目录的上一级是 TextureUpscaler，工具在它的 realesrgan 子目录
TOOL_DIR = os.path.dirname(BASE_DIR)
REALESRGAN_DIR = os.path.join(TOOL_DIR, 'realesrgan')
REALESRGAN_EXE = os.path.join(REALESRGAN_DIR, 'realesrgan-ncnn-vulkan.exe')

MODELS = {
    'animevideov3': {'name': 'realesr-animevideov3', 'label': 'AnimeVideo V3（动漫·2/3/4倍）', 'scales': [2, 3, 4]},
    'x4plus': {'name': 'realesrgan-x4plus', 'label': 'Real-ESRGAN x4plus（通用照片·4倍）', 'scales': [4]},
    'x4plus-anime': {'name': 'realesrgan-x4plus-anime', 'label': 'Real-ESRGAN x4plus-anime（动漫·4倍）', 'scales': [4]},
}

app = Flask(__name__, static_folder=STATIC_DIR)
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024

_tasks = {}
_tasks_lock = threading.Lock()


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def apply_adjustments(img, r, g, b, contrast, brightness, sharpen, saturation=1.0):
    """对 PIL 图片做基础调整，返回新图片。参数均为系数（1.0 = 原样）。"""
    mode = img.mode
    has_alpha = mode in ('RGBA', 'LA')
    if mode == 'P':
        img = img.convert('RGBA')
        has_alpha = True

    if has_alpha:
        alpha = img.convert('RGBA').getchannel('A')
        rgb = img.convert('RGB')
    else:
        alpha = None
        rgb = img.convert('RGB')

    # RGB 通道增益
    if (r, g, b) != (1.0, 1.0, 1.0):
        R, G, B = rgb.split()
        R = R.point(lambda x: _clamp(int(x * r + 0.5), 0, 255))
        G = G.point(lambda x: _clamp(int(x * g + 0.5), 0, 255))
        B = B.point(lambda x: _clamp(int(x * b + 0.5), 0, 255))
        rgb = Image.merge('RGB', (R, G, B))

    if contrast != 1.0:
        rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    if brightness != 1.0:
        rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    if saturation != 1.0:
        rgb = ImageEnhance.Color(rgb).enhance(saturation)
    if sharpen != 1.0:
        rgb = ImageEnhance.Sharpness(rgb).enhance(sharpen)

    if has_alpha:
        rgb = rgb.convert('RGBA')
        rgb.putalpha(alpha)
    return rgb


def _run_realesrgan(src_png, dst_png, model_name, scale, progress_cb=None):
    if not os.path.exists(REALESRGAN_EXE):
        raise RuntimeError('找不到 realesrgan-ncnn-vulkan.exe')
    cmd = [
        REALESRGAN_EXE,
        '-i', src_png,
        '-o', dst_png,
        '-n', model_name,
        '-s', str(scale),
        '-f', 'png',
        '-v',
    ]
    proc = subprocess.Popen(cmd, cwd=REALESRGAN_DIR,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors='replace', bufsize=1)
    tail = []
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        tail.append(line)
        if len(tail) > 20:
            tail.pop(0)
        if line.endswith('%'):
            try:
                pct = float(line[:-1].strip())
                if progress_cb:
                    progress_cb(pct)
            except ValueError:
                pass
    proc.wait(timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError('超分失败：' + ' | '.join(tail[-6:]))


def upscale_image(src_path, dst_path, model_name, scale, progress_cb=None):
    """超分一张图，保留透明通道（alpha 用 LANCZOS 放大后合回）。"""
    img = Image.open(src_path)
    if img.mode in ('RGBA', 'LA'):
        rgba = img.convert('RGBA')
        alpha = rgba.getchannel('A')
        rgb = rgba.convert('RGB')
    elif img.mode == 'P' and 'transparency' in img.info:
        rgba = img.convert('RGBA')
        alpha = rgba.getchannel('A')
        rgb = rgba.convert('RGB')
    else:
        rgb = img.convert('RGB')
        alpha = None

    tmp_rgb = os.path.join(WORK_DIR, uuid.uuid4().hex + '_rgb.png')
    tmp_out = os.path.join(WORK_DIR, uuid.uuid4().hex + '_out.png')
    try:
        rgb.save(tmp_rgb, 'PNG')
        _run_realesrgan(tmp_rgb, tmp_out, model_name, scale, progress_cb)

        if alpha is not None:
            out = Image.open(tmp_out).convert('RGBA')
            new_size = (alpha.width * scale, alpha.height * scale)
            a_big = alpha.resize(new_size, Image.LANCZOS)
            out.putalpha(a_big)
            out.save(dst_path, 'PNG')
        else:
            Image.open(tmp_out).convert('RGB').save(dst_path, 'PNG')
    finally:
        for p in (tmp_rgb, tmp_out):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def _parse_params(data):
    def f(key, default=1.0):
        try:
            return float(data.get(key, default))
        except (TypeError, ValueError):
            return default
    return {
        'r': _clamp(f('r'), 0.0, 2.0),
        'g': _clamp(f('g'), 0.0, 2.0),
        'b': _clamp(f('b'), 0.0, 2.0),
        'contrast': _clamp(f('contrast'), 0.0, 2.0),
        'brightness': _clamp(f('brightness'), 0.0, 2.0),
        'sharpen': _clamp(f('sharpen'), 0.0, 3.0),
        'saturation': _clamp(f('saturation'), 0.0, 2.0),
    }


@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/api/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': '没有文件'}), 400
    img_id = uuid.uuid4().hex
    ext = os.path.splitext(file.filename or '')[-1].lower() or '.png'
    save_path = os.path.join(UPLOAD_DIR, img_id + ext)
    file.save(save_path)
    try:
        im = Image.open(save_path)
        w, h = im.size
        im.load()
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        return jsonify({'error': '无法识别的图片：' + str(e)}), 400
    return jsonify({'id': img_id, 'width': w, 'height': h, 'ext': ext})


@app.route('/api/preview', methods=['POST'])
def preview():
    data = request.get_json(silent=True) or {}
    img_id = data.get('img_id')
    p = _parse_params(data)
    src = _find_upload(img_id)
    if not src:
        return jsonify({'error': '图片不存在'}), 404
    img = Image.open(src)
    out = apply_adjustments(img, p['r'], p['g'], p['b'],
                            p['contrast'], p['brightness'], p['sharpen'], p['saturation'])
    buf = io.BytesIO()
    out.save(buf, 'PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/api/process', methods=['POST'])
def process():
    data = request.get_json(silent=True) or {}
    img_id = data.get('img_id')
    p = _parse_params(data)
    upscale = bool(data.get('upscale'))
    model_key = data.get('model', 'animevideov3')
    if model_key not in MODELS:
        model_key = 'animevideov3'
    try:
        scale = int(data.get('scale', 2))
    except (TypeError, ValueError):
        scale = 2
    if scale not in MODELS[model_key]['scales']:
        scale = MODELS[model_key]['scales'][0]

    src = _find_upload(img_id)
    if not src:
        return jsonify({'error': '图片不存在'}), 404

    out_name = (data.get('out_name') or '').strip()
    replace_original = bool(data.get('replace_original'))

    if not upscale:
        # 仅调整，同步返回
        img = Image.open(src)
        out = apply_adjustments(img, p['r'], p['g'], p['b'],
                                p['contrast'], p['brightness'], p['sharpen'], p['saturation'])
        saved = _save_result(out, img_id, out_name, replace_original,
                             img_id + '_adjusted.png')
        return jsonify({'status': 'done', 'url': saved['url'],
                        'replaced': saved['replaced'],
                        'width': out.width, 'height': out.height})

    # 超分：后台任务
    task_id = uuid.uuid4().hex
    with _tasks_lock:
        _tasks[task_id] = {'status': 'pending', 'url': None, 'error': None,
                           'progress': 0, 'stage': '排队中'}
    t = threading.Thread(target=_run_upscale_task,
                         args=(task_id, img_id, src, p, model_key, scale,
                               out_name, replace_original), daemon=True)
    t.start()
    return jsonify({'status': 'running', 'task_id': task_id})


def _run_upscale_task(task_id, img_id, src, p, model_key, scale,
                      out_name='', replace_original=False):
    def set_task(**kw):
        with _tasks_lock:
            _tasks[task_id].update(kw)
    try:
        set_task(status='running', stage='应用调整', progress=0)
        img = Image.open(src)
        adjusted = apply_adjustments(img, p['r'], p['g'], p['b'],
                                     p['contrast'], p['brightness'], p['sharpen'], p['saturation'])
        tmp_adjusted = os.path.join(WORK_DIR, task_id + '_adj.png')
        adjusted.save(tmp_adjusted, 'PNG')

        set_task(stage='超分中', progress=0)
        tmp_out = os.path.join(WORK_DIR, task_id + '_out.png')
        upscale_image(tmp_adjusted, tmp_out, MODELS[model_key]['name'], scale,
                      progress_cb=lambda pct: set_task(progress=pct, stage='超分中'))

        result = Image.open(tmp_out)
        saved = _save_result(result, img_id, out_name, replace_original,
                             f'{img_id}_s{scale}_{model_key}.png')
        for tp in (tmp_adjusted, tmp_out):
            if os.path.exists(tp):
                try:
                    os.remove(tp)
                except OSError:
                    pass
        set_task(status='done', progress=100, stage='完成',
                 url=saved['url'], replaced=saved['replaced'], error=None)
    except Exception as e:
        set_task(status='error', error=str(e))


@app.route('/api/task/<task_id>')
def task(task_id):
    with _tasks_lock:
        t = _tasks.get(task_id)
    if not t:
        return jsonify({'status': 'error', 'error': '任务不存在'}), 404
    return jsonify(t)


@app.route('/files/<path:name>')
def files(name):
    return send_from_directory(OUTPUT_DIR, name)


def _find_upload(img_id):
    if not img_id:
        return None
    for f in os.listdir(UPLOAD_DIR):
        if f.startswith(img_id):
            return os.path.join(UPLOAD_DIR, f)
    return None


def _safe_name(s):
    for ch in '\\/:*?"<>|':
        s = s.replace(ch, '')
    return s.strip()


def _save_result(result_img, img_id, out_name, replace_original, default_name):
    if replace_original:
        old = _find_upload(img_id)
        newpath = os.path.join(UPLOAD_DIR, img_id + '.png')
        result_img.save(newpath, 'PNG')
        if old and os.path.abspath(old) != os.path.abspath(newpath) and os.path.exists(old):
            os.remove(old)
        return {'url': '/api/original/' + img_id, 'replaced': True}
    name = default_name
    if out_name:
        safe = _safe_name(out_name)
        if safe:
            name = safe
            if not name.lower().endswith('.png'):
                name += '.png'
    dst = os.path.join(OUTPUT_DIR, name)
    result_img.save(dst, 'PNG')
    return {'url': '/files/' + name, 'replaced': False}


@app.route('/api/original/<img_id>')
def original(img_id):
    src = _find_upload(img_id)
    if not src:
        abort(404)
    return send_file(src)


# ---------------- 模型（PMX/PMD）导入与预览资源 ----------------
def _norm_rel(p):
    """把客户端提交的相对路径规范化，防止目录穿越；非法返回 None。"""
    p = (p or '').replace('\\', '/').strip().strip('/')
    if not p or p.startswith('/') or ':' in p.split('/')[0]:
        return None
    parts = []
    for seg in p.split('/'):
        if seg in ('', '.'):
            continue
        if seg == '..':
            return None
        parts.append(seg)
    return '/'.join(parts)


@app.route('/api/model-upload', methods=['POST'])
def model_upload():
    """接收模型 + 贴图的多文件上传（含相对路径），存到独立会话目录。"""
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': '没有收到文件'}), 400
    paths = request.form.getlist('paths')
    if len(paths) != len(files):
        paths = [f.filename or '' for f in files]
    # 已有会话则追加到原目录（用于补传贴图后重载模型）
    sid = (request.form.get('sid') or '').strip() or uuid.uuid4().hex
    root = os.path.join(MODEL_DIR, sid)
    os.makedirs(root, exist_ok=True)
    saved, seen = [], set()
    for f, p in zip(files, paths):
        rel = _norm_rel(p)
        if not rel or rel.lower() in seen:
            continue
        seen.add(rel.lower())
        dest = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        f.save(dest)
        saved.append({'path': rel, 'size': os.path.getsize(dest)})
    if not saved:
        return jsonify({'error': '无法保存文件'}), 400
    return jsonify({'sid': sid, 'files': saved})


@app.route('/models/<sid>/<path:name>')
def model_files(sid, name):
    """提供某会话目录下的文件；精确路径找不到时按 basename 全目录回退（便于扁平上传）。"""
    root = os.path.join(MODEL_DIR, sid)
    if not os.path.isdir(root):
        abort(404)
    rel = _norm_rel(name)
    if not rel:
        abort(400)
    p = os.path.join(root, rel)
    if not os.path.isfile(p):
        base = os.path.basename(rel).lower()
        hit = None
        for dirpath, _dirs, fnames in os.walk(root):
            for fn in fnames:
                if fn.lower() == base:
                    hit = os.path.join(dirpath, fn)
                    break
            if hit:
                break
        if not hit:
            abort(404)
        p = hit
    return send_file(p)

@app.route('/api/model-files/<sid>')
def model_files_list(sid):
    """列出某模型会话目录里的全部文件（相对路径）。"""
    root = os.path.join(MODEL_DIR, sid)
    if not os.path.isdir(root):
        abort(404)
    out = []
    for dirpath, _dirs, fnames in os.walk(root):
        for fn in fnames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace('\\', '/')
            out.append({'rel': rel, 'size': os.path.getsize(full)})
    out.sort(key=lambda x: x['rel'])
    return jsonify({'sid': sid, 'files': out})


def _resolve_result_path(url):
    """把前端记录的挂载结果 URL 映射回本地文件。"""
    if not url:
        return None
    if url.startswith('/files/'):
        name = url[len('/files/'):].split('?', 1)[0]
        p = os.path.join(OUTPUT_DIR, name)
        return p if os.path.isfile(p) else None
    if url.startswith('/api/original/'):
        uid = url[len('/api/original/'):].split('?', 1)[0]
        p = _find_upload(uid)
        return p if p and os.path.isfile(p) else None
    return None


@app.route('/api/export-model', methods=['POST'])
def export_model():
    """把模型 + 贴图按模型相对路径打成 zip，已挂载的贴图用超分结果替换。

    zip 内部布局：模型放根目录，贴图放在模型引用它的相对位置，
    覆盖的贴图会按原扩展名尽量转回原格式（如 bmp），避免 MMD 认不出。
    """
    import zipfile
    data = request.get_json(silent=True) or {}
    sid = (data.get('sid') or '').strip()
    model_rel = (data.get('model_rel') or '').strip()
    root = os.path.join(MODEL_DIR, sid)
    if not os.path.isdir(root):
        return jsonify({'error': '模型会话不存在'}), 404
    if not model_rel:
        return jsonify({'error': '缺少模型文件路径'}), 400

    overrides = {}
    for ov in data.get('overrides') or []:
        rel = _norm_rel(ov.get('rel'))
        if not rel:
            continue
        src = _resolve_result_path(ov.get('url'))
        if src and os.path.isfile(src):
            overrides[rel.lower()] = src

    # 模型所在目录前缀：zip 内贴图相对模型文件放置
    mdl_dir = os.path.dirname(model_rel)
    prefix = (mdl_dir.replace('\\', '/') + '/') if mdl_dir else ''

    def _zip_name(rel):
        rel = rel.replace('\\', '/')
        if prefix and rel.startswith(prefix):
            rel = rel[len(prefix):]
        return rel

    def _img_bytes_for(path, ext):
        """把结果图尽量转回模型引用的扩展名格式。"""
        try:
            from PIL import Image as _I
            im = _I.open(path)
            if im.mode == 'RGBA' and ext not in ('.png', '.bmp', '.webp'):
                bg = _I.new('RGB', im.size, (255, 255, 255))
                bg.paste(im, mask=im.getchannel('A'))
                im = bg
            buf = io.BytesIO()
            fmt = {'.bmp': 'BMP', '.jpg': 'JPEG', '.jpeg': 'JPEG',
                   '.webp': 'WEBP', '.png': 'PNG'}.get(ext.lower())
            if fmt is None:
                return None
            im.save(buf, fmt)
            return buf.getvalue()
        except Exception:
            return None

    replaced_names = []

    arc_names = []
    for dirpath, _dirs, fnames in os.walk(root):
        for fn in fnames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace('\\', '/')
            target = _zip_name(rel)
            if not target:
                continue
            arc_names.append((rel, target, full))
    arc_names.sort(key=lambda x: x[1])

    model_name = os.path.basename(model_rel.replace('\\', '/'))
    base = os.path.splitext(model_name)[0] or 'model'
    zname = 'TextureUpscaler_%s_%s.zip' % (base, uuid.uuid4().hex[:8])
    zpath = os.path.join(EXPORT_DIR, zname)
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as zf:
        for rel, target, full in arc_names:
            data_b = None
            ov = overrides.get(rel.lower())
            if ov:
                ext = os.path.splitext(target)[1]
                data_b = _img_bytes_for(ov, ext)
                if data_b is not None:
                    replaced_names.append(target)
            if data_b is None:
                with open(full, 'rb') as f:
                    data_b = f.read()
            zf.writestr(target, data_b)
        readme = ('TextureUpscaler 模型导出\r\n'
                  '模型文件位于本压缩包内，贴图已按模型引用的相对路径放置。\r\n'
                  '已替换为超分结果的贴图：\r\n' +
                  ('\r\n'.join(' - ' + r for r in replaced_names) if replaced_names else ' - 无'))
        zf.writestr('导出说明.txt', readme)
    size = os.path.getsize(zpath)
    return jsonify({'file': zname, 'path': zpath, 'size': size})


@app.route('/api/open-export/<path:name>')
def open_export(name):
    # 在文件管理器中选中导出的 zip（本机应用，绕开浏览器下载与 IDM）
    safe = os.path.basename(name)
    p = os.path.join(EXPORT_DIR, safe)
    if not os.path.isfile(p):
        abort(404)
    try:
        subprocess.Popen(['explorer', '/select,', p])
    except Exception:
        try:
            subprocess.Popen('explorer /select,"%s"' % p)
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)})
    return jsonify({'ok': True})


# ---------------- PBR 贴图生成 ----------------
def _fnum(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def generate_pbr(img, strength, rough_base, metal_thresh, invert):
    from PIL import ImageFilter
    rgb = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.299*R + 0.587*G + 0.114*B
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    height = 1.0 - lum if invert else lum

    gy, gx = np.gradient(height)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(height)
    norm = np.sqrt(nx*nx + ny*ny + nz*nz)
    normal = np.stack([nx/norm*0.5+0.5, ny/norm*0.5+0.5, nz/norm*0.5+0.5], axis=2)

    h_img = Image.fromarray((np.clip(height, 0, 1)*255).astype(np.uint8), 'L')
    ao = np.asarray(h_img.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)/255.0
    ao = np.clip(ao*1.1, 0, 1)

    rough = np.clip(rough_base + (1.0-lum)*0.5 - sat*0.1, 0, 1)
    metal = np.clip((lum - metal_thresh) * (sat + 0.2) * 3.0, 0, 1)

    def toL(a):
        return Image.fromarray((np.clip(a, 0, 1)*255).astype(np.uint8), 'L')
    return {
        'height': toL(height),
        'normal': Image.fromarray((np.clip(normal, 0, 1)*255).astype(np.uint8), 'RGB'),
        'ao': toL(ao),
        'roughness': toL(rough),
        'metallic': toL(metal),
    }


@app.route('/pbr')
def pbr_page():
    return send_from_directory(STATIC_DIR, 'pbr.html')


@app.route('/api/pbr', methods=['POST'])
def pbr():
    data = request.get_json(silent=True) or {}
    img_id = data.get('img_id')
    src = _find_upload(img_id)
    if not src:
        return jsonify({'error': '图片不存在'}), 404
    strength = _clamp(_fnum(data.get('strength'), 1.0), 0.0, 5.0)
    rough_base = _clamp(_fnum(data.get('rough_base'), 0.5), 0.0, 1.0)
    metal_thresh = _clamp(_fnum(data.get('metal_thresh'), 0.6), 0.0, 1.0)
    invert = bool(data.get('invert'))
    try:
        maps = generate_pbr(Image.open(src), strength, rough_base, metal_thresh, invert)
    except Exception as e:
        return jsonify({'error': '生成失败：' + str(e)}), 500
    out = {}
    for key, pim in maps.items():
        name = f'{img_id}_pbr_{key}.png'
        pim.save(os.path.join(OUTPUT_DIR, name), 'PNG')
        out[key] = '/files/' + name
    return jsonify({'maps': out})


if __name__ == '__main__':
    print('服务已启动: http://127.0.0.1:5000')
    app.run(host='0.0.0.0', port=5000, threaded=True)
