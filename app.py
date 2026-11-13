import os
import io
import gc
import json
import tempfile
import urllib.request
import subprocess
import numpy as np
import cv2
import rawpy
import tifffile
import imagecodecs
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import streamlit as st

st.set_page_config(
    page_title="Laboratório de Diagnóstico de Sensores",
    layout="wide",
    initial_sidebar_state="expanded"
)

CATALOGO_SENSORES = {
    "ILCE-7M3": "Sony IMX410 (Full-Frame 24.2 MP BSI-CMOS, 14-bit)",
    "ILCE-6000": "Sony IMX193 (APS-C 24.3 MP Exmor APS HD CMOS, 12-bit)",
    "ILCE-7RM4": "Sony IMX455 (Full-Frame 61.0 MP BSI-CMOS, 14-bit)",
    "ILCE-7SM3": "Sony IMX510 (Full-Frame 12.1 MP BSI-CMOS, 14-bit)"
}

EXTENSOES_RAW = ('.arw', '.cr2', '.cr3', '.nef', '.dng', '.orf', '.rw2', '.pef', '.raf')

# ==============================================================================
# 1. FUNCOES DE LEITURA OTIMIZADAS PARA BAIXO CONSUMO DE RAM (STREAMLIT CLOUD)
# ==============================================================================

def extrair_metadados_exif(caminho_arquivo):
    try:
        res_num = subprocess.run(['exiftool', '-j', '-n', caminho_arquivo], capture_output=True, text=True)
        res_fmt = subprocess.run(['exiftool', '-j', caminho_arquivo], capture_output=True, text=True)
        dados_num = json.loads(res_num.stdout)[0] if res_num.stdout else {}
        dados_fmt = json.loads(res_fmt.stdout)[0] if res_fmt.stdout else {}

        chaves_shutter = ['ShutterCount', 'ImageCount', 'ShutterCount2', 'ShutterCount3', 'TotalShutterReleases', 'ImageNumber', 'ExposureCount']
        cliques = "Não disponível (DNG Mobile / Obturador Eletrônico)"
        for chave in chaves_shutter:
            if chave in dados_num:
                cliques = f"{dados_num[chave]} cliques (Tag: {chave})"
                break

        fabricante = str(dados_fmt.get('Make', 'Desconhecido')).strip()
        modelo = str(dados_fmt.get('Model', 'Desconhecido')).strip()
        sensor_id = dados_fmt.get('SensorModel', dados_fmt.get('UniqueCameraModel', CATALOGO_SENSORES.get(modelo, f"Sensor Nativo ({fabricante} {modelo})")))

        return {
            'Fabricante': fabricante,
            'Modelo': modelo,
            'Sensor': sensor_id,
            'Cliques': cliques,
            'ISO': dados_fmt.get('ISO', dados_fmt.get('ISOSpeedRatings', 'N/A')),
            'Exposicao': dados_fmt.get('ExposureTime', dados_fmt.get('ShutterSpeed', 'N/A')),
            'Temp_Sensor': dados_fmt.get('CameraTemperature', dados_fmt.get('SensorTemperature', 'N/A')),
            'Bits_EXIF': dados_num.get('BitsPerSample', 'N/A'),
            'CFA_EXIF': dados_fmt.get('CFAPattern', dados_fmt.get('CFAPattern2', 'N/A')),
            'BlackLevel_EXIF': dados_num.get('BlackLevel', dados_fmt.get('BlackLevel', 'N/A')),
            'WhiteLevel_EXIF': dados_num.get('WhiteLevel', 'N/A')
        }
    except Exception as e:
        return {'Fabricante': 'Erro', 'Modelo': str(e)}

def _converter_lista_floats(valor_exif, tamanho_padrao=4, padrao=0.0):
    if isinstance(valor_exif, (int, float)):
        return [float(valor_exif)] * tamanho_padrao
    if isinstance(valor_exif, list):
        vals = [float(x) for x in valor_exif if isinstance(x, (int, float))]
        return (vals * tamanho_padrao)[:tamanho_padrao] if vals else [padrao] * tamanho_padrao
    if isinstance(valor_exif, str):
        vals = []
        for token in valor_exif.replace(',', ' ').split():
            try:
                vals.append(float(token))
            except ValueError:
                continue
        return (vals * tamanho_padrao)[:tamanho_padrao] if vals else [padrao] * tamanho_padrao
    return [padrao] * tamanho_padrao

def _ler_maior_subifd_dng(caminho_arquivo):
    melhor_matriz = None
    maior_total_pixels = 0
    with tifffile.TiffFile(caminho_arquivo) as tif:
        paginas_candidatas = list(tif.pages)
        for pag in list(tif.pages):
            if getattr(pag, 'subifds', None):
                paginas_candidatas.extend(pag.subifds)
        for pag in paginas_candidatas:
            try:
                arr = pag.asarray()
                if arr is not None and arr.size > maior_total_pixels:
                    melhor_matriz = arr
                    maior_total_pixels = arr.size
            except Exception:
                continue
    if melhor_matriz is None:
        melhor_matriz = tifffile.imread(caminho_arquivo)
    return melhor_matriz

def _reduzir_para_preview(rgb_float, largura_max=1400):
    """Reduz a matriz RGB de visualizacao global para economizar ~285 MB de RAM."""
    h, w = rgb_float.shape[:2]
    if w > largura_max:
        fator = largura_max / float(w)
        novo_h = max(1, int(round(h * fator)))
        rgb_small = cv2.resize(rgb_float, (largura_max, novo_h), interpolation=cv2.INTER_AREA)
        return rgb_small.astype(np.float16)
    return rgb_float.astype(np.float16)

@st.cache_data(max_entries=2, show_spinner=False)
def processar_bytes_imagem(file_bytes, nome_original):
    """
    Decodifica o arquivo mantendo raw_adu em uint16 (metade da memoria de float32)
    e rgb_preview compactado em float16, evitando Out-Of-Memory no Streamlit Cloud.
    """
    ext = os.path.splitext(nome_original)[1].lower()
    tamanho_mb = len(file_bytes) / (1024.0 * 1024.0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        caminho_arquivo = tmp.name

    try:
        exif = extrair_metadados_exif(caminho_arquivo)
        raw_adu = None
        rgb_preview = None
        bl_canais = [0.0, 0.0, 0.0, 0.0]
        wl = 65535.0
        padrao_cfa = str(exif.get('CFA_EXIF', 'RGB / Linear'))
        tipo_arquivo = "RAW Bayer Nativo"

        if ext in EXTENSOES_RAW:
            try:
                with open(caminho_arquivo, 'rb') as f_bin:
                    with rawpy.imread(f_bin) as raw:
                        arr = raw.raw_image_visible.copy()
                        bl_raw = list(raw.black_level_per_channel)
                        bl_canais = [float(x) for x in bl_raw[:4]] if bl_raw else [0.0] * 4
                        wl = float(raw.white_level) if raw.white_level > 0 else float(np.max(arr))

                        try:
                            desc = raw.color_desc.decode('utf-8', errors='ignore')
                            pat = raw.raw_pattern
                            if pat is not None and pat.size == 4:
                                padrao_cfa = "".join([desc[i] for i in pat.flatten()])
                        except Exception:
                            pass

                        if arr.ndim == 2:
                            raw_adu = np.clip(arr, 0, 65535).astype(np.uint16)
                        else:
                            tipo_arquivo = "Linear DNG Mobile (Multicanal)"
                            raw_adu = np.clip(np.mean(arr[:, :, :3], axis=2), 0, 65535).astype(np.uint16)
                        del arr

                        try:
                            # Usa half_size=True para gerar o preview global gastando 4x menos RAM e CPU!
                            rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, half_size=True)
                            rgb_preview = _reduzir_para_preview(rgb.astype(np.float32) / 65535.0)
                            del rgb
                        except Exception:
                            bl_m = float(np.mean(bl_canais))
                            sub_raw = raw_adu[::2, ::2].astype(np.float32)
                            norm = np.clip((sub_raw - bl_m) / max(1.0, wl - bl_m), 0.0, 1.0)
                            rgb_preview = _reduzir_para_preview(np.stack([norm] * 3, axis=-1))
            except Exception:
                arr = _ler_maior_subifd_dng(caminho_arquivo)
                tipo_arquivo = "DNG Mobile (SubIFD TiffFile + ImageCodecs)"
                bl_canais = _converter_lista_floats(exif.get('BlackLevel_EXIF', 0), 4, 0.0)
                wl_lista = _converter_lista_floats(exif.get('WhiteLevel_EXIF', 'N/A'), 1, -1.0)
                if wl_lista[0] > 0:
                    wl = wl_lista[0]
                else:
                    max_obs = float(np.max(arr))
                    wl = 255.0 if max_obs <= 255 else (1023.0 if max_obs <= 1023 else (4095.0 if max_obs <= 4095 else (16383.0 if max_obs <= 16383 else 65535.0)))

                bl_m = float(np.mean(bl_canais))
                if arr.ndim == 3:
                    raw_adu = np.clip(np.mean(arr[:, :, :3], axis=2), 0, 65535).astype(np.uint16)
                    sub_rgb = arr[::2, ::2, :3].astype(np.float32)
                    norm_rgb = np.clip((sub_rgb - bl_m) / max(1.0, wl - bl_m), 0.0, 1.0)
                    rgb_preview = _reduzir_para_preview(norm_rgb)
                else:
                    raw_adu = np.clip(arr, 0, 65535).astype(np.uint16)
                    sub_raw = raw_adu[::2, ::2].astype(np.float32)
                    norm = np.clip((sub_raw - bl_m) / max(1.0, wl - bl_m), 0.0, 1.0)
                    rgb_preview = _reduzir_para_preview(np.stack([norm] * 3, axis=-1))
                del arr
        else:
            tipo_arquivo = "Imagem Processada (Não-RAW)"
            img = cv2.imread(caminho_arquivo, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Não foi possível decodificar a imagem.")
            wl = 65535.0 if img.dtype == np.uint16 else 255.0
            if img.ndim == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                raw_adu = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.uint16)
                rgb_preview = _reduzir_para_preview(img_rgb.astype(np.float32) / wl)
            else:
                raw_adu = img.astype(np.uint16)
                norm = img.astype(np.float32) / wl
                rgb_preview = _reduzir_para_preview(np.stack([norm] * 3, axis=-1))
            del img

        bits_adc = int(np.ceil(np.log2(max(2.0, wl + 1.0))))
        if wl <= 255:
            bits_adc = 8

        gc.collect()
        return {
            'arquivo': nome_original,
            'tamanho_mb': tamanho_mb,
            'tipo_arquivo': tipo_arquivo,
            'raw_adu': raw_adu,
            'rgb_preview': rgb_preview,
            'bl_canais': [round(x, 1) for x in bl_canais],
            'bl_medio': float(np.mean(bl_canais)),
            'white_level': wl,
            'bits_adc': bits_adc,
            'padrao_cfa': padrao_cfa,
            'exif': exif
        }
    finally:
        if os.path.exists(caminho_arquivo):
            os.remove(caminho_arquivo)

# ==============================================================================
# 2. CALIBRACAO AUTOMATICA ULTRA-LEVE EM MEMORIA
# ==============================================================================

def calibrar_sensor_automatico(dados_sensor, sigma_hot=8.0, fator_dead_max=0.25):
    raw_adu = dados_sensor['raw_adu']
    bl_medio = dados_sensor['bl_medio']
    wl = dados_sensor['white_level']

    media_global = float(np.mean(raw_adu))
    mediana_global = float(np.median(raw_adu))
    rms_total = float(np.std(raw_adu))

    sigmas_canais = []
    medianas_canais = []
    for r in range(2):
        for c in range(2):
            sub = raw_adu[r::2, c::2]
            med_c = float(np.median(sub))
            mad_c = float(np.median(np.abs(sub.astype(np.float32) - med_c)))
            sigmas_canais.append(1.4826 * mad_c)
            medianas_canais.append(med_c)

    sigma_mad = float(np.mean(sigmas_canais))
    if sigma_mad <= 0:
        sigma_mad = rms_total if rms_total > 0 else 1.0

    sigma_efetivo = max(sigma_mad, rms_total)
    pedestal_ref = bl_medio if bl_medio > 0 else mediana_global
    excursao_util = max(1.0, wl - pedestal_ref)

    dr_stops = float(np.log2(excursao_util / sigma_mad))
    dr_db = float(20.0 * np.log10(excursao_util / sigma_mad))

    fpn_h = float(np.std(np.mean(raw_adu, axis=1)))
    fpn_v = float(np.std(np.mean(raw_adu, axis=0)))

    margem_min_hot = max(15.0, excursao_util * 0.018)
    limiar_hot_auto = int(min(wl, max(mediana_global + sigma_hot * sigma_efetivo, pedestal_ref + margem_min_hot)))

    if pedestal_ref > 20.0:
        limiar_dead_auto = int(max(0.0, min(pedestal_ref * fator_dead_max, mediana_global - 12.0 * sigma_efetivo)))
    else:
        limiar_dead_auto = 0

    return {
        'media': media_global,
        'mediana': mediana_global,
        'rms_total': rms_total,
        'sigma_mad': sigma_mad,
        'medianas_cfa': [round(x, 1) for x in medianas_canais],
        'sigmas_cfa': [round(x, 2) for x in sigmas_canais],
        'dr_stops': dr_stops,
        'dr_db': dr_db,
        'fpn_h': fpn_h,
        'fpn_v': fpn_v,
        'limiar_hot_auto': limiar_hot_auto,
        'limiar_dead_auto': limiar_dead_auto
    }

def analisar_flat_frame(dados_flat, limiar_queda_dead=0.50):
    raw_flat = dados_flat['raw_adu']
    bl = dados_flat['bl_medio']

    prnu_canais = []
    num_dead_flat = 0
    num_poeira = 0

    for r in range(2):
        for c in range(2):
            sub = np.maximum(1.0, raw_flat[r::2, c::2].astype(np.float32) - bl)
            fundo_local = uniform_filter(sub, size=31)
            razao = sub / np.maximum(1.0, fundo_local)
            prnu_canais.append(float(np.std(razao)) * 100.0)
            mask_d = razao <= (1.0 - limiar_queda_dead)
            num_dead_flat += int(np.sum(mask_d))
            num_poeira += int(np.sum((razao < 0.90) & (~mask_d)))
            del sub, fundo_local, razao

    gc.collect()
    return {
        'prnu_pct': float(np.mean(prnu_canais)),
        'num_dead_flat': num_dead_flat,
        'pct_poeira': float(num_poeira) / raw_flat.size * 100.0
    }

@st.cache_data(max_entries=2, show_spinner=False)
def baixar_exemplo_github(nome_arq):
    url = f"https://raw.githubusercontent.com/renato-f-bandeira/files/main/{nome_arq}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=25) as resp:
        return resp.read()

# ==============================================================================
# 3. INTERFACE WEB (STREAMLIT)
# ==============================================================================

st.title("🔬 Laboratório de Diagnóstico e Certificação de Sensores")
st.markdown(
    "Envie seus arquivos **RAW** (`.ARW`, `.CR2`, `.CR3`, `.NEF`, `.DNG`, `.ORF`, `.RW2`) na barra lateral. "
    "Os arquivos são processados em memória temporária otimizada."
)

with st.sidebar:
    st.header("📥 1. Inserir Imagens do Sensor")
    arq_dark = st.file_uploader("1. Dark Frame (Principal - Lente Tampada)", type=['arw', 'cr2', 'cr3', 'nef', 'dng', 'orf', 'rw2', 'pef', 'raf', 'tif', 'tiff', 'png', 'jpg'])
    arq_flat = st.file_uploader("2. Flat Frame (Opcional - Fundo Uniforme Claro)", type=['arw', 'cr2', 'cr3', 'nef', 'dng', 'orf', 'rw2', 'pef', 'raf', 'tif', 'tiff', 'png', 'jpg'])
    arq_bias = st.file_uploader("3. Offset / Bias Frame (Opcional - Exp. Mínima)", type=['arw', 'cr2', 'cr3', 'nef', 'dng', 'orf', 'rw2', 'pef', 'raf', 'tif', 'tiff', 'png', 'jpg'])

    st.divider()
    st.subheader("🧪 Ou testar com Exemplo Oficial")
    exemplo_escolhido = st.selectbox(
        "Carregar Dark Frame do GitHub:",
        ["Nenhum (Usar meu upload)", "DSC02212.ARW (Sony A7 III)", "DSC00906.ARW (Sony A6000)"]
    )

dados_principal = None
try:
    if arq_dark is not None:
        with st.spinner(f"Decodificando matriz RAW de '{arq_dark.name}'..."):
            dados_principal = processar_bytes_imagem(arq_dark.getvalue(), arq_dark.name)
    elif exemplo_escolhido != "Nenhum (Usar meu upload)":
        nome_ex = exemplo_escolhido.split(" ")[0]
        with st.spinner(f"Baixando e decodificando '{nome_ex}' do GitHub..."):
            bytes_ex = baixar_exemplo_github(nome_ex)
            dados_principal = processar_bytes_imagem(bytes_ex, nome_ex)
except Exception as erro_leitura:
    st.error(f"Erro ao processar a imagem: {erro_leitura}")
    st.stop()

if dados_principal is None:
    st.info("👈 Faça o upload de um **Dark Frame** na barra lateral esquerda ou selecione uma das câmeras de exemplo para iniciar o diagnóstico.")
    st.stop()

cal = calibrar_sensor_automatico(dados_principal)
raw_adu = dados_principal['raw_adu']
h, w = raw_adu.shape
wl_int = int(max(255, dados_principal['white_level']))

with st.sidebar:
    st.divider()
    st.header("🎛️ 2. Parâmetros de Inspeção")
    ev = st.slider("Ganho Visual de Exposição (EV):", -2.0, 12.0, 4.0, 0.5)
    modo_zoom = st.selectbox("Modo da Janela de Zoom:", ["Matriz Bayer Bruta 1:1 (ADU)", "RGB Simulado (Nível de Canal)"])
    tam_janela = st.selectbox("Tamanho do Recorte de Zoom:", [30, 60, 120, 300], index=1)
    marcar_defeitos = st.checkbox("Destacar Hot/Dead Pixels nos Mapas", value=True)

    limiar_hot = st.slider("Limiar Hot Pixel (ADU):", 0, wl_int, int(cal['limiar_hot_auto']))
    max_dead_slider = max(50, int(dados_principal['bl_medio'] * 1.2) if dados_principal['bl_medio'] > 0 else wl_int // 4)
    limiar_dead = st.slider("Limiar Dead Pixel no Dark (ADU):", 0, max_dead_slider, int(cal['limiar_dead_auto']))

    st.subheader("🔍 Coordenadas do Zoom (X, Y)")
    if "zoom_x" not in st.session_state or st.session_state.get("arq_atual") != dados_principal['arquivo']:
        st.session_state.zoom_x = w // 2
        st.session_state.zoom_y = h // 2
        st.session_state.arq_atual = dados_principal['arquivo']

    if st.button("🔥 Focar no Pixel Mais Quente"):
        idx_max = int(np.argmax(raw_adu))
        ym, xm = np.unravel_index(idx_max, raw_adu.shape)
        st.session_state.zoom_x = int(xm)
        st.session_state.zoom_y = int(ym)

    cx = st.slider("Centro Zoom X (px):", 0, w - 1, key="zoom_x")
    cy = st.slider("Centro Zoom Y (px):", 0, h - 1, key="zoom_y")

# Coordenadas de pixels defeituosos (extraidas diretamente sem criar mascaras booleanas gigantes permanentes)
yh, xh = np.where(raw_adu >= limiar_hot)
yd, xd = np.where(raw_adu <= limiar_dead)
num_hot = len(xh)
num_dead = len(xd)
total_pixels = raw_adu.size

# Painel de Metricas Superiores
exif = dados_principal['exif']
col1, col2, col3, col4 = st.columns(4)
col1.metric("Câmera / Modelo", f"{exif.get('Fabricante')} {exif.get('Modelo')}", exif.get('Cliques'))
col2.metric("Ruído Robusto (MAD)", f"{cal['sigma_mad']:.2f} ADU", f"RMS Global: {cal['rms_total']:.2f} ADU", delta_color="off")
col3.metric("Faixa Dinâmica (DR)", f"{cal['dr_stops']:.2f} EV", f"{cal['dr_db']:.1f} dB | {dados_principal['bits_adc']}-bits")
col4.metric("Hot / Dead Pixels (Dark)", f"{num_hot} Hot", f"{num_dead} Dead (Bias: {dados_principal['bl_medio']:.0f} ADU)", delta_color="off")

# Analise combinada opcional (Bias / Flat)
if arq_bias is not None or arq_flat is not None:
    st.subheader("🧩 Diagnóstico Multi-Frame (Dark + Flat + Offset/Bias)")
    cols_multi = st.columns(2)

    if arq_bias is not None:
        dados_bias = processar_bytes_imagem(arq_bias.getvalue(), arq_bias.name)
        cal_bias = calibrar_sensor_automatico(dados_bias)
        corrente_termica = max(0.0, cal['sigma_mad']**2 - cal_bias['sigma_mad']**2)**0.5
        with cols_multi[0]:
            st.success(
                f"**Análise de Offset/Bias (`{arq_bias.name}`):**\n"
                f"- **Ruído de Leitura Puro (Read Noise):** `{cal_bias['sigma_mad']:.2f} ADU`\n"
                f"- **Ruído Térmico Isolado (Dark Current):** `{corrente_termica:.2f} ADU`"
            )

    if arq_flat is not None:
        dados_flat = processar_bytes_imagem(arq_flat.getvalue(), arq_flat.name)
        res_flat = analisar_flat_frame(dados_flat)
        with cols_multi[1]:
            st.warning(
                f"**Análise de Flat Frame (`{arq_flat.name}`):**\n"
                f"- **Dead Pixels Verdadeiros (Cegos à Luz):** `{res_flat['num_dead_flat']} pixels`\n"
                f"- **Não-Uniformidade de Resposta (PRNU):** `{res_flat['prnu_pct']:.2f}%`\n"
                f"- **Área Afetada por Poeira/Manchas:** `{res_flat['pct_poeira']:.3f}% do sensor`"
            )

# Preparacao leve para plotagem
ganho = 2.0 ** ev
rgb_preview_exp = np.clip(dados_principal['rgb_preview'].astype(np.float32) * ganho, 0.0, 1.0)

meia_janela = tam_janela // 2
cx_clip = int(np.clip(cx, meia_janela, w - meia_janela))
cy_clip = int(np.clip(cy, meia_janela, h - meia_janela))
x1, x2 = cx_clip - meia_janela, cx_clip + meia_janela
y1, y2 = cy_clip - meia_janela, cy_clip + meia_janela

recorte_adu = raw_adu[y1:y2, x1:x2].astype(np.float32)
bl = dados_principal['bl_medio']
wl = dados_principal['white_level']

fig = plt.figure(figsize=(15, 9.8), dpi=110, facecolor='white')
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.25, 1], width_ratios=[1.15, 1], hspace=0.30, wspace=0.22)

ax_full = fig.add_subplot(gs[0, 0])
ax_zoom = fig.add_subplot(gs[0, 1])
ax_hist = fig.add_subplot(gs[1, :])

# 1. Quadro Completo (usa o preview em coordenadas originais extent=[0, w, h, 0] gastando apenas 4 MB de RAM)
ax_full.imshow(rgb_preview_exp, interpolation='nearest', extent=[0, w, h, 0])
ax_full.set_title(f"1. Quadro Completo (+{ev:.1f} EV | {ganho:.1f}x)", fontsize=11, fontweight='bold')
ax_full.set_xlabel("Largura X (px)", fontsize=9)
ax_full.set_ylabel("Altura Y (px)", fontsize=9)

rect = patches.Rectangle((x1, y1), tam_janela, tam_janela, linewidth=2.0, edgecolor='#FFD700', facecolor='none', label='Janela de Zoom')
ax_full.add_patch(rect)
ax_full.plot(cx_clip, cy_clip, marker='+', color='#FFD700', markersize=10, markeredgewidth=1.8)

if marcar_defeitos:
    if num_hot > 0:
        ax_full.scatter(xh[:2000], yh[:2000], s=28, facecolors='none', edgecolors='#FF2222', linewidth=1.2, label=f'Hot ({num_hot})')
    if num_dead > 0:
        ax_full.scatter(xd[:2000], yd[:2000], s=28, facecolors='none', edgecolors='#00FFFF', linewidth=1.2, label=f'Dead ({num_dead})')
ax_full.legend(loc='upper right', framealpha=0.9, fontsize=8)

# 2. Janela de Zoom 1:1 (Resolucao 100% Nativa)
base_ref = bl if bl > 0 else cal['mediana']
amplitude_adu = max(30.0, (wl - base_ref) / ganho)
vmin_adu = max(0.0, base_ref - 4.0 * cal['sigma_mad'])
vmax_adu = min(wl, base_ref + amplitude_adu)

if "Bayer" in modo_zoom:
    im_z = ax_zoom.imshow(recorte_adu, cmap='inferno', vmin=vmin_adu, vmax=vmax_adu, interpolation='nearest', extent=[x1, x2, y2, y1])
    cbar = fig.colorbar(im_z, ax=ax_zoom, fraction=0.046, pad=0.04)
    cbar.set_label("Leitura Bruta (ADU)", fontsize=9)
else:
    recorte_norm = np.clip((recorte_adu - base_ref) / max(1.0, wl - base_ref) * ganho, 0.0, 1.0)
    ax_zoom.imshow(np.stack([recorte_norm]*3, axis=-1), interpolation='nearest', extent=[x1, x2, y2, y1])

ax_zoom.set_title(f"2. Inspeção 1:1 ({tam_janela}x{tam_janela} px) em ({cx_clip}, {cy_clip})", fontsize=11, fontweight='bold')
ax_zoom.set_xlabel(f"Coordenada X (Centro: {cx_clip} px)", fontsize=9)
ax_zoom.set_ylabel(f"Coordenada Y (Centro: {cy_clip} px)", fontsize=9)

if tam_janela <= 60:
    ax_zoom.set_xticks(np.arange(x1, x2 + 1, 1), minor=True)
    ax_zoom.set_yticks(np.arange(y1, y2 + 1, 1), minor=True)
    ax_zoom.grid(which='minor', color='white', linestyle='-', linewidth=0.35, alpha=0.35)

if marcar_defeitos:
    yh_z, xh_z = np.where(recorte_adu >= limiar_hot)
    if len(xh_z) > 0:
        ax_zoom.scatter(x1 + xh_z + 0.5, y1 + yh_z + 0.5, s=100, facecolors='none', edgecolors='#00FF00', linewidth=1.8, label='Hot Pixel')
    yd_z, xd_z = np.where(recorte_adu <= limiar_dead)
    if len(xd_z) > 0:
        ax_zoom.scatter(x1 + xd_z + 0.5, y1 + yd_z + 0.5, s=100, facecolors='none', edgecolors='#00FFFF', linewidth=1.8, label='Dead Pixel')
    if len(xh_z) > 0 or len(xd_z) > 0:
        ax_zoom.legend(loc='upper right', framealpha=0.9, fontsize=8)

# 3. Histograma ADU via np.bincount (Zero alocacao extra de RAM e 20x mais rapido)
contagens_adu = np.bincount(raw_adu.ravel())
valores_adu_x = np.nonzero(contagens_adu)[0]
contagens_y = contagens_adu[valores_adu_x]

ax_hist.bar(valores_adu_x, contagens_y, width=max(1.0, (valores_adu_x[-1] - valores_adu_x[0]) / 250.0), color='#1f4e78', log=True, alpha=0.88, label='Contagem de Pixels (ADU)')
ax_hist.axvline(bl if bl > 0 else cal['mediana'], color='#2ca02c', linestyle='-', linewidth=1.8, label=f'Pedestal ({bl if bl > 0 else cal["mediana"]:.0f} ADU)')
ax_hist.axvline(cal['mediana'] + cal['sigma_mad'], color='#ff7f0e', linestyle=':', linewidth=1.5, label=f'+1 Sigma MAD (+{cal["sigma_mad"]:.1f} ADU)')
ax_hist.axvline(limiar_hot, color='#d62728', linestyle='--', linewidth=1.8, label=f'Limiar Hot ({limiar_hot} ADU)')
ax_hist.axvline(limiar_dead, color='#17becf', linestyle='--', linewidth=1.8, label=f'Limiar Dead ({limiar_dead} ADU)')
ax_hist.set_title(f"3. Distribuição ADU | Banding FPN (H/V): {cal['fpn_h']:.3f} / {cal['fpn_v']:.3f} ADU", fontsize=11, fontweight='bold')
ax_hist.set_xlabel("Valor de Leitura do Conversor ADC (ADU)", fontsize=9)
ax_hist.set_ylabel("Contagem de Pixels (Escala Log)", fontsize=9)
ax_hist.legend(loc='upper right', fontsize=9)
ax_hist.grid(True, which="both", ls="--", alpha=0.35)

# Salva em buffer PNG de 160 DPI e exibe direto na tela para liberar a figura da RAM imediatamente
buf_img = io.BytesIO()
fig.savefig(buf_img, format="png", dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
gc.collect()

bytes_png = buf_img.getvalue()
st.image(bytes_png, use_container_width=True)

st.download_button(
    label="📥 Baixar Relatório Gráfico em Alta Resolução (PNG)",
    data=bytes_png,
    file_name=f"Relatorio_Sensor_{os.path.splitext(dados_principal['arquivo'])[0]}.png",
    mime="image/png"
)

with st.expander("📄 Ver Relatório Técnico Completo em Texto", expanded=True):
    st.code(
        f"DIAGNOSTICO TECNICO DE SENSOR | Arquivo: {dados_principal['arquivo']} ({dados_principal['tamanho_mb']:.2f} MB) | {dados_principal['tipo_arquivo']}\n"
        f"- Fabricante / Modelo    : {exif.get('Fabricante')} {exif.get('Modelo')}\n"
        f"- Identificacao Sensor   : {exif.get('Sensor')}\n"
        f"- Contagem de Cliques    : {exif.get('Cliques')}\n"
        f"- ISO / Exposicao / Temp : ISO {exif.get('ISO')} | Exp: {exif.get('Exposicao')} s | Temp: {exif.get('Temp_Sensor')}\n"
        f"- Resolucao / CFA / Bits : {w} x {h} px ({total_pixels/1e6:.2f} MP) | CFA: {dados_principal['padrao_cfa']} | ADC: {dados_principal['bits_adc']}-bits\n"
        f"- Black Level por Canal  : {dados_principal['bl_canais']} ADU (Mediana Subcanais: {cal['medianas_cfa']} ADU | White Level: {wl:.0f} ADU)\n"
        f"- Ruido Robusto (MAD)    : {cal['sigma_mad']:.2f} ADU (Subcanais: {cal['sigmas_cfa']} ADU) | RMS Global: {cal['rms_total']:.2f} ADU\n"
        f"- Faixa Dinamica (DR)    : {cal['dr_stops']:.2f} EV | {cal['dr_db']:.1f} dB | Banding FPN (H/V): {cal['fpn_h']:.3f} / {cal['fpn_v']:.3f} ADU\n"
        f"- Hot / Dead Detectados  : Hot = {num_hot} px (>= {limiar_hot} ADU) | Dead = {num_dead} px (<= {limiar_dead} ADU)\n"
        f"- Recorte Atual do Zoom  : Centro ({cx_clip}, {cy_clip}) -> X[{x1}:{x2}], Y[{y1}:{y2}] | Min = {np.min(recorte_adu):.0f} ADU, Max = {np.max(recorte_adu):.0f} ADU"
    )
