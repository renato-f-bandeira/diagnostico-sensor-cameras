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
from PIL import Image, ImageDraw
from scipy.ndimage import median_filter, gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import streamlit as st

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    SUPORTA_CLIQUE_IMAGEM = True
except ImportError:
    SUPORTA_CLIQUE_IMAGEM = False

st.set_page_config(
    page_title="Sistema de Metrologia e Diagnóstico de Sensores CMOS/CCD",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1, h2, h3 { font-family: 'Segoe UI', Roboto, Helvetica, sans-serif; font-weight: 600; letter-spacing: -0.01em; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 600; }
    div[data-testid="stMetricLabel"] { font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. BASE DE DADOS DE SENSORES HOMOLOGADOS E ESPECIFICACOES FISICAS
# ==============================================================================
CATALOGO_SENSORES_HOMOLOGADOS = {
    "ILCE-7M3": {
        "nome_sensor": "Sony IMX410 (Full-Frame 24.2 MP BSI-CMOS)",
        "bits_adc_real": 14,
        "white_level_padrao": 16380.0,
        "black_level_padrao": 512.0,
        "cfa_padrao": "RGGB",
        "bias_clipado_zero": False
    },
    "ILCE-6000": {
        "nome_sensor": "Sony IMX193 (APS-C 24.3 MP Exmor CMOS, ADC 12-bit escalonado em 14-bit)",
        "bits_adc_real": 12,
        "white_level_padrao": 16383.0,
        "black_level_padrao": 512.0,
        "cfa_padrao": "RGGB",
        "bias_clipado_zero": False
    },
    "ILCE-7RM4": {
        "nome_sensor": "Sony IMX455 (Full-Frame 61.0 MP BSI-CMOS)",
        "bits_adc_real": 14,
        "white_level_padrao": 16380.0,
        "black_level_padrao": 512.0,
        "cfa_padrao": "RGGB",
        "bias_clipado_zero": False
    },
    "ILCE-7SM3": {
        "nome_sensor": "Sony IMX510 (Full-Frame 12.1 MP BSI-CMOS)",
        "bits_adc_real": 14,
        "white_level_padrao": 16380.0,
        "black_level_padrao": 512.0,
        "cfa_padrao": "RGGB",
        "bias_clipado_zero": False
    },
    "COOLPIX P1000": {
        "nome_sensor": "Nikon 1/2.3\" BSI-CMOS (16.0 MP, ADC 12-bit NRW)",
        "bits_adc_real": 12,
        "white_level_padrao": 4095.0,
        "black_level_padrao": 0.0,
        "cfa_padrao": "RGGB",
        "bias_clipado_zero": True
    },
    "COOLPIX P950": {
        "nome_sensor": "Nikon 1/2.3\" BSI-CMOS (16.0 MP, ADC 12-bit NRW)",
        "bits_adc_real": 12,
        "white_level_padrao": 4095.0,
        "black_level_padrao": 0.0,
        "cfa_padrao": "RGGB",
        "bias_clipado_zero": True
    }
}

EXTENSOES_RAW = ('.arw', '.cr2', '.cr3', '.nef', '.nrw', '.dng', '.orf', '.rw2', '.pef', '.raf', '.srw')

# ==============================================================================
# 2. CAMADA DE LEITURA DE HARDWARE E PROCESSAMENTO DE STACK (MASTER FRAMES)
# ==============================================================================

def _parse_tempo_exposicao_segundos(val_num, val_fmt):
    if isinstance(val_num, (int, float)) and val_num > 0:
        return float(val_num)
    if isinstance(val_fmt, str):
        limpo = val_fmt.strip().lower().replace('s', '').strip()
        if '/' in limpo:
            partes = limpo.split('/')
            try:
                return float(partes[0]) / max(1e-9, float(partes[1]))
            except ValueError:
                pass
        try:
            return float(limpo)
        except ValueError:
            pass
    return 0.0

def extrair_metadados_exif(caminho_arquivo):
    try:
        res_num = subprocess.run(['exiftool', '-j', '-n', caminho_arquivo], capture_output=True, text=True)
        res_fmt = subprocess.run(['exiftool', '-j', caminho_arquivo], capture_output=True, text=True)
        dados_num = json.loads(res_num.stdout)[0] if res_num.stdout else {}
        dados_fmt = json.loads(res_fmt.stdout)[0] if res_fmt.stdout else {}

        chaves_shutter = ['ShutterCount', 'ImageCount', 'ShutterCount2', 'ShutterCount3', 'TotalShutterReleases', 'ImageNumber', 'ExposureCount']
        cliques = "Nao disponivel"
        for chave in chaves_shutter:
            if chave in dados_num:
                cliques = f"{dados_num[chave]} acionamentos ({chave})"
                break

        fabricante = str(dados_fmt.get('Make', 'Desconhecido')).strip()
        modelo = str(dados_fmt.get('Model', 'Desconhecido')).strip()

        spec_homologada = None
        for mod_key, spec in CATALOGO_SENSORES_HOMOLOGADOS.items():
            if mod_key.upper() in modelo.upper():
                spec_homologada = spec
                break

        nome_sensor = spec_homologada["nome_sensor"] if spec_homologada else dados_fmt.get('SensorModel', dados_fmt.get('UniqueCameraModel', 'Nao catalogado'))

        exp_num = dados_num.get('ExposureTime', dados_num.get('ShutterSpeed', 0.0))
        exp_fmt = dados_fmt.get('ExposureTime', dados_fmt.get('ShutterSpeed', 'N/A'))
        exp_segundos = _parse_tempo_exposicao_segundos(exp_num, exp_fmt)

        return {
            'Fabricante': fabricante,
            'Modelo': modelo,
            'Sensor': nome_sensor,
            'Homologado_Catalogo': spec_homologada is not None,
            'Spec_Catalogo': spec_homologada,
            'Cliques': cliques,
            'ISO': dados_fmt.get('ISO', dados_fmt.get('ISOSpeedRatings', 'N/A')),
            'Exposicao_Str': str(exp_fmt),
            'Exposicao_Seg': exp_segundos,
            'Temp_Sensor': dados_fmt.get('CameraTemperature', dados_fmt.get('SensorTemperature', 'N/A')),
            'Bits_EXIF': dados_num.get('BitsPerSample', 'N/A'),
            'Compression_EXIF': dados_fmt.get('Compression', dados_fmt.get('NikonCompression', 'N/A')),
            'CFA_EXIF': dados_fmt.get('CFAPattern', dados_fmt.get('CFAPattern2', 'N/A')),
            'BlackLevel_EXIF': dados_fmt.get('BlackLevel', dados_fmt.get('BlackLevel', None)),
            'WhiteLevel_EXIF': dados_fmt.get('WhiteLevel', None)
        }
    except Exception as e:
        return {'Fabricante': 'Erro', 'Modelo': str(e), 'Homologado_Catalogo': False, 'Spec_Catalogo': None, 'Exposicao_Seg': 0.0, 'Exposicao_Str': 'N/A', 'ISO': 'N/A'}

def _converter_lista_floats(valor_exif, tamanho_padrao=4, padrao=None):
    if valor_exif is None:
        return [padrao] * tamanho_padrao
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

def _reduzir_para_preview(rgb_float, largura_max=880):
    h, w = rgb_float.shape[:2]
    if w > largura_max:
        fator = largura_max / float(w)
        novo_h = max(1, int(round(h * fator)))
        rgb_small = cv2.resize(rgb_float, (largura_max, novo_h), interpolation=cv2.INTER_AREA)
        return rgb_small.astype(np.float16)
    return rgb_float.astype(np.float16)

def _detectar_passo_quantizacao_adc(raw_adu):
    amostra = raw_adu[::8, ::8].ravel()
    unicos = np.unique(amostra)
    if len(unicos) > 10:
        diffs = np.diff(unicos)
        diffs_pos = diffs[diffs > 0]
        if len(diffs_pos) > 0:
            passo = int(np.median(diffs_pos))
            return max(1, passo)
    return 1

def processar_arquivo_individual(file_bytes, nome_original, exigir_catalogo_estrito=False):
    ext = os.path.splitext(nome_original)[1].lower()
    tamanho_mb = len(file_bytes) / (1024.0 * 1024.0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        caminho_arquivo = tmp.name

    try:
        exif = extrair_metadados_exif(caminho_arquivo)
        spec_cat = exif.get('Spec_Catalogo')

        raw_adu = None
        rgb_preview = None
        bl_canais = None
        wl = None
        padrao_cfa = str(exif.get('CFA_EXIF', 'N/A'))
        tipo_arquivo = "RAW Bayer Nativo"
        metodo_decodificacao = None

        if ext in EXTENSOES_RAW:
            try:
                with open(caminho_arquivo, 'rb') as f_bin:
                    with rawpy.imread(f_bin) as raw:
                        arr = raw.raw_image_visible.copy()
                        bl_raw = list(raw.black_level_per_channel)
                        if bl_raw and any(x is not None for x in bl_raw):
                            bl_canais = [float(x) for x in bl_raw[:4]]
                        if raw.white_level and raw.white_level > 0:
                            wl = float(raw.white_level)

                        try:
                            desc = raw.color_desc.decode('utf-8', errors='ignore')
                            pat = raw.raw_pattern
                            if pat is not None and pat.size == 4:
                                padrao_cfa = "".join([desc[i] for i in pat.flatten()])
                        except Exception:
                            pass

                        if arr.ndim == 2:
                            raw_adu = np.clip(arr, 0, 65535).astype(np.uint16)
                            metodo_decodificacao = "LibRaw_Bayer2D"
                        else:
                            tipo_arquivo = "Linear DNG (Multicanal)"
                            raw_adu = np.clip(np.mean(arr[:, :, :3], axis=2), 0, 65535).astype(np.uint16)
                            metodo_decodificacao = "LibRaw_Linear3D"
                        del arr

                        if spec_cat is not None:
                            if wl is None or wl <= 0:
                                wl = spec_cat["white_level_padrao"]
                            if bl_canais is None:
                                bl_canais = [spec_cat["black_level_padrao"]] * 4

                        bl_m_temp = float(np.mean(bl_canais)) if bl_canais else 0.0
                        wl_temp = wl if (wl and wl > 0) else float(np.max(raw_adu))

                        try:
                            rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, half_size=True)
                            rgb_preview = _reduzir_para_preview(rgb.astype(np.float32) / 65535.0)
                            del rgb
                        except Exception:
                            sub_raw = raw_adu[::2, ::2].astype(np.float32)
                            norm = np.clip((sub_raw - bl_m_temp) / max(1.0, wl_temp - bl_m_temp), 0.0, 1.0)
                            rgb_preview = _reduzir_para_preview(np.stack([norm] * 3, axis=-1))
            except Exception as erro_libraw:
                try:
                    arr = _ler_maior_subifd_dng(caminho_arquivo)
                    tipo_arquivo = "Container TIFF/DNG (SubIFD)"
                    metodo_decodificacao = "TiffFile_SubIFD"

                    bl_exif_list = _converter_lista_floats(exif.get('BlackLevel_EXIF'), 4, None)
                    if any(x is not None for x in bl_exif_list):
                        bl_canais = [float(x) if x is not None else 0.0 for x in bl_exif_list]
                    elif spec_cat is not None:
                        bl_canais = [spec_cat["black_level_padrao"]] * 4

                    wl_exif_list = _converter_lista_floats(exif.get('WhiteLevel_EXIF'), 1, None)
                    if wl_exif_list[0] is not None and wl_exif_list[0] > 0:
                        wl = float(wl_exif_list[0])
                    elif spec_cat is not None:
                        wl = spec_cat["white_level_padrao"]

                    bl_m_temp = float(np.mean(bl_canais)) if bl_canais else 0.0
                    wl_temp = wl if (wl and wl > 0) else float(np.max(arr))

                    if arr.ndim == 3:
                        raw_adu = np.clip(np.mean(arr[:, :, :3], axis=2), 0, 65535).astype(np.uint16)
                        sub_rgb = arr[::2, ::2, :3].astype(np.float32)
                        norm_rgb = np.clip((sub_rgb - bl_m_temp) / max(1.0, wl_temp - bl_m_temp), 0.0, 1.0)
                        rgb_preview = _reduzir_para_preview(norm_rgb)
                    else:
                        raw_adu = np.clip(arr, 0, 65535).astype(np.uint16)
                        sub_raw = raw_adu[::2, ::2].astype(np.float32)
                        norm = np.clip((sub_raw - bl_m_temp) / max(1.0, wl_temp - bl_m_temp), 0.0, 1.0)
                        rgb_preview = _reduzir_para_preview(np.stack([norm] * 3, axis=-1))
                    del arr
                except Exception:
                    return {
                        'elegivel': False,
                        'motivo_rejeicao': f"Falha na decodificação do RAW '{nome_original}' (LibRaw: {erro_libraw}).",
                        'exif': exif
                    }
        else:
            return {
                'elegivel': False,
                'motivo_rejeicao': f"O arquivo '{nome_original}' não é um formato RAW compatível.",
                'exif': exif
            }

        motivos_invalidez = []
        if exigir_catalogo_estrito and not exif.get('Homologado_Catalogo', False):
            motivos_invalidez.append("Modelo de câmera não consta no Catálogo Homologado.")
        if wl is None or wl <= 255:
            motivos_invalidez.append("White Level ADC inválido.")
        if bl_canais is None:
            motivos_invalidez.append("Black Level por canal ausente.")

        bl_medio = float(np.mean(bl_canais)) if bl_canais else 0.0
        fracao_zeros = float(np.sum(raw_adu == 0)) / raw_adu.size

        if not exif.get('Homologado_Catalogo', False):
            if bl_medio == 0.0 and fracao_zeros > 0.15:
                motivos_invalidez.append("Black Level truncado em 0 ADU sem parâmetro de catálogo correspondente.")

        if motivos_invalidez:
            return {
                'elegivel': False,
                'motivo_rejeicao': " | ".join(motivos_invalidez),
                'exif': exif
            }

        passo_adc = _detectar_passo_quantizacao_adc(raw_adu)
        bits_container = int(np.ceil(np.log2(max(2.0, wl + 1.0))))
        bits_efetivos = spec_cat["bits_adc_real"] if spec_cat else max(10, bits_container - int(np.round(np.log2(passo_adc))))

        gc.collect()
        return {
            'elegivel': True,
            'arquivo': nome_original,
            'tamanho_mb': tamanho_mb,
            'tipo_arquivo': tipo_arquivo,
            'raw_adu': raw_adu,
            'rgb_preview': rgb_preview,
            'bl_canais': [round(x, 1) for x in bl_canais],
            'bl_medio': bl_medio,
            'white_level': wl,
            'bits_container': bits_container,
            'bits_efetivos': bits_efetivos,
            'passo_adc': passo_adc,
            'padrao_cfa': padrao_cfa if padrao_cfa not in ('N/A', '') else (spec_cat["cfa_padrao"] if spec_cat else "RGGB"),
            'fracao_zeros': fracao_zeros,
            'exif': exif
        }
    finally:
        if os.path.exists(caminho_arquivo):
            os.remove(caminho_arquivo)

@st.cache_data(max_entries=3, show_spinner=False)
def processar_pilha_imagens(uploaded_files_list, exigir_catalogo_estrito=False):
    if not uploaded_files_list:
        return None

    resultados_individuais = []
    for file_obj in uploaded_files_list[:3]:
        res_ind = processar_arquivo_individual(file_obj.getvalue(), file_obj.name, exigir_catalogo_estrito)
        if res_ind and res_ind.get('elegivel', False):
            resultados_individuais.append(res_ind)

    if not resultados_individuais:
        return processar_arquivo_individual(uploaded_files_list[0].getvalue(), uploaded_files_list[0].name, exigir_catalogo_estrito)

    if len(resultados_individuais) == 1:
        return resultados_individuais[0]

    h_ref, w_ref = resultados_individuais[0]['raw_adu'].shape
    pilha_arrays = []
    for r in resultados_individuais:
        if r['raw_adu'].shape == (h_ref, w_ref):
            pilha_arrays.append(r['raw_adu'].astype(np.float32))

    master_raw = np.median(np.stack(pilha_arrays, axis=0), axis=0).astype(np.uint16)
    master_dict = resultados_individuais[0].copy()
    master_dict['arquivo'] = f"Master Frame ({len(resultados_individuais)} quadros empilhados)"
    master_dict['raw_adu'] = master_raw
    master_dict['tamanho_mb'] = sum(r['tamanho_mb'] for r in resultados_individuais)

    gc.collect()
    return master_dict

# ==============================================================================
# 3. MOTOR DE METROLOGIA: DARK, BIAS E STACK TERMICO
# ==============================================================================

def _encontrar_fim_cauda_termica_continua(raw_adu, pedestal_ref, wl, sigma_mad):
    contagens = np.bincount(raw_adu.ravel(), minlength=int(wl) + 1)
    inicio_busca = int(pedestal_ref + 6.0 * sigma_mad)
    largura_bin = max(16, int((wl - pedestal_ref) / 250))

    fim_cauda = int(pedestal_ref + 25.0 * sigma_mad)
    for b_ini in range(inicio_busca, int(wl) - largura_bin, largura_bin):
        soma_bin = int(np.sum(contagens[b_ini:b_ini + largura_bin]))
        if soma_bin <= 3:
            fim_cauda = b_ini
            break
    return fim_cauda

def analisar_dark_e_bias(dados_dark, dados_bias=None):
    raw_dark = dados_dark['raw_adu']
    bl_medio = dados_dark['bl_medio']
    wl = dados_dark['white_level']
    passo_adc = dados_dark['passo_adc']
    spec_cat = dados_dark['exif'].get('Spec_Catalogo')
    bias_clipado = (spec_cat and spec_cat.get("bias_clipado_zero", False)) or (bl_medio == 0.0)

    media_global = float(np.mean(raw_dark))
    mediana_global = float(np.median(raw_dark))
    rms_total = float(np.std(raw_dark))

    sigmas_canais = []
    medianas_canais = []
    fpn_h_canais = []
    fpn_v_canais = []

    for r in range(2):
        for c in range(2):
            sub = raw_dark[r::2, c::2]
            med_c = float(np.median(sub))
            if bias_clipado and med_c <= passo_adc:
                positivos = sub[sub > 0]
                q68 = float(np.percentile(positivos, 68.27)) if len(positivos) > 100 else float(np.std(sub))
                sigma_c = max(float(passo_adc), q68)
            else:
                mad_c = float(np.median(np.abs(sub.astype(np.float32) - med_c)))
                sigma_c = 1.4826 * max(mad_c, float(passo_adc) * 0.5)

            sigmas_canais.append(sigma_c)
            medianas_canais.append(med_c)
            fpn_h_canais.append(float(np.std(np.mean(sub, axis=1))))
            fpn_v_canais.append(float(np.std(np.mean(sub, axis=0))))

    sigma_mad = float(np.mean(sigmas_canais))
    if sigma_mad <= 0:
        sigma_mad = rms_total if rms_total > 0 else 1.0

    fpn_h = float(np.mean(fpn_h_canais))
    fpn_v = float(np.mean(fpn_v_canais))

    pedestal_ref = bl_medio if bl_medio > 0 else mediana_global
    excursao_util = max(1.0, wl - pedestal_ref)

    dr_stops = float(np.log2(excursao_util / sigma_mad))
    dr_db = float(20.0 * np.log10(excursao_util / sigma_mad))

    fim_cauda_continua = _encontrar_fim_cauda_termica_continua(raw_dark, pedestal_ref, wl, sigma_mad)
    limiar_warm_auto = int(min(wl * 0.5, max(pedestal_ref + 12.0 * sigma_mad, pedestal_ref + excursao_util * 0.035)))

    minimo_visivel_15ev = pedestal_ref + excursao_util * 0.15
    limiar_hot_auto = int(min(wl * 0.95, max(fim_cauda_continua, minimo_visivel_15ev)))
    if limiar_warm_auto >= limiar_hot_auto:
        limiar_warm_auto = int(pedestal_ref + (limiar_hot_auto - pedestal_ref) * 0.45)

    if pedestal_ref > 20.0 and not bias_clipado:
        limiar_dead_auto = int(max(0.0, min(pedestal_ref * 0.25, mediana_global - 12.0 * max(sigma_mad, rms_total))))
        suporta_dead_no_dark = True
    else:
        limiar_dead_auto = -1
        suporta_dead_no_dark = False

    metricas_bias = None
    if dados_bias is not None and dados_bias.get('elegivel', False):
        raw_bias = dados_bias['raw_adu']
        sigmas_b = []
        medianas_b = []
        for r in range(2):
            for c in range(2):
                sub_b = raw_bias[r::2, c::2]
                med_b = float(np.median(sub_b))
                if bias_clipado and med_b <= dados_bias['passo_adc']:
                    pos_b = sub_b[sub_b > 0]
                    sig_b = float(np.percentile(pos_b, 68.27)) if len(pos_b) > 100 else float(np.std(sub_b))
                    sigmas_b.append(max(float(dados_bias['passo_adc']), sig_b))
                else:
                    mad_b = float(np.median(np.abs(sub_b.astype(np.float32) - med_b)))
                    sigmas_b.append(1.4826 * max(mad_b, float(dados_bias['passo_adc']) * 0.5))
                medianas_b.append(med_b)

        read_noise_mad = float(np.mean(sigmas_b))
        rms_bias = float(np.std(raw_bias))
        mediana_bias = float(np.mean(medianas_b))

        ruido_termico_isolado = float(np.sqrt(max(0.0, sigma_mad**2 - read_noise_mad**2)))
        deriva_pedestal = media_global - float(np.mean(raw_bias))
        tempo_exp = dados_dark['exif'].get('Exposicao_Seg', 0.0)
        taxa_corrente_escuro = (ruido_termico_isolado / tempo_exp) if tempo_exp > 0 else 0.0
        dr_read_stops = float(np.log2(excursao_util / max(1e-5, read_noise_mad)))

        metricas_bias = {
            'arquivo_bias': dados_bias['arquivo'],
            'exposicao_bias': dados_bias['exif'].get('Exposicao_Str', 'N/A'),
            'iso_bias': dados_bias['exif'].get('ISO', 'N/A'),
            'mediana_bias': mediana_bias,
            'read_noise_mad': read_noise_mad,
            'rms_bias': rms_bias,
            'ruido_termico_isolado': ruido_termico_isolado,
            'deriva_pedestal_adu': deriva_pedestal,
            'taxa_corrente_escuro_adu_s': taxa_corrente_escuro,
            'dr_read_stops': dr_read_stops
        }

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
        'limiar_warm_auto': limiar_warm_auto,
        'limiar_dead_auto': limiar_dead_auto,
        'suporta_dead_no_dark': suporta_dead_no_dark,
        'bias_clipado': bias_clipado,
        'bias_cal': metricas_bias
    }

def analisar_flat_frame_profissional(dados_flat, dados_bias=None, limiar_dead_rel=0.50, sigma_outlier_dead=10.0):
    raw_flat = dados_flat['raw_adu']
    wl = dados_flat['white_level']
    bl = dados_bias['bl_medio'] if (dados_bias is not None and dados_bias.get('elegivel') and dados_bias['bl_medio'] > 0) else dados_flat['bl_medio']

    excursao_max = max(1.0, wl - bl)
    mediana_bruta = float(np.median(raw_flat))
    sinal_medio_util = max(1.0, mediana_bruta - bl)
    nivel_exposicao_pct = (sinal_medio_util / excursao_max) * 100.0
    saturados_pct = float(np.sum(raw_flat >= wl * 0.98)) / raw_flat.size * 100.0

    avisos_flat = []
    if nivel_exposicao_pct < 8.0:
        avisos_flat.append(
            f"Aviso Metrológico: O Flat Frame enviado possui baixa exposição útil ({nivel_exposicao_pct:.1f}% da capacidade do conversor ADC). "
            f"Recomenda-se capturar o Flat Frame com exposição entre 30% e 70% do histograma em ISO base."
        )
    if saturados_pct > 1.0:
        avisos_flat.append(
            f"Aviso Metrológico: {saturados_pct:.2f}% dos fotodiodos estão saturados no Flat Frame."
        )

    num_dead_verdadeiro = 0
    num_baixa_sensibilidade = 0
    num_poeira_px = 0
    prnu_liquido_canais = []
    coordenadas_dead_flat_x = []
    coordenadas_dead_flat_y = []

    piso_inatividade_adu = max(12.0, float(dados_flat['passo_adc']) * 4.0)

    for r in range(2):
        for c in range(2):
            sub_liquido = np.maximum(0.0, raw_flat[r::2, c::2].astype(np.float32) - bl)
            vizinhanca_3x3 = np.maximum(1.0, median_filter(sub_liquido, size=3))

            diff_alta_freq = sub_liquido - vizinhanca_3x3
            mad_local = float(np.median(np.abs(diff_alta_freq)))
            sigma_alta_freq = max(1.0, 1.4826 * mad_local)

            mask_dead_true = (
                (vizinhanca_3x3 > 4.0 * piso_inatividade_adu) &
                ((sub_liquido <= piso_inatividade_adu) | (sub_liquido <= 0.05 * vizinhanca_3x3)) &
                (diff_alta_freq <= -sigma_outlier_dead * sigma_alta_freq)
            )
            qtd_d = int(np.sum(mask_dead_true))
            num_dead_verdadeiro += qtd_d

            if qtd_d > 0 and len(coordenadas_dead_flat_x) < 1500:
                yy, xx = np.where(mask_dead_true)
                coordenadas_dead_flat_x.extend((xx * 2 + c)[:1500 - len(coordenadas_dead_flat_x)])
                coordenadas_dead_flat_y.extend((yy * 2 + r)[:1500 - len(coordenadas_dead_flat_y)])

            mask_sub_sensivel = (
                (~mask_dead_true) &
                (sub_liquido <= vizinhanca_3x3 * 0.65) &
                (diff_alta_freq <= -7.0 * sigma_alta_freq)
            )
            num_baixa_sensibilidade += int(np.sum(mask_sub_sensivel))

            sub_sem_ruido = gaussian_filter(vizinhanca_3x3, sigma=2.0)
            perfil_vinhetagem = np.maximum(1.0, gaussian_filter(vizinhanca_3x3, sigma=35.0))
            transmissao_optica = sub_sem_ruido / perfil_vinhetagem

            sigma_trans = float(np.median(np.abs(transmissao_optica - 1.0))) * 1.4826
            limiar_sombra_poeira = min(0.94, 1.0 - max(0.06, 5.0 * sigma_trans))

            mask_poeira_sub = (transmissao_optica < limiar_sombra_poeira) & (~mask_dead_true)
            num_poeira_px += int(np.sum(mask_poeira_sub))

            razao_prnu = diff_alta_freq / vizinhanca_3x3
            sigma_razao_obs = float(np.median(np.abs(razao_prnu))) * 1.4826
            prnu_liquido_canais.append(sigma_razao_obs * 100.0)

            del sub_liquido, vizinhanca_3x3, diff_alta_freq, sub_sem_ruido, perfil_vinhetagem, transmissao_optica

    gc.collect()
    return {
        'arquivo_flat': dados_flat['arquivo'],
        'exposicao_flat': dados_flat['exif'].get('Exposicao_Str', 'N/A'),
        'iso_flat': dados_flat['exif'].get('ISO', 'N/A'),
        'nivel_exposicao_pct': nivel_exposicao_pct,
        'prnu_pct': float(np.mean(prnu_liquido_canais)),
        'num_dead_flat': num_dead_verdadeiro,
        'num_baixa_sens_flat': num_baixa_sensibilidade,
        'pct_poeira': (float(num_poeira_px) / raw_flat.size) * 100.0,
        'coords_dead_x': np.array(coordenadas_dead_flat_x),
        'coords_dead_y': np.array(coordenadas_dead_flat_y),
        'avisos': avisos_flat
    }

@st.cache_data(max_entries=2, show_spinner=False)
def baixar_exemplo_github(nome_arq):
    url = f"https://raw.githubusercontent.com/renato-f-bandeira/files/main/{nome_arq}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=25) as resp:
        return resp.read()

def gerar_imagem_mapa_clicavel(rgb_preview_exp, w_real, h_real, x1, y1, tam_janela, cx_clip, cy_clip, xh_top, yh_top, xd_top, yd_top, marcar_defeitos):
    hp, wp = rgb_preview_exp.shape[:2]
    esc_x = wp / float(w_real)
    esc_y = hp / float(h_real)

    arr_u8 = (np.clip(rgb_preview_exp, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(arr_u8)
    draw = ImageDraw.Draw(pil_img)

    if marcar_defeitos:
        for px, py in zip(xh_top, yh_top):
            ix, iy = int(px * esc_x), int(py * esc_y)
            draw.ellipse([ix - 3, iy - 3, ix + 3, iy + 3], outline=(230, 57, 70), width=1)
        for px, py in zip(xd_top, yd_top):
            ix, iy = int(px * esc_x), int(py * esc_y)
            draw.ellipse([ix - 3, iy - 3, ix + 3, iy + 3], outline=(0, 180, 216), width=1)

    rx1, ry1 = int(x1 * esc_x), int(y1 * esc_y)
    rx2, ry2 = max(rx1 + 6, int((x1 + tam_janela) * esc_x)), max(ry1 + 6, int((y1 + tam_janela) * esc_y))
    draw.rectangle([rx1, ry1, rx2, ry2], outline=(255, 215, 0), width=2)

    icx, icy = int(cx_clip * esc_x), int(cy_clip * esc_y)
    draw.line([icx - 6, icy, icx + 6, icy], fill=(255, 215, 0), width=2)
    draw.line([icx, icy - 6, icx, icy + 6], fill=(255, 215, 0), width=2)

    return pil_img, wp, hp

# ==============================================================================
# 4. INTERFACE TECNICA COM ABAS DE NAVEGACAO (DIAGNOSTICO, COMPARACAO TERMICA E GLOSSARIO)
# ==============================================================================

st.title("Laboratório de Metrologia e Diagnóstico de Sensores")

tab_diag, tab_termico, tab_tutorial = st.tabs([
    "Diagnóstico Principal",
    "Comparação Térmica (Série Temporal)",
    "Guia Didático e Glossário"
] )

with st.sidebar:
    st.header("1. Seleção de Quadros RAW (Pilha de 1 a 3 imgs)")
    tipos_aceitos = ['arw', 'cr2', 'cr3', 'nef', 'nrw', 'dng', 'orf', 'rw2', 'pef', 'raf', 'srw']
    arquivos_dark_up = st.file_uploader("Quadro 1: Dark Frame (Até 3 arquivos)", type=tipos_aceitos, accept_multiple_files=True)
    arquivos_bias_up = st.file_uploader("Quadro 2: Offset / Bias Frame (Até 3 arquivos)", type=tipos_aceitos, accept_multiple_files=True)
    arquivos_flat_up = st.file_uploader("Quadro 3: Flat Frame (Até 3 arquivos)", type=tipos_aceitos, accept_multiple_files=True)

    st.divider()
    st.subheader("Ou usar Banco de Referência")
    exemplo_escolhido = st.selectbox(
        "Amostra oficial do repositório:",
        ["Nenhum (Usar meus arquivos)", "DSC02212.ARW (Sony ILCE-7M3)", "DSC00906.ARW (Sony ILCE-6000)"]
    )

    exigir_catalogo = st.checkbox("Restringir apenas a modelos do Catálogo Homologado", value=False)
    btn_iniciar = st.button("Iniciar Diagnóstico Metrológico", type="primary", use_container_width=True)

if not btn_iniciar and "dados_dark_cache" not in st.session_state:
    st.info(
        "Instruções:\n"
        "1. Selecione na barra lateral até 3 arquivos para cada tipo de quadro (Dark, Bias, Flat) ou escolha uma amostra de referência.\n"
        "2. Clique no botão **Iniciar Diagnóstico Metrológico**.\n"
        "3. Navegue entre as abas para ver o mapa global interativo, a degradação térmica ou o glossário explicativo."
    )
    st.stop()

if btn_iniciar:
    try:
        with st.spinner("Processando e empilhando quadros (Master Frames)..."):
            if arquivos_dark_up:
                st.session_state.dados_dark_cache = processar_pilha_imagens(arquivos_dark_up, exigir_catalogo)
            elif exemplo_escolhido != "Nenhum (Usar meus arquivos)":
                nome_ex = exemplo_escolhido.split(" ")[0]
                bytes_ex = baixar_exemplo_github(nome_ex)
                st.session_state.dados_dark_cache = processar_bytes_imagem(bytes_ex, nome_ex, exigir_catalogo)
            else:
                st.session_state.dados_dark_cache = None

            if arquivos_bias_up:
                st.session_state.dados_bias_cache = processar_pilha_imagens(arquivos_bias_up, exigir_catalogo)
            else:
                st.session_state.dados_bias_cache = None

            if arquivos_flat_up:
                st.session_state.dados_flat_cache = processar_pilha_imagens(arquivos_flat_up, exigir_catalogo)
            else:
                st.session_state.dados_flat_cache = None
    except Exception as erro_proc:
        st.error(f"Erro no processamento dos arquivos: {erro_proc}")
        st.stop()

dados_dark = st.session_state.get("dados_dark_cache", None)
dados_bias = st.session_state.get("dados_bias_cache", None)
dados_flat = st.session_state.get("dados_flat_cache", None)

if dados_dark is None:
    st.warning("Nenhum Dark Frame válido foi carregado. Envie um arquivo Dark Frame na barra lateral e clique em Iniciar.")
    st.stop()

if not dados_dark.get('elegivel', False):
    exif_rej = dados_dark.get('exif', {})
    st.error(
        f"**Modelo de Câmera/Sensor Não Homologado ou Incompatível**\n\n"
        f"- **Equipamento:** `{exif_rej.get('Fabricante', 'Desconhecido')} {exif_rej.get('Modelo', 'Desconhecido')}`\n"
        f"- **Motivo:** {dados_dark.get('motivo_rejeicao')}"
    )
    st.stop()

h, w = dados_dark['raw_adu'].shape
if dados_bias is not None and (not dados_bias.get('elegivel', False) or dados_bias['raw_adu'].shape != (h, w)):
    dados_bias = None
if dados_flat is not None and (not dados_flat.get('elegivel', False) or dados_flat['raw_adu'].shape != (h, w)):
    dados_flat = None

cal = analisar_dark_e_bias(dados_dark, dados_bias)
res_flat = analisar_flat_frame_profissional(dados_flat, dados_bias) if dados_flat is not None else None

raw_adu = dados_dark['raw_adu']
wl_int = int(max(255, dados_dark['white_level']))

# ==============================================================================
# ABA 1: DIAGNOSTICO PRINCIPAL
# ==============================================================================
with tab_diag:
    with st.sidebar:
        st.divider()
        st.header("Controles Visuais (Aba Principal)")
        quadros_disp = ["Dark Frame (Principal)"]
        if dados_flat is not None:
            quadros_disp.append("Flat Frame (Iluminação)")
        if dados_bias is not None:
            quadros_disp.append("Offset / Bias Frame")

        quadro_visualizado = st.selectbox("Quadro exibido no Mapa:", quadros_disp, key="qv_select")
        ev = st.slider("Ganho Visual (EV):", -2.0, 12.0, 1.5, 0.5, key="ev_slider")
        modo_zoom = st.selectbox("Zoom 1:1 Render:", ["Matriz Bayer Bruta 1:1 (ADU)", "Escala Linear Normalizada"], key="zoom_mode_sel")
        tam_janela = st.selectbox("Tamanho Janela Recorte:", [30, 60, 120, 300], index=1, key="win_size_sel")
        marcar_defeitos = st.checkbox("Sobrepor marcadores de defeitos", value=True, key="mark_def_chk")

        limiar_hot = st.slider("Limiar Hot Pixel Severo (ADU):", 0, wl_int, int(cal['limiar_hot_auto']), key="l_hot_sld")
        if cal['suporta_dead_no_dark']:
            max_d_sld = max(50, int(dados_dark['bl_medio'] * 1.2))
            limiar_dead = st.slider("Limiar Dead Pixel no Dark (ADU):", 0, max_d_sld, int(cal['limiar_dead_auto']), key="l_dead_sld")
        else:
            limiar_dead = -1

    st.subheader("Parâmetros de Captura dos Quadros Carregados (EXIF)")
    cols_exif = st.columns(3)
    ex_d = dados_dark['exif']
    with cols_exif[0]:
        st.markdown(f"**1. Dark Frame**\n- ISO: `{ex_d.get('ISO')}` | Exp: `{ex_d.get('Exposicao_Str')} s`\n- Câmera: `{ex_d.get('Fabricante')} {ex_d.get('Modelo')}`")
    with cols_exif[1]:
        if dados_bias is not None:
            ex_b = dados_bias['exif']
            st.markdown(f"**2. Offset / Bias**\n- ISO: `{ex_b.get('ISO')}` | Exp: `{ex_b.get('Exposicao_Str')} s`\n- Câmera: `{ex_b.get('Fabricante')} {ex_b.get('Modelo')}`")
        else:
            st.markdown("**2. Offset / Bias Frame**\n- *Não enviado*")
    with cols_exif[2]:
        if dados_flat is not None:
            ex_f = dados_flat['exif']
            st.markdown(f"**3. Flat Frame**\n- ISO: `{ex_f.get('ISO')}` | Exp: `{ex_f.get('Exposicao_Str')} s`\n- Câmera: `{ex_f.get('Fabricante')} {ex_f.get('Modelo')}`")
        else:
            st.markdown("**3. Flat Frame**\n- *Não enviado*")

    yh_all, xh_all = np.where(raw_adu >= limiar_hot)
    num_hot = len(xh_all)
    if num_hot > 1200:
        vals_hot = raw_adu[yh_all, xh_all]
        idx_top = np.argsort(vals_hot)[::-1][:1200]
        xh_plot, yh_plot = xh_all[idx_top], yh_all[idx_top]
    else:
        xh_plot, yh_plot = xh_all, yh_all

    limiar_warm_efetivo = min(limiar_hot, cal['limiar_warm_auto'])
    num_warm = int(np.sum((raw_adu >= limiar_warm_efetivo) & (raw_adu < limiar_hot)))

    if cal['suporta_dead_no_dark'] and limiar_dead >= 0:
        yd_dark, xd_dark = np.where(raw_adu <= limiar_dead)
        num_dead_dark = len(xd_dark)
        str_dead_dark = f"{num_dead_dark}"
    else:
        yd_dark, xd_dark = np.array([]), np.array([])
        num_dead_dark = 0
        str_dead_dark = "Requer Flat (Bias=0)"

    total_pixels = raw_adu.size

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Equipamento / Obturador", f"{ex_d.get('Fabricante')} {ex_d.get('Modelo')}", ex_d.get('Cliques'))
    col2.metric("Piso de Ruído no Dark (MAD)", f"{cal['sigma_mad']:.2f} ADU", f"RMS Global: {cal['rms_total']:.2f} ADU", delta_color="off")
    col3.metric("Faixa Dinâmica Efetiva (DR)", f"{cal['dr_stops']:.2f} EV", f"{cal['dr_db']:.1f} dB | ADC {dados_dark['bits_efetivos']}-bit real", delta_color="off")

    if res_flat is not None:
        col4.metric("Defeitos Pontuais", f"{num_hot} Hot | {res_flat['num_dead_flat']} Dead", f"Baixa Sensib.: {res_flat['num_baixa_sens_flat']} px", delta_color="off")
    else:
        col4.metric("Anomalias no Dark", f"{num_hot} Hot Severos", f"Warm: {num_warm} | Dead: {str_dead_dark}", delta_color="off")

    if cal['bias_cal'] is not None or res_flat is not None:
        st.divider()
        c_bias, c_flat = st.columns(2)
        if cal['bias_cal'] is not None:
            b = cal['bias_cal']
            with c_bias:
                st.markdown(f"**Desacoplamento de Leitura (Offset / Bias):**")
                st.markdown(f"- **Ruído de Leitura Puro (Read Noise):** `{b['read_noise_mad']:.2f} ADU`\n- **Ruído Térmico Isolado:** `{b['ruido_termico_isolado']:.2f} ADU`\n- **Taxa de Geração Térmica:** `{b['taxa_corrente_escuro_adu_s']:.3f} ADU/s`")
        if res_flat is not None:
            with c_flat:
                st.markdown(f"**Certificação Óptica (Flat Frame):**")
                st.markdown(f"- **Dead Pixels Verdadeiros:** `{res_flat['num_dead_flat']} px`\n- **Pixels de Baixa Sensibilidade (PRNU local):** `{res_flat['num_baixa_sens_flat']} px`\n- **Área de Sombra/Poeira:** `{res_flat['pct_poeira']:.3f}%`")

    st.divider()

    if quadro_visualizado == "Flat Frame (Iluminação)" and dados_flat is not None:
        dados_vis = dados_flat
        cal_vis = analisar_dark_e_bias(dados_flat)
    elif quadro_visualizado == "Offset / Bias Frame" and dados_bias is not None:
        dados_vis = dados_bias
        cal_vis = analisar_dark_e_bias(dados_bias)
    else:
        dados_vis = dados_dark
        cal_vis = cal

    raw_vis = dados_vis['raw_adu']
    bl_vis = dados_vis['bl_medio']
    wl_vis = dados_vis['white_level']
    ganho_v = 2.0 ** ev
    rgb_prev_v = np.clip(dados_vis['rgb_preview'].astype(np.float32) * ganho_v, 0.0, 1.0)

    half_j = tam_janela // 2
    cx_c = int(np.clip(st.session_state.centro_x, half_j, w - half_j))
    cy_c = int(np.clip(st.session_state.centro_y, half_j, h - half_j))
    x1_c, x2_c = cx_c - half_j, cx_c + half_j
    y1_c, y2_c = cy_c - half_j, cy_c + half_j

    xd_map = res_flat['coords_dead_x'] if (res_flat is not None and res_flat['num_dead_flat'] > 0) else xd_dark[:1200]
    yd_map = res_flat['coords_dead_y'] if (res_flat is not None and res_flat['num_dead_flat'] > 0) else yd_dark[:1200]

    if SUPORTA_CLIQUE_IMAGEM:
        st.subheader("Mapa Global Interativo (Clique na imagem para mover a Janela de Zoom)")
        col_m_click, col_i_click = st.columns([1.35, 1])
        with col_m_click:
            pil_m, wp_p, hp_p = gerar_imagem_mapa_clicavel(
                rgb_prev_v, w, h, x1_c, y1_c, tam_janela, cx_c, cy_c, xh_plot, yh_plot, xd_map, yd_map, marcar_defeitos
            )
            c_res = streamlit_image_coordinates(pil_m, key=f"mapa_click_{dados_vis['arquivo']}_{ev}_{limiar_hot}")
            if c_res is not None:
                nx = int(np.clip(round(c_res["x"] / float(wp_p) * w), half_j, w - half_j))
                ny = int(np.clip(round(c_res["y"] / float(hp_p) * h), half_j, h - half_j))
                if nx != st.session_state.centro_x or ny != st.session_state.centro_y:
                    st.session_state.centro_x = nx
                    st.session_state.centro_y = ny
                    st.rerun()
        with col_i_click:
            st.markdown(
                f"**Centro Atual:** `({cx_c}, {cy_c})` | `X[{x1_c}:{x2_c}], Y[{y1_c}:{y2_c}]`\n\n"
                f"- **Leitura no Pixel Central:** `{raw_vis[cy_c, cx_c]} ADU`\n"
                f"- **Mínimo / Máximo no Zoom:** `{np.min(raw_vis[y1_c:y2_c, x1_c:x2_c])} ADU` / `{np.max(raw_vis[y1_c:y2_c, x1_c:x2_c])} ADU`\n"
                f"- **Limiar Hot Severo:** `>= {limiar_hot} ADU` (`{num_hot}` px)"
            )

    rec_adu_v = raw_vis[y1_c:y2_c, x1_c:x2_c].astype(np.float32)

    fig = plt.figure(figsize=(15, 9.8), dpi=115, facecolor='white')
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.25, 1], width_ratios=[1.15, 1], hspace=0.32, wspace=0.22)
    ax_full = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[1, :])

    ax_full.imshow(rgb_prev_v, interpolation='nearest', extent=[0, w, h, 0])
    ax_full.set_title(f"1. Mapa Global - {quadro_visualizado} (+{ev:.1f} EV | {ganho:.1f}x)", fontsize=10.5, fontweight='bold')
    ax_full.set_xlabel("Eixo X (px)", fontsize=9)
    ax_full.set_ylabel("Eixo Y (px)", fontsize=9)
    rect_z = patches.Rectangle((x1_c, y1_c), tam_janela, tam_janela, linewidth=1.8, edgecolor='#FFD700', facecolor='none')
    ax_full.add_patch(rect_z)
    ax_full.plot(cx_c, cy_c, marker='+', color='#FFD700', markersize=9, markeredgewidth=1.5)

    if marcar_defeitos:
        if num_hot > 0:
            ax_full.scatter(xh_plot, yh_plot, s=24, facecolors='none', edgecolors='#E63946', linewidth=1.1, label=f'Hot ({num_hot})')
        if len(xd_map) > 0:
            ax_full.scatter(xd_map, yd_map, s=24, facecolors='none', edgecolors='#00B4D8', linewidth=1.1, label='Defeito / Dead')
    ax_full.legend(loc='upper right', framealpha=0.92, fontsize=8)

    b_ref = bl_vis if bl_vis > 0 else cal_vis['mediana']
    amp_adu = max(30.0, (wl_vis - b_ref) / ganho)
    vmin_v = max(0.0, b_ref - 2.0 * cal_vis['sigma_mad'])
    vmax_v = min(wl_vis, b_ref + amp_adu)

    if "Bayer" in modo_zoom:
        im_zv = ax_zoom.imshow(rec_adu_v, cmap='inferno', vmin=vmin_v, vmax=vmax_v, interpolation='nearest', extent=[x1_c, x2_c, y2_c, y1_c])
        cb_v = fig.colorbar(im_zv, ax=ax_zoom, fraction=0.046, pad=0.04)
        cb_v.set_label("Leitura ADU", fontsize=8.5)
    else:
        rec_norm = np.clip((rec_adu_v - b_ref) / max(1.0, wl_vis - b_ref) * ganho, 0.0, 1.0)
        ax_zoom.imshow(np.stack([rec_norm]*3, axis=-1), interpolation='nearest', extent=[x1_c, x2_c, y2_c, y1_c])

    ax_zoom.set_title(f"2. Zoom 1:1 ({tam_janela}x{tam_janela} px) | Centro ({cx_c}, {cy_c})", fontsize=10.5, fontweight='bold')
    ax_zoom.set_xlabel(f"Min: {np.min(rec_adu_v):.0f} ADU | Max: {np.max(rec_adu_v):.0f} ADU", fontsize=9)
    ax_zoom.set_ylabel("Eixo Y (px)", fontsize=9)

    if tam_janela <= 60:
        ax_zoom.set_xticks(np.arange(x1_c, x2_c + 1, 1), minor=True)
        ax_zoom.set_yticks(np.arange(y1_c, y2_c + 1, 1), minor=True)
        ax_zoom.grid(which='minor', color='white', linestyle='-', linewidth=0.35, alpha=0.35)

    if marcar_defeitos:
        yh_zz, xh_zz = np.where(raw_vis[y1_c:y2_c, x1_c:x2_c] >= limiar_hot)
        if len(xh_zz) > 0:
            ax_zoom.scatter(x1_c + xh_zz + 0.5, y1_c + yh_zz + 0.5, s=90, facecolors='none', edgecolors='#2EC4B6', linewidth=1.6, label='Hot')
        if cal['suporta_dead_no_dark'] and limiar_dead >= 0:
            yd_zz, xd_zz = np.where(raw_vis[y1_c:y2_c, x1_c:x2_c] <= limiar_dead)
            if len(xd_zz) > 0:
                ax_zoom.scatter(x1_c + xd_zz + 0.5, y1_c + yd_zz + 0.5, s=90, facecolors='none', edgecolors='#00B4D8', linewidth=1.6, label='Dead')
        if len(xh_zz) > 0:
            ax_zoom.legend(loc='upper right', framealpha=0.92, fontsize=8)

    cnt_v = np.bincount(raw_vis.ravel())
    vx = np.nonzero(cnt_v)[0]
    vy = cnt_v[vx]
    w_bar = max(1.0, (vx[-1] - vx[0]) / 260.0)
    ax_hist.bar(vx, vy, width=w_bar, color='#1D3557', log=True, alpha=0.88, label='Distribuição ADU')
    ax_hist.axvline(b_ref, color='#2A9D8F', linestyle='-', linewidth=1.6, label=f'Mediana ({b_ref:.0f} ADU)')
    if quadro_visualizado == "Dark Frame (Principal)":
        ax_hist.axvline(limiar_warm_efetivo, color='#F4A261', linestyle=':', linewidth=1.5, label=f'Warm ({limiar_warm_efetivo} ADU)')
        ax_hist.axvline(limiar_hot, color='#E63946', linestyle='--', linewidth=1.6, label=f'Hot Severo ({limiar_hot} ADU)')

    ax_hist.set_title(f"3. Curva ADU | FPN H: {cal_vis['fpn_h']:.3f} ADU | FPN V: {cal_vis['fpn_v']:.3f} ADU", fontsize=10.5, fontweight='bold')
    ax_hist.set_xlabel("Nível Digital do Conversor ADC (ADU)", fontsize=9)
    ax_hist.set_ylabel("Fotodiodos (Escala Log)", fontsize=9)
    ax_hist.legend(loc='upper right', fontsize=8.5)
    ax_hist.grid(True, which="both", ls="--", alpha=0.32)

    buf_img = io.BytesIO()
    fig.savefig(buf_img, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    gc.collect()

    bytes_png = buf_img.getvalue()
    st.image(bytes_png, use_container_width=True)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.download_button("Baixar Laudo Gráfico (PNG)", data=bytes_png, file_name=f"Laudo_{dados_dark['arquivo']}.png", mime="image/png", use_container_width=True)

# ==============================================================================
# ABA 2: COMPARACAO TERMICA (SERIE TEMPORAL)
# ==============================================================================
with tab_termico:
    st.subheader("Análise de Degradação Térmica e Aquecimento do Sensor")
    st.markdown(
        "Compare dois Dark Frames tirados em momentos diferentes da sessão (ex: **Dark 1: Câmera fria** logo ao ligar vs. "
        "**Dark 2: Câmera aquecida** após 30 minutos de uso ou disparo contínuo) para medir o impacto do calor no ruído."
    )

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        arq_dark_frio = st.file_uploader("Dark Frame Frio (Início da Sessão)", type=tipos_aceitos, key="dk_frio")
    with col_t2:
        arq_dark_quente = st.file_uploader("Dark Frame Quente (Final da Sessão)", type=tipos_aceitos, key="dk_quente")

    if arq_dark_frio and arq_dark_quente:
        d_frio = processar_bytes_imagem(arq_dark_frio.getvalue(), arq_dark_frio.name, exigir_catalogo)
        d_quente = processar_bytes_imagem(arq_dark_quente.getvalue(), arq_dark_quente.name, exigir_catalogo)

        if d_frio['elegivel'] and d_quente['elegivel']:
            cal_frio = analisar_dark_e_bias(d_frio)
            cal_quente = analisar_dark_e_bias(d_quente)

            delta_mad = cal_quente['sigma_mad'] - cal_frio['sigma_mad']
            delta_hot = int(np.sum(d_quente['raw_adu'] >= limiar_hot)) - int(np.sum(d_frio['raw_adu'] >= limiar_hot))
            delta_mediana = cal_quente['mediana'] - cal_frio['mediana']

            st.markdown("### Resultados da Comparação Térmica")
            ct1, ct2, ct3 = st.columns(3)
            ct1.metric("Aumento do Ruído de Fundo (MAD)", f"+{delta_mad:.2f} ADU", f"Frio: {cal_frio['sigma_mad']:.2f} -> Quente: {cal_quente['sigma_mad']:.2f}", delta_color="inverse")
            ct2.metric("Proliferação de Hot Pixels", f"+{delta_hot} px", "Novos pixels quentes ativados pelo calor", delta_color="inverse")
            ct3.metric("Deriva de Pedestal (Bias Shift)", f"+{delta_mediana:.1f} ADU", "Elevação do nível de preto por aquecimento do circuito", delta_color="inverse")

            st.success(
                f"**Interpretação Térmica:** O aquecimento do sensor elevou o ruído de fundo em `{delta_mad:.2f} ADU` e ativou `{delta_hot}` novos hot pixels. "
                f"Isso demonstra a importância de utilizar Dark Frames capturados sob a mesma temperatura ambiente e tempo de sessão para calibração adequada."
            )
        else:
            st.error("Um dos arquivos enviados falhou na auditoria de elegibilidade.")
    else:
        st.info("Envie dois Dark Frames (frio e quente) para habilitar o relatório de degradação térmica.")

# ==============================================================================
# ABA 3: GUIA DIDATICO E GLOSSARIO
# ==============================================================================
with tab_tutorial:
    st.subheader("Guia Didático e Dicionário Técnico de Metrologia de Sensores")
    st.markdown(
        "Esta seção explica de forma simplificada o significado de cada métrica exibida nos laudos e orienta "
        "sobre como capturar os quadros em condições ideais."
    )

    st.markdown("### 1. Guia Prático de Captura dos Quadros de Calibração")
    st.markdown(
        "- **Dark Frame (Quadro Escuro):** Coloque a tampa na lente, certifique-se de que nenhum feixe de luz entre pelo visor (tampe o visor se necessário) e dispare com o **mesmo ISO e tempo de exposição** da sua foto real. Sirve para medir o calor e a corrente de escuro.\n"
        "- **Offset / Bias Frame (Leitura Mínima):** Com a lente tampada, dispare na **velocidade de obturação mais rápida possível** da sua câmera (ex: 1/4000s ou 1/8000s) no mesmo ISO. Isola puramente o ruído eletrônico dos amplificadores.\n"
        "- **Flat Frame (Campo Uniforme):** Aponte a câmera para uma superfície branca iluminada de forma homogênea (como uma mesa de luz, a tela de um computador totalmente branca ou o céu limpo no crepúsculo com a lente totalmente desfocada). A exposição deve deixar o histograma entre **30% e 70% da saturação** em ISO base."
    )

    st.markdown("### 2. Dicionário Técnico (Glossário)")
    st.markdown(
        "- **ADU (Analog-to-Digital Unit):** O valor numérico bruto gerado pelo conversor da câmera. Em uma câmera de 14-bits, vai de 0 a 16.383.\n"
        "- **Ruído Robusto (MAD - Median Absolute Deviation):** Mede o 'chiado' de fundo do sensor sem distorção causada por pixels quentes isolados. Quanto menor, mais limpa é a imagem nas sombras.\n"
        "- **Faixa Dinâmica (DR - Dynamic Range):** Mede quantos *stops* de luz (EV) o sensor consegue registrar entre o silêncio do ruído de fundo e a saturação total.\n"
        "- **Banding FPN (Fixed Pattern Noise):** Variação estrutural em forma de listras horizontais ou verticais geradas pelos amplificadores de coluna do chip. Valores abaixo de `0.5 ADU` indicam excelente uniformidade.\n"
        "- **PRNU (Photo-Response Non-Uniformity):** Variação percentual na sensibilidade individual de cada fotodiodo à luz. Em sensores modernos, fica abaixo de `1.5%`."
    )
