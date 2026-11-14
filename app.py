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
from scipy.ndimage import median_filter, gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import streamlit as st

st.set_page_config(
    page_title="Sistema de Metrologia e Diagnóstico de Sensores CMOS/CCD",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1, h2, h3 { font-family: 'Segoe UI', Roboto, Helvetica, sans-serif; font-weight: 600; letter-spacing: -0.01em; }
    div[data-testid="stMetricValue"] { font-size: 1.35rem; font-weight: 600; }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. BASE DE DADOS DE SENSORES HOMOLOGADOS E ESPECIFICACOES FISICAS
# ==============================================================================
# Define os parametros fisicos conhecidos de fabrica para garantir calibracao exata
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

# Inclui .nrw (Nikon Coolpix RAW) e demais extensoes RAW cientificas/comerciais
EXTENSOES_RAW = ('.arw', '.cr2', '.cr3', '.nef', '.nrw', '.dng', '.orf', '.rw2', '.pef', '.raf', '.srw')

# ==============================================================================
# 2. CAMADA DE LEITURA DE HARDWARE E VALIDACAO DE ELEGIBILIDADE METROLOGICA
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
        cliques = "Nao disponivel nos metadados deste modelo"
        for chave in chaves_shutter:
            if chave in dados_num:
                cliques = f"{dados_num[chave]} acionamentos ({chave})"
                break

        fabricante = str(dados_fmt.get('Make', 'Desconhecido')).strip()
        modelo = str(dados_fmt.get('Model', 'Desconhecido')).strip()

        # Busca correspondencia no catalogo homologado
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
            'BlackLevel_EXIF': dados_num.get('BlackLevel', dados_fmt.get('BlackLevel', None)),
            'WhiteLevel_EXIF': dados_num.get('WhiteLevel', None)
        }
    except Exception as e:
        return {'Fabricante': 'Erro', 'Modelo': str(e), 'Homologado_Catalogo': False, 'Spec_Catalogo': None, 'Exposicao_Seg': 0.0, 'Exposicao_Str': 'N/A'}

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

def _reduzir_para_preview(rgb_float, largura_max=1400):
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

@st.cache_data(max_entries=3, show_spinner=False)
def processar_bytes_imagem(file_bytes, nome_original, exigir_catalogo_estrito=False):
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
            # 1a Via: LibRaw (rawpy) - Leitor cientifico primario (suporta .ARW, .NEF, .NRW, .CR2, .CR3, .ORF, .RW2)
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

                        # Aplica fallback do catalogo se a camera estiver homologada e o LibRaw omitir White/Black Level
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
                # 2a Via: TiffFile + ImageCodecs (para containers DNG/TIFF com SubIFDs)
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
                except Exception as erro_tiff:
                    return {
                        'elegivel': False,
                        'motivo_rejeicao': (
                            f"O arquivo '{nome_original}' ({exif.get('Fabricante')} {exif.get('Modelo')}) utiliza uma codificação proprietária "
                            f"que não pôde ser descompactada sem perda metrológica (Detalhe LibRaw: {erro_libraw})."
                        ),
                        'exif': exif
                    }
        else:
            return {
                'elegivel': False,
                'motivo_rejeicao': (
                    f"O arquivo '{nome_original}' não é um arquivo RAW nativo de sensor (extensão '{ext}'). "
                    f"Imagens processadas (JPEG, PNG, TIFF 8-bit) sofrem compressão, curva gama e subtração de pedestal, "
                    f"tornando impossível a calibração confiável de ruído e pixels defeituosos."
                ),
                'exif': exif
            }

        # ==============================================================================
        # PORTAO DE ELEGIBILIDADE E CONFIABILIDADE METROLOGICA DO SENSOR
        # ==============================================================================
        motivos_invalidez = []

        if exigir_catalogo_estrito and not exif.get('Homologado_Catalogo', False):
            motivos_invalidez.append(
                f"O modelo '{exif.get('Fabricante')} {exif.get('Modelo')}' ainda não consta no Catálogo de Sensores Homologados."
            )

        # Verifica se o White Level e Black Level foram obtidos de forma autentica (sem chutes arbitrarios)
        if wl is None or wl <= 255:
            motivos_invalidez.append(
                "Não foi possível extrair o limite de saturação ADC (White Level) autêntico dos metadados nem do catálogo, "
                "ou os dados possuem profundidade insuficiente (8-bits)."
            )

        if bl_canais is None:
            motivos_invalidez.append(
                "O nível de pedestal eletrônico (Black Level por canal CFA) não está presente nos metadados do arquivo "
                "e este sensor ainda não possui parâmetros de fábrica cadastrados no catálogo."
            )

        # Se o sensor nao esta no catalogo homologado e apresenta Black Level = 0 (pedestal cortado no zero),
        # avisa que a calibracao automatica nao e confiavel sem os parametros fisicos do modelo no catalogo
        bl_medio = float(np.mean(bl_canais)) if bl_canais else 0.0
        mediana_obs = float(np.median(raw_adu))
        fracao_zeros = float(np.sum(raw_adu == 0)) / raw_adu.size

        if not exif.get('Homologado_Catalogo', False):
            if bl_medio == 0.0 and fracao_zeros > 0.15:
                motivos_invalidez.append(
                    f"O sensor '{exif.get('Fabricante')} {exif.get('Modelo')}' não está no catálogo homologado e grava arquivos com "
                    f"subtração prévia de nível de preto (Black Level = 0 ADU, com {fracao_zeros*100:.1f}% dos pixels truncados em zero). "
                    f"Sem a curva de calibração específica deste sensor no catálogo, o cálculo de ruído e Dead Pixels seria impreciso."
                )
            if padrao_cfa in ('N/A', 'Desconhecido', '') and metodo_decodificacao != "LibRaw_Linear3D":
                motivos_invalidez.append(
                    f"A matriz de filtro de cor (CFA Pattern) do modelo '{exif.get('Fabricante')} {exif.get('Modelo')}' não pôde ser identificada."
                )

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

# ==============================================================================
# 3. MOTOR DE METROLOGIA: DARK, OFFSET/BIAS E FLAT FRAME (COM SUPORTE A BIAS ZERO)
# ==============================================================================

def analisar_dark_e_bias(dados_dark, dados_bias=None, k_sigma_hot=15.0, k_sigma_warm=8.0):
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
                # Para sensores homologados com pedestal clipado em zero (ex: Nikon P1000 .NRW),
                # estima o sigma robusto a partir da meia-normal direita (percentil 68.27% dos pixels > 0)
                positivos = sub[sub > 0]
                if len(positivos) > 100:
                    q68 = float(np.percentile(positivos, 68.27))
                    sigma_c = max(float(passo_adc), q68)
                else:
                    sigma_c = max(float(passo_adc), float(np.std(sub)))
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

    margem_warm = max(k_sigma_warm * sigma_mad, excursao_util * 0.012)
    limiar_warm_auto = int(min(wl, pedestal_ref + margem_warm))

    margem_hot = max(k_sigma_hot * sigma_mad, excursao_util * 0.035)
    limiar_hot_auto = int(min(wl, pedestal_ref + margem_hot))

    # Se o sensor possui pedestal > 20 ADU, pode detectar Dead Pixel por colapso de Bias no Dark.
    # Se o sensor ja grava Black Level = 0 ADU (como Nikon NRW), avisa que Dead Pixels exigem Flat Frame.
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
            f"Recomenda-se capturar o Flat Frame com exposição entre 30% e 70% do histograma para maximizar a relação sinal-ruído."
        )
    if saturados_pct > 1.0:
        avisos_flat.append(
            f"Aviso Metrológico: {saturados_pct:.2f}% dos fotodiodos estão saturados no Flat Frame."
        )

    num_dead_flat = 0
    num_poeira_px = 0
    prnu_liquido_canais = []
    coordenadas_dead_flat_x = []
    coordenadas_dead_flat_y = []

    for r in range(2):
        for c in range(2):
            sub = np.maximum(1.0, raw_flat[r::2, c::2].astype(np.float32) - bl)
            vizinhanca_3x3 = median_filter(sub, size=3)

            diff_alta_freq = sub - vizinhanca_3x3
            mad_local = float(np.median(np.abs(diff_alta_freq)))
            sigma_alta_freq = max(1.0, 1.4826 * mad_local)

            mask_dead_sub = (sub <= vizinhanca_3x3 * (1.0 - limiar_dead_rel)) & (diff_alta_freq <= -sigma_outlier_dead * sigma_alta_freq)
            qtd_d = int(np.sum(mask_dead_sub))
            num_dead_flat += qtd_d

            if qtd_d > 0 and len(coordenadas_dead_flat_x) < 2000:
                yy, xx = np.where(mask_dead_sub)
                coordenadas_dead_flat_x.extend((xx * 2 + c)[:2000 - len(coordenadas_dead_flat_x)])
                coordenadas_dead_flat_y.extend((yy * 2 + r)[:2000 - len(coordenadas_dead_flat_y)])

            sub_sem_ruido = gaussian_filter(vizinhanca_3x3, sigma=2.0)
            perfil_vinhetagem = np.maximum(1.0, gaussian_filter(vizinhanca_3x3, sigma=35.0))
            transmissao_optica = sub_sem_ruido / perfil_vinhetagem

            sigma_trans = float(np.median(np.abs(transmissao_optica - 1.0))) * 1.4826
            limiar_sombra_poeira = min(0.94, 1.0 - max(0.06, 5.0 * sigma_trans))

            mask_poeira_sub = (transmissao_optica < limiar_sombra_poeira) & (~mask_dead_sub)
            num_poeira_px += int(np.sum(mask_poeira_sub))

            razao_prnu = diff_alta_freq / np.maximum(1.0, vizinhanca_3x3)
            sigma_razao_obs = float(np.median(np.abs(razao_prnu))) * 1.4826
            prnu_liquido_canais.append(sigma_razao_obs * 100.0)

            del sub, vizinhanca_3x3, diff_alta_freq, sub_sem_ruido, perfil_vinhetagem, transmissao_optica

    gc.collect()
    return {
        'arquivo_flat': dados_flat['arquivo'],
        'nivel_exposicao_pct': nivel_exposicao_pct,
        'prnu_pct': float(np.mean(prnu_liquido_canais)),
        'num_dead_flat': num_dead_flat,
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

# ==============================================================================
# 4. INTERFACE TECNICA PROFISSIONAL (STREAMLIT)
# ==============================================================================

st.title("Laboratório de Metrologia e Diagnóstico de Sensores")
st.markdown(
    "Plataforma de análise radiométrica de matrizes CMOS/CCD em leitura bruta (ADU). "
    "Suporta calibração por **Dark Frame** ou certificação completa combinada (**Dark + Offset/Bias + Flat Frame**)."
)

with st.sidebar:
    st.header("1. Entrada de Quadros RAW")
    tipos_aceitos = ['arw', 'cr2', 'cr3', 'nef', 'nrw', 'dng', 'orf', 'rw2', 'pef', 'raf', 'srw']
    arq_dark = st.file_uploader("Quadro 1: Dark Frame (Obrigatório - Lente Tampada)", type=tipos_aceitos)
    arq_bias = st.file_uploader("Quadro 2: Offset / Bias Frame (Opcional - Exp. Mínima)", type=tipos_aceitos)
    arq_flat = st.file_uploader("Quadro 3: Flat Frame (Opcional - Campo Uniforme)", type=tipos_aceitos)

    exigir_catalogo = st.checkbox(
        "Restringir avaliação apenas a modelos do Catálogo Homologado",
        value=False,
        help="Quando marcado, bloqueia qualquer modelo de câmera que não esteja cadastrado explicitamente em CATALOGO_SENSORES_HOMOLOGADOS, mesmo que o arquivo RAW possua metadados completos."
    )

    with st.expander("Modelos Homologados no Catálogo"):
        for mod_k, info_s in CATALOGO_SENSORES_HOMOLOGADOS.items():
            st.caption(f"**{mod_k}**: {info_s['nome_sensor']}")

    st.divider()
    st.subheader("Banco de Referência (GitHub)")
    exemplo_escolhido = st.selectbox(
        "Carregar amostra de referência:",
        ["Nenhum (Usar arquivos enviados)", "DSC02212.ARW (Sony ILCE-7M3)", "DSC00906.ARW (Sony ILCE-6000)"]
    )

dados_dark = None
dados_bias = None
dados_flat = None

try:
    if arq_dark is not None:
        with st.spinner(f"Auditando metadados e matriz de '{arq_dark.name}'..."):
            dados_dark = processar_bytes_imagem(arq_dark.getvalue(), arq_dark.name, exigir_catalogo)
    elif exemplo_escolhido != "Nenhum (Usar arquivos enviados)":
        nome_ex = exemplo_escolhido.split(" ")[0]
        with st.spinner(f"Carregando amostra '{nome_ex}' do repositório..."):
            bytes_ex = baixar_exemplo_github(nome_ex)
            dados_dark = processar_bytes_imagem(bytes_ex, nome_ex, exigir_catalogo)

    if arq_bias is not None:
        with st.spinner(f"Processando Offset/Bias '{arq_bias.name}'..."):
            dados_bias = processar_bytes_imagem(arq_bias.getvalue(), arq_bias.name, exigir_catalogo)

    if arq_flat is not None:
        with st.spinner(f"Processando Flat Frame '{arq_flat.name}'..."):
            dados_flat = processar_bytes_imagem(arq_flat.getvalue(), arq_flat.name, exigir_catalogo)
except Exception as erro_leitura:
    st.error(f"Falha na leitura do arquivo: {erro_leitura}")
    st.stop()

if dados_dark is None:
    st.info("Selecione um arquivo RAW de Dark Frame na barra lateral esquerda (ou escolha uma amostra do banco de referência) para iniciar a avaliação.")
    st.stop()

# Exibe bloqueio formal caso o sensor/camera nao esteja apto ou nao conste no catalogo confiavel
if not dados_dark.get('elegivel', False):
    exif_rej = dados_dark.get('exif', {})
    st.error(
        f"**Modelo de Câmera/Sensor Não Homologado ou Incompatível para Avaliação Confiável**\n\n"
        f"- **Equipamento Identificado:** `{exif_rej.get('Fabricante', 'Desconhecido')} {exif_rej.get('Modelo', 'Desconhecido')}`\n"
        f"- **Diagnóstico de Elegibilidade:** {dados_dark.get('motivo_rejeicao')}\n\n"
        f"Este modelo de sensor/câmera ainda não possui parâmetros de calibração validados no catálogo da plataforma (ou o arquivo não preserva dados brutos lineares suficientes) e, por segurança metrológica, não pode ser avaliado."
    )
    st.stop()

if not dados_dark['exif'].get('Homologado_Catalogo', False):
    st.warning(
        f"**Aviso de Calibração Autônoma:** O modelo `{dados_dark['exif'].get('Fabricante')} {dados_dark['exif'].get('Modelo')}` "
        f"ainda não consta nominalmente no catálogo fixo, mas passou na auditoria física porque seu arquivo RAW forneceu "
        f"diretamente os parâmetros nativos de fábrica (Black Level = `{dados_dark['bl_canais']} ADU`, White Level = `{dados_dark['white_level']:.0f} ADU`, CFA = `{dados_dark['padrao_cfa']}`)."
    )

h, w = dados_dark['raw_adu'].shape
if dados_bias is not None:
    if not dados_bias.get('elegivel', False) or dados_bias['raw_adu'].shape != (h, w):
        st.warning("O arquivo de Offset/Bias enviado é incompatível ou possui resolução diferente do Dark Frame e foi desconsiderado.")
        dados_bias = None
if dados_flat is not None:
    if not dados_flat.get('elegivel', False) or dados_flat['raw_adu'].shape != (h, w):
        st.warning("O arquivo de Flat Frame enviado é incompatível ou possui resolução diferente do Dark Frame e foi desconsiderado.")
        dados_flat = None

cal = analisar_dark_e_bias(dados_dark, dados_bias)
res_flat = analisar_flat_frame_profissional(dados_flat, dados_bias) if dados_flat is not None else None

raw_adu = dados_dark['raw_adu']
wl_int = int(max(255, dados_dark['white_level']))

with st.sidebar:
    st.divider()
    st.header("2. Parâmetros de Inspeção")

    quadros_disponiveis = ["Dark Frame (Principal)"]
    if dados_flat is not None:
        quadros_disponiveis.append("Flat Frame (Iluminação)")
    if dados_bias is not None:
        quadros_disponiveis.append("Offset / Bias Frame")

    quadro_visualizado = st.selectbox("Quadro exibido no Mapa Visual:", quadros_disponiveis)
    ev = st.slider("Ganho Visual de Exposição (EV):", -2.0, 12.0, 4.0, 0.5)
    modo_zoom = st.selectbox("Renderização da Janela de Zoom:", ["Matriz Bayer Bruta 1:1 (ADU)", "Escala Linear Normalizada"])
    tam_janela = st.selectbox("Dimensão do Recorte de Zoom (px):", [30, 60, 120, 300], index=1)
    marcar_defeitos = st.checkbox("Sobrepor marcadores de pixels anômalos", value=True)

    st.subheader("Limiares de Detecção (ADU)")
    limiar_hot = st.slider(
        "Limiar Hot Pixel Severo (ADU):", 0, wl_int, int(cal['limiar_hot_auto']),
        help="Acima de Mediana + 15 Sigmas MAD (fuga térmica estrutural)."
    )
    if cal['suporta_dead_no_dark']:
        max_dead_slider = max(50, int(dados_dark['bl_medio'] * 1.2))
        limiar_dead = st.slider(
            "Limiar Dead Pixel no Dark (ADU):", 0, max_dead_slider, int(cal['limiar_dead_auto']),
            help="Pixels cujo circuito de pedestal entrou em colapso próximo de zero ADU."
        )
    else:
        limiar_dead = -1
        st.caption("Nota Técnica: Este sensor grava arquivos RAW com Black Level = 0 ADU. A detecção de Dead Pixels requer o envio de um Flat Frame.")

    st.subheader("Coordenadas de Inspeção (X, Y)")
    if "zoom_x" not in st.session_state or st.session_state.get("arq_atual") != dados_dark['arquivo']:
        st.session_state.zoom_x = w // 2
        st.session_state.zoom_y = h // 2
        st.session_state.arq_atual = dados_dark['arquivo']

    if st.button("Centralizar no Pixel de Maior Sinal (Hot)"):
        idx_max = int(np.argmax(raw_adu))
        ym, xm = np.unravel_index(idx_max, raw_adu.shape)
        st.session_state.zoom_x = int(xm)
        st.session_state.zoom_y = int(ym)

    cx = st.slider("Coordenada X (px):", 0, w - 1, key="zoom_x")
    cy = st.slider("Coordenada Y (px):", 0, h - 1, key="zoom_y")

yh, xh = np.where(raw_adu >= limiar_hot)
num_hot = len(xh)

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

# --- PAINEL DE INDICADORES PRINCIPAIS ---
exif = dados_dark['exif']
col1, col2, col3, col4 = st.columns(4)
col1.metric("Equipamento / Obturador", f"{exif.get('Fabricante')} {exif.get('Modelo')}", exif.get('Cliques'))
col2.metric("Piso de Ruído no Dark (MAD)", f"{cal['sigma_mad']:.2f} ADU", f"RMS Global: {cal['rms_total']:.2f} ADU", delta_color="off")
col3.metric("Faixa Dinâmica Efetiva (DR)", f"{cal['dr_stops']:.2f} EV", f"{cal['dr_db']:.1f} dB | ADC {dados_dark['bits_efetivos']}-bit real", delta_color="off")

if res_flat is not None:
    col4.metric(
        "Defeitos Pontuais (Hot / Dead)",
        f"{num_hot} Hot | {res_flat['num_dead_flat']} Dead (Flat)",
        f"Warm Pixels: {num_warm} px",
        delta_color="off"
    )
else:
    col4.metric(
        "Anomalias no Dark (Hot / Warm)",
        f"{num_hot} Hot Pixels",
        f"Warm: {num_warm} | Dead: {str_dead_dark}",
        delta_color="off"
    )

# --- SECAO DE ANALISE MULTI-FRAME ---
if cal['bias_cal'] is not None or res_flat is not None:
    st.divider()
    st.subheader("Diagnóstico Radiométrico Multi-Frame (Desacoplamento de Ruído e Defeitos Ópticos)")
    c_bias, c_flat = st.columns(2)

    if cal['bias_cal'] is not None:
        b = cal['bias_cal']
        with c_bias:
            st.markdown(f"**1. Análise de Offset / Bias Frame (`{b['arquivo_bias']}` | Exp: {b['exposicao_bias']} s)**")
            st.markdown(
                f"- **Ruído de Leitura Eletrônico Puro (Read Noise MAD):** `{b['read_noise_mad']:.2f} ADU` (RMS: `{b['rms_bias']:.2f} ADU`)\n"
                f"- **Ruído Térmico Acumulado Isolado (Dark Current Noise):** `{b['ruido_termico_isolado']:.2f} ADU`\n"
                f"- **Taxa de Geração Térmica (Dark Current Rate):** `{b['taxa_corrente_escuro_adu_s']:.3f} ADU/s` (Deriva média: `{b['deriva_pedestal_adu']:+.2f} ADU`)\n"
                f"- **Faixa Dinâmica Base de Leitura (Sem Ruído Térmico):** `{b['dr_read_stops']:.2f} EV`"
            )
    else:
        with c_bias:
            st.info("Offset / Bias Frame não anexado. Envie um quadro capturado na velocidade mais rápida da câmera (ex: 1/4000s com lente tampada) para separar o Ruído de Leitura da Corrente de Escuro.")

    if res_flat is not None:
        with c_flat:
            st.markdown(f"**2. Análise de Flat Frame (`{res_flat['arquivo_flat']}` | Nível de Sinal: {res_flat['nivel_exposicao_pct']:.1f}% do Full Well)**")
            for aviso in res_flat['avisos']:
                st.warning(aviso)
            st.markdown(
                f"- **Dead / Stuck Pixels Verdadeiros (Filtro Mediano 3x3 CFA):** `{res_flat['num_dead_flat']} pixels` (`{res_flat['num_dead_flat']/total_pixels*100:.5f}%` da matriz)\n"
                f"- **Não-Uniformidade de Resposta de Alta Frequência (PRNU):** `{res_flat['prnu_pct']:.2f}%`\n"
                f"- **Fração de Área com Sombras Ópticas (Poeira / Manchas):** `{res_flat['pct_poeira']:.3f}%` do sensor"
            )
    else:
        with c_flat:
            st.info("Flat Frame não anexado. Envie uma foto de superfície uniformemente iluminada (exposição entre 30% e 70% do histograma) para detectar Dead Pixels cegos à luz e manchas de poeira.")

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

ganho = 2.0 ** ev
rgb_preview_exp = np.clip(dados_vis['rgb_preview'].astype(np.float32) * ganho, 0.0, 1.0)

meia_janela = tam_janela // 2
cx_clip = int(np.clip(cx, meia_janela, w - meia_janela))
cy_clip = int(np.clip(cy, meia_janela, h - meia_janela))
x1, x2 = cx_clip - meia_janela, cx_clip + meia_janela
y1, y2 = cy_clip - meia_janela, cy_clip + meia_janela

recorte_adu = raw_vis[y1:y2, x1:x2].astype(np.float32)

# --- PLOTAGEM DOS 3 PAINEIS TECNICOS ---
fig = plt.figure(figsize=(15, 9.8), dpi=115, facecolor='white')
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.25, 1], width_ratios=[1.15, 1], hspace=0.32, wspace=0.22)

ax_full = fig.add_subplot(gs[0, 0])
ax_zoom = fig.add_subplot(gs[0, 1])
ax_hist = fig.add_subplot(gs[1, :])

ax_full.imshow(rgb_preview_exp, interpolation='nearest', extent=[0, w, h, 0])
ax_full.set_title(f"1. Mapa Global - {quadro_visualizado} (+{ev:.1f} EV | {ganho:.1f}x)", fontsize=10.5, fontweight='bold')
ax_full.set_xlabel("Coordenada Horizontal X (px)", fontsize=9)
ax_full.set_ylabel("Coordenada Vertical Y (px)", fontsize=9)

rect = patches.Rectangle((x1, y1), tam_janela, tam_janela, linewidth=1.8, edgecolor='#FFD700', facecolor='none', label='Região de Zoom')
ax_full.add_patch(rect)
ax_full.plot(cx_clip, cy_clip, marker='+', color='#FFD700', markersize=9, markeredgewidth=1.5)

if marcar_defeitos:
    if num_hot > 0:
        ax_full.scatter(xh[:1500], yh[:1500], s=24, facecolors='none', edgecolors='#E63946', linewidth=1.1, label=f'Hot Severo ({num_hot})')
    if res_flat is not None and res_flat['num_dead_flat'] > 0:
        ax_full.scatter(res_flat['coords_dead_x'], res_flat['coords_dead_y'], s=24, facecolors='none', edgecolors='#00B4D8', linewidth=1.1, label=f'Dead Flat ({res_flat["num_dead_flat"]})')
    elif num_dead_dark > 0:
        ax_full.scatter(xd_dark[:1500], yd_dark[:1500], s=24, facecolors='none', edgecolors='#00B4D8', linewidth=1.1, label=f'Dead Dark ({num_dead_dark})')
ax_full.legend(loc='upper right', framealpha=0.92, fontsize=8)

base_ref = bl_vis if bl_vis > 0 else cal_vis['mediana']
amplitude_adu = max(30.0, (wl_vis - base_ref) / ganho)
vmin_adu = max(0.0, base_ref - 4.0 * cal_vis['sigma_mad'])
vmax_adu = min(wl_vis, base_ref + amplitude_adu)

if "Bayer" in modo_zoom:
    im_z = ax_zoom.imshow(recorte_adu, cmap='inferno', vmin=vmin_adu, vmax=vmax_adu, interpolation='nearest', extent=[x1, x2, y2, y1])
    cbar = fig.colorbar(im_z, ax=ax_zoom, fraction=0.046, pad=0.04)
    cbar.set_label("Leitura ADU", fontsize=8.5)
else:
    recorte_norm = np.clip((recorte_adu - base_ref) / max(1.0, wl_vis - base_ref) * ganho, 0.0, 1.0)
    ax_zoom.imshow(np.stack([recorte_norm]*3, axis=-1), interpolation='nearest', extent=[x1, x2, y2, y1])

ax_zoom.set_title(f"2. Inspeção 1:1 ({tam_janela}x{tam_janela} px) | Centro ({cx_clip}, {cy_clip})", fontsize=10.5, fontweight='bold')
ax_zoom.set_xlabel(f"Eixo X (Min: {np.min(recorte_adu):.0f} ADU | Max: {np.max(recorte_adu):.0f} ADU)", fontsize=9)
ax_zoom.set_ylabel("Eixo Y (px)", fontsize=9)

if tam_janela <= 60:
    ax_zoom.set_xticks(np.arange(x1, x2 + 1, 1), minor=True)
    ax_zoom.set_yticks(np.arange(y1, y2 + 1, 1), minor=True)
    ax_zoom.grid(which='minor', color='white', linestyle='-', linewidth=0.35, alpha=0.35)

if marcar_defeitos:
    yh_z, xh_z = np.where(raw_adu[y1:y2, x1:x2] >= limiar_hot)
    if len(xh_z) > 0:
        ax_zoom.scatter(x1 + xh_z + 0.5, y1 + yh_z + 0.5, s=90, facecolors='none', edgecolors='#2EC4B6', linewidth=1.6, label='Hot Pixel')
    if cal['suporta_dead_no_dark'] and limiar_dead >= 0:
        yd_z, xd_z = np.where(raw_adu[y1:y2, x1:x2] <= limiar_dead)
        if len(xd_z) > 0:
            ax_zoom.scatter(x1 + xd_z + 0.5, y1 + yd_z + 0.5, s=90, facecolors='none', edgecolors='#00B4D8', linewidth=1.6, label='Dead Pixel')
    if len(xh_z) > 0:
        ax_zoom.legend(loc='upper right', framealpha=0.92, fontsize=8)

contagens_adu = np.bincount(raw_vis.ravel())
valores_adu_x = np.nonzero(contagens_adu)[0]
contagens_y = contagens_adu[valores_adu_x]

largura_barra = max(1.0, (valores_adu_x[-1] - valores_adu_x[0]) / 260.0)
ax_hist.bar(valores_adu_x, contagens_y, width=largura_barra, color='#1D3557', log=True, alpha=0.88, label=f'Distribuição ADU ({dados_vis["arquivo"]})')
ax_hist.axvline(base_ref, color='#2A9D8F', linestyle='-', linewidth=1.6, label=f'Pedestal / Mediana ({base_ref:.0f} ADU)')

if quadro_visualizado == "Dark Frame (Principal)":
    ax_hist.axvline(limiar_warm_efetivo, color='#F4A261', linestyle=':', linewidth=1.5, label=f'Limiar Warm Pixel ({limiar_warm_efetivo} ADU)')
    ax_hist.axvline(limiar_hot, color='#E63946', linestyle='--', linewidth=1.6, label=f'Limiar Hot Severo ({limiar_hot} ADU)')
    if cal['suporta_dead_no_dark'] and limiar_dead > 0:
        ax_hist.axvline(limiar_dead, color='#00B4D8', linestyle='--', linewidth=1.6, label=f'Limiar Dead ({limiar_dead} ADU)')

ax_hist.set_title(
    f"3. Curva de Distribuição ADU | FPN Horizontal: {cal_vis['fpn_h']:.3f} ADU | FPN Vertical: {cal_vis['fpn_v']:.3f} ADU",
    fontsize=10.5, fontweight='bold'
)
ax_hist.set_xlabel("Nível Digital do Conversor Analógico-Digital (ADU)", fontsize=9)
ax_hist.set_ylabel("Contagem de Fotodiodos (Escala Log)", fontsize=9)
ax_hist.legend(loc='upper right', fontsize=8.5)
ax_hist.grid(True, which="both", ls="--", alpha=0.32)

buf_img = io.BytesIO()
fig.savefig(buf_img, format="png", dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
gc.collect()

bytes_png = buf_img.getvalue()
st.image(bytes_png, use_container_width=True)

status_homologacao = "Homologado no Catálogo Oficial" if exif.get('Homologado_Catalogo') else "Calibração Autônoma via Metadados Nativos"
linhas_relatorio = [
    "=" * 102,
    f"LAUDO METROLOGICO DE SENSOR DE IMAGEM | Quadro Base: {dados_dark['arquivo']} ({dados_dark['tamanho_mb']:.2f} MB) | {dados_dark['tipo_arquivo']}",
    "=" * 102,
    f"- Status de Homologacao       : {status_homologacao}",
    f"- Fabricante / Modelo         : {exif.get('Fabricante')} {exif.get('Modelo')}",
    f"- Arquitetura do Sensor       : {exif.get('Sensor')}",
    f"- Contagem de Obturador       : {exif.get('Cliques')}",
    f"- Parametros de Captura (Dark): ISO {exif.get('ISO')} | Exposicao: {exif.get('Exposicao_Str')} s | Temperatura EXIF: {exif.get('Temp_Sensor')}",
    f"- Matriz / CFA / Quantizacao  : {w} x {h} px ({total_pixels/1e6:.2f} MP) | CFA: {dados_dark['padrao_cfa']} | Container: {dados_dark['bits_container']}-bit (ADC Efetivo: {dados_dark['bits_efetivos']}-bit, Passo: {dados_dark['passo_adc']} ADU)",
    f"- Nivel de Preto (Bias)       : {dados_dark['bl_canais']} ADU (Mediana Real por Subcanal CFA: {cal['medianas_cfa']} ADU | Saturacao: {dados_dark['white_level']:.0f} ADU)",
    "-" * 102,
    "1. METRICAS DO DARK FRAME (RUIDO, FAIXA DINAMICA E DEFEITOS TERMICOS):",
    f"- Ruido de Fundo Robusto (MAD): {cal['sigma_mad']:.2f} ADU (Por Subcanal CFA: {cal['sigmas_cfa']} ADU) | Ruido RMS Global: {cal['rms_total']:.2f} ADU",
    f"- Faixa Dinamica no Dark (DR) : {cal['dr_stops']:.2f} EV (Stops) | {cal['dr_db']:.2f} dB",
    f"- Fixed Pattern Noise (FPN)   : Banding Horizontal = {cal['fpn_h']:.3f} ADU | Banding Vertical = {cal['fpn_v']:.3f} ADU",
    f"- Hot Pixels Severos (Fuga)   : {num_hot} pixels ({num_hot/total_pixels*100:.5f}% da matriz | Limiar >= {limiar_hot} ADU)",
    f"- Warm Pixels (Cauda Termica) : {num_warm} pixels ({num_warm/total_pixels*100:.5f}% da matriz | Entre {limiar_warm_efetivo} e {limiar_hot} ADU)",
    f"- Dead Pixels em Escuro       : {str_dead_dark}"
]

if cal['bias_cal'] is not None:
    b = cal['bias_cal']
    linhas_relatorio.extend([
        "-" * 102,
        f"2. DESACOPLAMENTO ELETRONICO VIA OFFSET / BIAS FRAME ({b['arquivo_bias']} | Exp: {b['exposicao_bias']} s):",
        f"- Ruido de Leitura Puro (MAD) : {b['read_noise_mad']:.2f} ADU (RMS do Bias: {b['rms_bias']:.2f} ADU | Mediana: {b['mediana_bias']:.1f} ADU)",
        f"- Ruido Termico Isolado       : {b['ruido_termico_isolado']:.2f} ADU (Calculado por quadratura entre Dark e Bias)",
        f"- Taxa de Corrente de Escuro  : {b['taxa_corrente_escuro_adu_s']:.4f} ADU/s | Deriva de Pedestal: {b['deriva_pedestal_adu']:+.2f} ADU",
        f"- Faixa Dinamica de Leitura   : {b['dr_read_stops']:.2f} EV"
    ])

if res_flat is not None:
    linhas_relatorio.extend([
        "-" * 102,
        f"3. CERTIFICACAO OPTICA VIA FLAT FRAME ({res_flat['arquivo_flat']} | Exposicao Util: {res_flat['nivel_exposicao_pct']:.1f}% do Full Well):",
        f"- Dead / Stuck Pixels Reais   : {res_flat['num_dead_flat']} pixels ({res_flat['num_dead_flat']/total_pixels*100:.5f}% | Isolados via Filtro Mediano 3x3 intra-canal CFA)",
        f"- PRNU de Alta Frequencia     : {res_flat['prnu_pct']:.2f}%",
        f"- Area com Sombra de Poeira   : {res_flat['pct_poeira']:.3f}% da area util do sensor"
    ])

linhas_relatorio.append("=" * 102)
texto_relatorio_final = "\n".join(linhas_relatorio)

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    st.download_button(
        label="Exportar Laudo Gráfico (PNG)",
        data=bytes_png,
        file_name=f"Laudo_Grafico_{os.path.splitext(dados_dark['arquivo'])[0]}.png",
        mime="image/png",
        use_container_width=True
    )
with col_btn2:
    st.download_button(
        label="Exportar Relatório Técnico (TXT)",
        data=texto_relatorio_final.encode('utf-8'),
        file_name=f"Laudo_Tecnico_{os.path.splitext(dados_dark['arquivo'])[0]}.txt",
        mime="text/plain",
        use_container_width=True
    )

with st.expander("Relatório Técnico Completo (Texto Formatado)", expanded=True):
    st.code(texto_relatorio_final, language="text")
