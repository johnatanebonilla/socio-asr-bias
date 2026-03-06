import os
import re
import string
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

import pandas as pd
import numpy as np
import jiwer
from tqdm.auto import tqdm

# --- Configuration ---
DATA_DIR = Path("/core_dataset/whisper_age/coser_dataset")
ORIG_TRANSCRIPTS_DIR = DATA_DIR / "orig_transcripts_structured"
WHISPER_TRANSCRIPTS_DIR = DATA_DIR / "whisper_transcripts"
METADATA_FILE = DATA_DIR / "coser_metadata_geocoded.xlsx"
OUTPUT_DIR = DATA_DIR / "08_wer_results_speaker"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "08_calculate_wer_speaker_unaligned.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Text Normalization Rules ---
def normalize_text_clean(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    translator = str.maketrans('', '', string.punctuation + '¿¡')
    text = text.translate(translator)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_text_ortho(text):
    if not text: return ""
    text = str(text)
    translator = str.maketrans('', '', '¿¡')
    text = text.translate(translator)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

transform_ortho = jiwer.Compose([
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords()
])

def map_speakers_to_words(ref_segments, cleaner_func):
    """Returns a list of words, and a parallel list of speaker tags"""
    words = []
    speakers = []
    last_i = "I1"  
    
    for seg in ref_segments:
        text = cleaner_func(seg['text'])
        spk = str(seg['speaker']).strip()
        
        if spk.startswith('I') and spk != 'IE' and not spk.startswith('II'):
            last_i = spk
            
        if spk.startswith('IE') or spk.startswith('II'):
            eff_spk = 'U' # Excluded per NLP+CSS scope guidelines to avoid unfair misattribution
        elif not spk or spk.lower() == 'nan':
            eff_spk = 'U'
        else:
            eff_spk = spk
            
        # Tokenize roughly same as jiwer usually does (split by space)
        seg_words = text.split()
        for w in seg_words:
            words.append(w)
            speakers.append(eff_spk)
            
    return words, speakers

def calc_alignment_metrics(ref_words, ref_speakers, hyp_text, is_ortho=False):
    ref_text = " ".join(ref_words)
    res = {spk: {'N': 0, 'S': 0, 'D': 0, 'I': 0} for spk in set(ref_speakers)}
    for spk in ref_speakers:
        res[spk]['N'] += 1

    if not ref_text.strip(): return res

    # Edge case: totally blank hypothesis = 100% deletion
    if not hyp_text.strip():
         for spk, stats in res.items():
             res[spk]['D'] = stats['N']
         return res
         
    try:
        if is_ortho:
            alignments = jiwer.process_words(ref_text, hyp_text, reference_transform=transform_ortho, hypothesis_transform=transform_ortho)
        else:
            alignments = jiwer.process_words(ref_text, hyp_text)
    except Exception as e:
        logger.warning(f"Error computing Jiwer char alignment: {e}")
        return res
        
    for op_chunk in alignments.alignments[0]:
        op = op_chunk.type
        rs = op_chunk.ref_start_idx
        re = op_chunk.ref_end_idx
        
        if op == 'equal': continue
            
        if op in ('substitute', 'delete'):
            for idx in range(rs, re):
                if idx < len(ref_speakers):
                    spk = ref_speakers[idx]
                    if op == 'substitute': res[spk]['S'] += 1
                    elif op == 'delete': res[spk]['D'] += 1
        elif op == 'insert':
            # Attribution: nearest preceding reference word (per Gold Standard instructions)
            target_idx = max(0, rs - 1)
            if target_idx >= len(ref_speakers) and len(ref_speakers) > 0:
                target_idx = len(ref_speakers) - 1
            if target_idx < 0: continue
            spk = ref_speakers[target_idx]
            ins_len = op_chunk.hyp_end_idx - op_chunk.hyp_start_idx
            res[spk]['I'] += ins_len
            
    return res

def calculate_speaker_wer_detailed(ref_segments: List[Dict], hyp_text_raw: str) -> Dict[str, Dict]:
    # 1. Clean
    ref_words_clean, ref_speakers_clean = map_speakers_to_words(ref_segments, normalize_text_clean)
    hyp_clean = normalize_text_clean(hyp_text_raw)
    clean_metrics = calc_alignment_metrics(ref_words_clean, ref_speakers_clean, hyp_clean, is_ortho=False)
    
    # 2. Ortho
    ref_words_ortho, ref_speakers_ortho = map_speakers_to_words(ref_segments, normalize_text_ortho)
    hyp_ortho = normalize_text_ortho(hyp_text_raw)
    ortho_metrics = calc_alignment_metrics(ref_words_ortho, ref_speakers_ortho, hyp_ortho, is_ortho=True)
    
    # Combine
    all_spks = set(list(clean_metrics.keys()) + list(ortho_metrics.keys()))
    res = {}
    for spk in all_spks:
        if spk == 'U': continue
        c_stats = clean_metrics.get(spk, {'N': 0, 'S': 0, 'D': 0, 'I': 0})
        o_stats = ortho_metrics.get(spk, {'N': 0, 'S': 0, 'D': 0, 'I': 0})
        
        n_c, n_o = c_stats['N'], o_stats['N']
        if n_c > 0 or n_o > 0:
            err_c = c_stats['S'] + c_stats['D'] + c_stats['I']
            err_o = o_stats['S'] + o_stats['D'] + o_stats['I']
            res[spk] = {
                'N_clean': n_c, 
                'wer_clean': err_c / n_c if n_c > 0 else 0, 
                'S_clean': c_stats['S'], 
                'D_clean': c_stats['D'], 
                'I_clean': c_stats['I'],
                'N_ortho': n_o,
                'wer_ortho': err_o / n_o if n_o > 0 else 0,
                'S_ortho': o_stats['S'],
                'D_ortho': o_stats['D'],
                'I_ortho': o_stats['I']
            }
    return res

def main():
    logger.info("Initializing Speaker-Level WER script for large-v3/original.")
    refs_structured = load_original_transcripts()
    metadata = load_metadata()
    if metadata.empty: metadata_map = {}
    else: metadata_map = metadata.set_index('id_base').to_dict('index')

    results = []
    model, method = "large-v3", "original"
    method_path = WHISPER_TRANSCRIPTS_DIR / model / method
    if not method_path.exists(): return
        
    hyp_files = list(method_path.glob("*.txt"))
    logger.info(f"Found {len(hyp_files)} transcriptions to evaluate.")
    
    for hyp_file in tqdm(hyp_files, desc=f"{model}/{method}"):
        audio_id = hyp_file.stem  
        base_audio_id = "-".join(audio_id.split("-")[:3]) 
        
        if base_audio_id not in refs_structured: continue
        ref_segments = refs_structured[base_audio_id]
        meta_row = metadata_map.get(base_audio_id, {})
        
        try:
            with open(hyp_file, 'r', encoding='utf-8') as f:
                hyp_text_raw = f.read().strip()
        except: continue
        
        speaker_wers = calculate_speaker_wer_detailed(ref_segments, hyp_text_raw)
        
        for spk, stats in speaker_wers.items():
            role = "Informant" if spk.startswith('I') else "Interviewer" if spk.startswith('E') else "Other"
            row = {
                "id": base_audio_id,
                "model": model,
                "method": method,
                "speaker": spk,
                "role": role,
                "N_clean": stats['N_clean'],
                "S_clean": stats['S_clean'],
                "D_clean": stats['D_clean'],
                "I_clean": stats['I_clean'],
                "wer_clean": stats['wer_clean'],
                "N_ortho": stats['N_ortho'],
                "S_ortho": stats['S_ortho'],
                "D_ortho": stats['D_ortho'],
                "I_ortho": stats['I_ortho'],
                "wer_ortho": stats['wer_ortho'],
                "speaker_sex": np.nan, "speaker_age": np.nan,
                "province": meta_row.get("province", np.nan),
                "Enclave": meta_row.get("Enclave", np.nan),
                "year": np.nan,
            }
            
            # Extract year from date
            date_val = meta_row.get("date")
            if pd.notna(date_val):
                date_str = str(date_val)
                # Naive 4-digit extraction
                m_year = re.search(r'(\d{4})', date_str)
                if m_year:
                    row['year'] = int(m_year.group(1))

            if spk.startswith('I'):
                m = re.match(r'I(\d+)', spk)
                if m:
                    idx = m.group(1)
                    row['speaker_sex'] = meta_row.get(f'Inf{idx}_Sexo', np.nan)
                    row['speaker_age'] = meta_row.get(f'Inf{idx}_Edad', np.nan)
            results.append(row)
            
    if not results: return
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "01_speaker_granular.csv", index=False)
    
    # Validation Macro
    df_audio = df.groupby("id").agg(
        errs_clean=("S_clean", lambda x: df.loc[x.index, "S_clean"].sum() + df.loc[x.index, "D_clean"].sum() + df.loc[x.index, "I_clean"].sum()),
        N=("N_clean", "sum")
    ).reset_index()
    df_audio = df_audio[df_audio["N"] > 0]
    macro_norm_recalc = (df_audio["errs_clean"] / df_audio["N"]).mean()
    logger.info(f"VALIDATION GLOBAL MACRO_NORM: {macro_norm_recalc:.4f}")

    def calc_group_metric_full(subset):
        _empty = pd.Series({
            "macro_norm": np.nan, "micro_norm": np.nan,
            "macro_orth": np.nan, "micro_orth": np.nan,
            "macro_norm_E": np.nan, "macro_norm_I": np.nan,
            "audio_count": 0, "speaker_count": 0
        })
        if subset.empty: return _empty

        # Pre-filter per role — guard for case where subset already pre-filtered (no 'role' column)
        if "role" in subset.columns:
            subset_E = subset[subset["role"] == "Interviewer"]
            subset_I = subset[subset["role"] == "Informant"]
        else:
            subset_E = pd.DataFrame()
            subset_I = pd.DataFrame()

        def audio_totals(sub):
            return sub.groupby("id").agg(
                err_c=("S_clean", "sum"),
                n_c=("N_clean", "sum"),
                err_o=("S_ortho", "sum"),
                n_o=("N_ortho", "sum"),
                err_c_D=("D_clean", "sum"),
                err_c_I=("I_clean", "sum"),
                err_o_D=("D_ortho", "sum"),
                err_o_I=("I_ortho", "sum"),
            ).reset_index().assign(
                err_c=lambda x: x["err_c"] + x["err_c_D"] + x["err_c_I"],
                err_o=lambda x: x["err_o"] + x["err_o_D"] + x["err_o_I"],
            )

        df_a = audio_totals(subset)
        df_a_clean = df_a[df_a["n_c"] > 0]
        macro_norm = (df_a_clean["err_c"] / df_a_clean["n_c"]).mean() if not df_a_clean.empty else np.nan
        micro_norm = df_a_clean["err_c"].sum() / df_a_clean["n_c"].sum() if not df_a_clean.empty and df_a_clean["n_c"].sum() > 0 else np.nan

        df_a_ortho = df_a[df_a["n_o"] > 0]
        macro_orth = (df_a_ortho["err_o"] / df_a_ortho["n_o"]).mean() if not df_a_ortho.empty else np.nan
        micro_orth = df_a_ortho["err_o"].sum() / df_a_ortho["n_o"].sum() if not df_a_ortho.empty and df_a_ortho["n_o"].sum() > 0 else np.nan

        # E-only and I-only per audio
        macro_norm_E = np.nan
        if not subset_E.empty:
            df_E = audio_totals(subset_E)
            df_E = df_E[df_E["n_c"] > 0]
            macro_norm_E = (df_E["err_c"] / df_E["n_c"]).mean() if not df_E.empty else np.nan

        macro_norm_I = np.nan
        if not subset_I.empty:
            df_I = audio_totals(subset_I)
            df_I = df_I[df_I["n_c"] > 0]
            macro_norm_I = (df_I["err_c"] / df_I["n_c"]).mean() if not df_I.empty else np.nan

        unique_audios = subset["id"].nunique()
        speaker_count = subset["speaker"].count()

        return pd.Series({
            "macro_norm": macro_norm,
            "micro_norm": micro_norm,
            "macro_orth": macro_orth,
            "micro_orth": micro_orth,
            "macro_norm_E": macro_norm_E,
            "macro_norm_I": macro_norm_I,
            "audio_count": unique_audios,
            "speaker_count": speaker_count
        })

    # Summaries
    df_informants = df[df["role"] == "Informant"]
    if not df_informants.empty and "speaker_sex" in df.columns:
        df_sex = df_informants.dropna(subset=['speaker_sex']).groupby(['speaker_sex']).apply(calc_group_metric_full, include_groups=False).reset_index()
        df_sex.to_csv(OUTPUT_DIR / "02_summary_by_informant_sex.csv", index=False)
        
        df_age = df_informants.dropna(subset=['speaker_age']).copy()
        bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120]
        labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', '100+']
        df_age['age_range'] = pd.cut(pd.to_numeric(df_age['speaker_age'], errors='coerce'), bins=bins, labels=labels, right=True)
        df_age_summary = df_age.dropna(subset=['age_range']).groupby(['age_range']).apply(calc_group_metric_full, include_groups=False).reset_index()
        df_age_summary.to_csv(OUTPUT_DIR / "03_summary_by_informant_age_ranges.csv", index=False)
        
        df_exact_age = df_informants.dropna(subset=['speaker_age']).groupby(['speaker_age']).apply(calc_group_metric_full, include_groups=False).reset_index()
        df_exact_age.to_csv(OUTPUT_DIR / "06_summary_by_informant_exact_age.csv", index=False)
        
    df_role = df.groupby(['role']).apply(calc_group_metric_full, include_groups=False).reset_index()
    df_role.to_csv(OUTPUT_DIR / "04_summary_E_vs_I.csv", index=False)
    
    df_global = df.groupby(['model', 'method']).apply(calc_group_metric_full, include_groups=False).reset_index()
    df_global.to_csv(OUTPUT_DIR / "05_global_summary_speaker.csv", index=False)

    df_interviewers = df[df["role"] == "Interviewer"]
    if not df_interviewers.empty and "province" in df.columns:
        df_E_prov = df_interviewers.dropna(subset=['province']).groupby(['province']).apply(calc_group_metric_full, include_groups=False).reset_index()
        df_E_prov.to_csv(OUTPUT_DIR / "07_summary_E_by_Province.csv", index=False)
        
        df_E_enc = df_interviewers.dropna(subset=['Enclave']).groupby(['Enclave']).apply(calc_group_metric_full, include_groups=False).reset_index()
        df_E_enc.to_csv(OUTPUT_DIR / "09_summary_E_by_Enclave.csv", index=False)

    if not df_informants.empty and "province" in df.columns:
        df_I_prov = df_informants.dropna(subset=['province']).groupby(['province']).apply(calc_group_metric_full, include_groups=False).reset_index()
        df_I_prov.to_csv(OUTPUT_DIR / "08_summary_I_by_Province.csv", index=False)
        
        df_I_enc = df_informants.dropna(subset=['Enclave']).groupby(['Enclave']).apply(calc_group_metric_full, include_groups=False).reset_index()
        df_I_enc.to_csv(OUTPUT_DIR / "10_summary_I_by_Enclave.csv", index=False)
        
    if "year" in df.columns:
        df_year = df.dropna(subset=['year']).groupby(['year']).apply(calc_group_metric_full, include_groups=False).reset_index()
        df_year.to_csv(OUTPUT_DIR / "11_summary_by_interview_year.csv", index=False)
    
    logger.info("Done generating speaker summary CSVs.")

def load_original_transcripts() -> Dict[str, List[Dict]]:
    refs = {}
    for filepath in ORIG_TRANSCRIPTS_DIR.glob("*.jsonl"):
        audio_id = filepath.stem
        segments = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try: segments.append(json.loads(line))
                    except: pass
        refs[audio_id] = segments
    return refs

def load_metadata() -> pd.DataFrame:
    try:
        df = pd.read_excel(METADATA_FILE)
        df['id_base'] = df['id'].str.extract(r'(COSER-\d{4}-\d{2})', expand=False)
        return df
    except: return pd.DataFrame()

if __name__ == "__main__":
    main()
