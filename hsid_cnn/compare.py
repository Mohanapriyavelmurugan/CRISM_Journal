"""
compare.py  —  Comparison between the CRISM pipeline, HSID-CNN, and N2N4M
"""

import os, sys, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_rel, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hsid_cnn.metrics import compute_sam, compute_snr
from hsid_cnn.config  import FIGURES_DIR

def compute_pearson(cube1, cube2):
    flat1 = cube1.flatten()
    flat2 = cube2.flatten()
    ok = np.isfinite(flat1) & np.isfinite(flat2)
    if ok.sum() < 2: return np.nan
    return pearsonr(flat1[ok], flat2[ok])[0]

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    if pooled_std == 0: return 0
    return (np.mean(x) - np.mean(y)) / pooled_std

def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def compare_pipelines(physics_dir: str, ml_dir: str, hsid_out_dir: str, n2n4m_out_dir: str = "results/n2n4m_output"):
    s3_files = sorted(list(set(
        glob.glob(os.path.join(physics_dir, '**', '*_S3.npy'), recursive=True) +
        glob.glob(os.path.join(physics_dir, '*_S3.npy'))
    )))
    
    results = []
    
    n2n_profiling_file = os.path.join(n2n4m_out_dir, "n2n4m_profiling.json")
    n2n_stats = {}
    if os.path.exists(n2n_profiling_file):
        with open(n2n_profiling_file, 'r') as f:
            n2n_stats = json.load(f)
            
    stage4_json = os.path.join(ml_dir, 'stage4_validation.json')
    s4_stats = {}
    if os.path.exists(stage4_json):
        with open(stage4_json, 'r') as f:
            s4_stats = json.load(f)

    l_scenes, s_scenes, total_scenes = 0, 0, 0
    
    for s3_path in s3_files:
        stem = os.path.basename(s3_path).replace('_S3.npy', '')
        s4_path = os.path.join(ml_dir, f'{stem}_S4.npy')
        hsid_path = os.path.join(hsid_out_dir, f'{stem}_HSID.npy')
        n2n_path = os.path.join(n2n4m_out_dir, f'{stem}_N2N4M.npy')
        
        if not os.path.exists(s4_path) or not os.path.exists(hsid_path):
            continue
            
        noisy = np.load(s3_path).astype(np.float32)
        pipe_clean = np.load(s4_path).astype(np.float32)
        hsid_clean = np.load(hsid_path).astype(np.float32)
        
        total_scenes += 1
        is_l_sensor = noisy.shape[-1] > 200
        if is_l_sensor: l_scenes += 1
        else: s_scenes += 1

        snr_noisy = compute_snr(noisy)
        snr_pipe = compute_snr(pipe_clean)
        sam_pipe = compute_sam(noisy, pipe_clean)
        corr_pipe = compute_pearson(noisy, pipe_clean)
        
        snr_hsid = compute_snr(hsid_clean)
        sam_hsid = compute_sam(noisy, hsid_clean)
        corr_hsid = compute_pearson(noisy, hsid_clean)
        
        # N2N4M
        snr_n2n, sam_n2n, corr_n2n = np.nan, np.nan, np.nan
        n2n_rt_val, n2n_mem_val = np.nan, np.nan
        
        if is_l_sensor and os.path.exists(n2n_path):
            n2n_clean = np.load(n2n_path).astype(np.float32)
            snr_n2n = compute_snr(n2n_clean)
            sam_n2n = compute_sam(noisy, n2n_clean)
            corr_n2n = compute_pearson(noisy, n2n_clean)
            if stem in n2n_stats:
                n2n_rt_val = float(n2n_stats[stem].get('runtime_seconds', np.nan))
                n2n_mem_val = float(n2n_stats[stem].get('peak_memory_mb', np.nan))
        
        hsid_stats_path = hsid_path.replace('.npy', '_stats.json')
        hsid_rt_val, hsid_mem_val = np.nan, np.nan
        if os.path.exists(hsid_stats_path):
            with open(hsid_stats_path, 'r') as f:
                stats = json.load(f)
                hsid_rt_val = float(stats.get('runtime_seconds', np.nan))
                hsid_mem_val = float(stats.get('peak_memory_mb', np.nan))

        pipe_rt_val, pipe_mem_val = np.nan, np.nan
        if stem in s4_stats:
            pipe_rt_val = float(s4_stats[stem].get('runtime_seconds', np.nan))
            pipe_mem_val = float(s4_stats[stem].get('peak_memory_mb', np.nan))

        results.append({
            'Cube': stem,
            'Noisy_SNR': snr_noisy,
            'Pipe_SNR': snr_pipe, 'Pipe_SAM': sam_pipe, 'Pipe_Corr': corr_pipe,
            'HSID_SNR': snr_hsid, 'HSID_SAM': sam_hsid, 'HSID_Corr': corr_hsid,
            'N2N_SNR': snr_n2n, 'N2N_SAM': sam_n2n, 'N2N_Corr': corr_n2n,
            'Pipe_Runtime_s': pipe_rt_val, 'Pipe_Mem_MB': pipe_mem_val,
            'HSID_Runtime_s': hsid_rt_val, 'HSID_Mem_MB': hsid_mem_val,
            'N2N_Runtime_s': n2n_rt_val, 'N2N_Mem_MB': n2n_mem_val
        })
        
        # Per-cube Comparison Bar Chart
        plt.figure(figsize=(8, 5))
        labels = ['Original Pipeline', 'HSID-CNN', 'N2N4M']
        snrs = [snr_pipe, snr_hsid, snr_n2n]
        if np.isnan(snr_n2n):
            labels = ['Original Pipeline', 'HSID-CNN']
            snrs = [snr_pipe, snr_hsid]
            plt.bar(labels, snrs, color=['#1f77b4', '#ff7f0e'])
        else:
            plt.bar(labels, snrs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            
        plt.axhline(y=snr_noisy, color='r', linestyle='--', label='Noisy Input')
        plt.ylabel('Signal-to-Noise Ratio (SNR)')
        plt.title(f'{stem} - Denoising Performance')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'{stem}_snr_comparison.png'), dpi=600)
        plt.close()

    if not results:
        return
        
    df = pd.DataFrame(results)
    tab_dir = os.path.join(os.path.dirname(FIGURES_DIR), "tables")
    rep_dir = os.path.join(os.path.dirname(FIGURES_DIR), "reports")
    os.makedirs(tab_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)
    
    df.to_csv(os.path.join(tab_dir, "per_cube_metrics.csv"), index=False)
    
    # Write coverage report
    with open(os.path.join(rep_dir, "coverage_report.md"), 'w') as f:
        f.write("# Dataset Coverage Report\n\n")
        f.write("| Method | L-scenes | S-scenes | Total Evaluated |\n")
        f.write("|--------|----------|----------|-----------------|\n")
        f.write(f"| Proposed Pipeline | {l_scenes} | {s_scenes} | {total_scenes} |\n")
        f.write(f"| HSID-CNN | {l_scenes} | {s_scenes} | {total_scenes} |\n")
        f.write(f"| N2N4M | {l_scenes} | 0 | {l_scenes} |\n")

    # Generate Boxplots
    def boxplot_metric(m_pipe, m_hsid, m_n2n, title, ylabel, savename):
        plt.figure(figsize=(6, 5))
        
        # N2N4M only has L-sensor scenes, so filter NaNs
        n2n_valid = df[m_n2n].dropna().values
        bp = plt.boxplot([df[m_pipe].values, df[m_hsid].values, n2n_valid], 
                         tick_labels=['Proposed Pipeline', 'HSID-CNN', 'N2N4M'],
                         patch_artist=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in bp['medians']:
            median.set(color='black', linewidth=1.5)
            
        plt.title(title, fontsize=12)
        plt.ylabel(ylabel, fontsize=10)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, savename), dpi=600)
        plt.close()

    boxplot_metric('Pipe_SNR', 'HSID_SNR', 'N2N_SNR', 'SNR Comparison', 'SNR', 'boxplot_snr.png')
    boxplot_metric('Pipe_SAM', 'HSID_SAM', 'N2N_SAM', 'SAM Comparison', 'SAM (rad)', 'boxplot_sam.png')
    boxplot_metric('Pipe_Corr', 'HSID_Corr', 'N2N_Corr', 'Pearson Correlation Comparison', 'Pearson r', 'boxplot_pearson.png')
    boxplot_metric('Pipe_Runtime_s', 'HSID_Runtime_s', 'N2N_Runtime_s', 'Runtime Comparison', 'Time (s)', 'boxplot_runtime.png')
    boxplot_metric('Pipe_Mem_MB', 'HSID_Mem_MB', 'N2N_Mem_MB', 'Peak Memory Comparison', 'Memory (MB)', 'boxplot_memory.png')

    # Aggregate & Statistical Analysis
    with open(os.path.join(tab_dir, "statistical_analysis.txt"), "w") as f:
        f.write("STATISTICAL ANALYSIS REPORT\n")
        f.write("===========================\n\n")
        
        for metric_name, (m_p, m_h, m_n) in [
            ('SNR', ('Pipe_SNR', 'HSID_SNR', 'N2N_SNR')),
            ('SAM', ('Pipe_SAM', 'HSID_SAM', 'N2N_SAM')),
            ('Pearson', ('Pipe_Corr', 'HSID_Corr', 'N2N_Corr'))
        ]:
            f.write(f"--- {metric_name} ---\n")
            
            # Proposed vs HSID (Full Set)
            valid_ph = df[[m_p, m_h]].dropna()
            vp, vh = valid_ph[m_p].values, valid_ph[m_h].values
            t_st, t_p = ttest_rel(vp, vh)
            w_st, w_p = wilcoxon(vp, vh)
            cd = cohen_d(vp, vh)
            f.write(f"Proposed vs HSID-CNN (n={len(vp)}):\n")
            f.write(f"  Paired t-test: p = {t_p:.4e}\n")
            f.write(f"  Wilcoxon:      p = {w_p:.4e}\n")
            f.write(f"  Cohen's d:     {cd:.3f}\n\n")
            
            # Proposed vs N2N4M (L-sensor subset only)
            valid_pn = df[[m_p, m_n]].dropna()
            vp_n, vn = valid_pn[m_p].values, valid_pn[m_n].values
            t_st2, t_p2 = ttest_rel(vp_n, vn)
            w_st2, w_p2 = wilcoxon(vp_n, vn)
            cd2 = cohen_d(vp_n, vn)
            f.write(f"Proposed vs N2N4M (n={len(vp_n)}):\n")
            f.write(f"  Paired t-test: p = {t_p2:.4e}\n")
            f.write(f"  Wilcoxon:      p = {w_p2:.4e}\n")
            f.write(f"  Cohen's d:     {cd2:.3f}\n\n")

    # Aggregate formatting for LaTeX
    def fmt_ci(col):
        val = df[col].dropna().values
        m, h = mean_confidence_interval(val)
        return f"{m:.2f} ± {np.std(val):.2f} (95% CI: [{m-h:.2f}, {m+h:.2f}]) [n={len(val)}]"
        
    tex_out = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lccc}\n\\hline\n"
    tex_out += "Metric & Proposed Pipeline & HSID-CNN & N2N4M \\\\\n\\hline\n"
    tex_out += f"SNR & {fmt_ci('Pipe_SNR')} & {fmt_ci('HSID_SNR')} & {fmt_ci('N2N_SNR')} \\\\\n"
    tex_out += f"SAM & {fmt_ci('Pipe_SAM')} & {fmt_ci('HSID_SAM')} & {fmt_ci('N2N_SAM')} \\\\\n"
    tex_out += f"Pearson & {fmt_ci('Pipe_Corr')} & {fmt_ci('HSID_Corr')} & {fmt_ci('N2N_Corr')} \\\\\n"
    tex_out += "\\hline\n\\end{tabular}\n\\end{table}"
    
    with open(os.path.join(tab_dir, "aggregate_comparison_all_methods.tex"), "w") as f:
        f.write(tex_out)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--physics', default='test/physics_output')
    p.add_argument('--ml',      default='test/ml_output')
    p.add_argument('--hsid',    default='hsid_cnn/results')
    p.add_argument('--n2n',     default='results/n2n4m_output')
    args = p.parse_args()
    
    compare_pipelines(args.physics, args.ml, args.hsid, args.n2n)
