from evaluation.ablation_study import evaluate_ablation_run, run_ablation_experiments
from evaluation.report_generator import generate_thesis_report, calculate_ci95
from evaluation.plot_gate_dynamics import (
    plot_gate_weights_across_datasets,
    plot_od_heatmap_sdn,
    plot_burst_spike_case_study,
    plot_all_thesis_figures
)

__all__ = [
    'evaluate_ablation_run',
    'run_ablation_experiments',
    'generate_thesis_report',
    'calculate_ci95',
    'plot_gate_weights_across_datasets',
    'plot_od_heatmap_sdn',
    'plot_burst_spike_case_study',
    'plot_all_thesis_figures'
]
