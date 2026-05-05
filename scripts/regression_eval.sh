#!/bin/bash
#
# NR-SLAM Automated Regression & Evaluation Script
# Usage: ./scripts/regression_eval.sh [dataset_path] [config_label]
#
# Example:
#   ./scripts/regression_eval.sh "/path/to/hamlyn_01" "stageD4_fusion"
#   ./scripts/regression_eval.sh "/path/to/simulation" "stageD4_no_proj"
#

set -e

# ============ Configuration ============

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BINARY="${PROJECT_ROOT}/build/bin/simulation"
RESULTS_DIR="${PROJECT_ROOT}/regression_results"
SETTINGS="${PROJECT_ROOT}/data/simulation/settings.yaml"

# Default values
DATASET_PATH="${1:-.}"
CONFIG_LABEL="${2:-default}"
STARTING_FRAME=60
END_FRAME=90
WINDOW_SIZE=$((END_FRAME - STARTING_FRAME))

# Output files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_SUBDIR="${RESULTS_DIR}/${CONFIG_LABEL}_${TIMESTAMP}"
METRICS_CSV="${RESULT_SUBDIR}/metrics.csv"
SDF_PLY="${RESULT_SUBDIR}/sdf_export.ply"
REPORT="${RESULT_SUBDIR}/report.txt"

# ============ Validation ============

if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found: $BINARY"
    echo "Please run: cmake -S . -B build && cmake --build build"
    exit 1
fi

if [ "$DATASET_PATH" = "." ]; then
    echo "ERROR: Dataset path required"
    echo "Usage: $0 <dataset_path> [config_label]"
    exit 1
fi

if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset path does not exist: $DATASET_PATH"
    exit 1
fi

# ============ Setup ============

mkdir -p "$RESULT_SUBDIR"

echo "==========================================="
echo "NR-SLAM Regression Evaluation"
echo "==========================================="
echo "Config Label:  $CONFIG_LABEL"
echo "Dataset:       $DATASET_PATH"
echo "Frame Range:   $STARTING_FRAME - $END_FRAME ($WINDOW_SIZE frames)"
echo "Results Dir:   $RESULT_SUBDIR"
echo "Timestamp:     $TIMESTAMP"
echo "==========================================="
echo ""

# ============ Run Regression ============

echo "[1/4] Running SLAM regression (frames $STARTING_FRAME-$END_FRAME)..."
"$BINARY" \
    --dataset_path="$DATASET_PATH" \
    --settings_path="$SETTINGS" \
    --starting_frame="$STARTING_FRAME" \
    --end_frame="$END_FRAME" \
    --metrics_csv_path="$METRICS_CSV" \
    --sdf_export_path="$SDF_PLY" 2>&1 | grep -E "Processing|Runtime|Exporting|written"

echo ""
echo "[2/4] Parsing CSV metrics..."

# ============ Parse CSV ============

if [ ! -f "$METRICS_CSV" ]; then
    echo "ERROR: Metrics CSV not generated"
    exit 1
fi

# Count frames and collect statistics
{
    echo "CSV Analysis Report"
    echo "===================="
    echo ""
    
    # Line count
    TOTAL_LINES=$(wc -l < "$METRICS_CSV")
    DATA_ROWS=$((TOTAL_LINES - 1))
    echo "Frame rows processed: $DATA_ROWS"
    echo ""
    
    # NN Alignment stats (columns 2-7)
    echo "Neural Network Alignment Statistics:"
    awk -F, 'NR>1 {
        mode=$3;
        count[mode]++;
        inlier_ratios[NR]=$8;
    }
    END {
        for (m in count) print "  Mode " m ": " count[m] " frames";
        
        n=0;
        for (i in inlier_ratios) {
            if (inlier_ratios[i] ~ /^[0-9.]/) {
                ratio = inlier_ratios[i];
                n++;
                if (n==1 || ratio < min) min=ratio;
                if (ratio > max) max=ratio;
                avg += ratio;
            }
        }
        if (n>0) {
            printf "  Inlier ratio [min, max]=%.6f, %.6f (n=%d)\n", min, max, n;
            printf "  Inlier ratio avg=%.6f\n", avg/n;
        }
    }' "$METRICS_CSV"
    echo ""
    
    # SDF stats (columns 8-12)
    echo "SDF Voxel Statistics:"
    awk -F, 'NR>1 {
        if ($9 ~ /^[0-9]+$/) {
            voxels[NR]=$9;
            dense[NR]=$10;
            sparse[NR]=$11;
            updates[NR]=$12;
        }
    }
    END {
        vox_sum=0; dense_sum=0; sparse_sum=0; upd_sum=0;
        for (i in voxels) {
            vox_sum += voxels[i];
            dense_sum += dense[i];
            sparse_sum += sparse[i];
            upd_sum += updates[i];
            count++;
        }
        if (count>0) {
            printf "  Avg active voxels: %.0f\n", vox_sum/count;
            printf "  Total dense points: %d\n", dense_sum;
            printf "  Total sparse points: %d\n", sparse_sum;
            printf "  Total voxel updates: %d\n", upd_sum;
        }
    }' "$METRICS_CSV"
    echo ""
    
    # Surfel stats (columns 13-14)
    echo "Surfel Statistics:"
    awk -F, 'NR>1 {
        if ($14 ~ /^[0-9]+$/) {
            active[NR]=$14;
            finalized[NR]=$15;
        }
    }
    END {
        act_min=999999; fin_max=0;
        for (i in active) {
            if (active[i] < act_min) act_min=active[i];
            if (active[i] > act_max) act_max=active[i];
            if (finalized[i] > fin_max) fin_max=finalized[i];
            act_sum += active[i];
            fin_sum += finalized[i];
            count++;
        }
        if (count>0) {
            printf "  Active surfels [min,max]: %d, %d\n", act_min, act_max;
            printf "  Avg active: %.0f\n", act_sum/count;
            printf "  Max finalized: %d\n", fin_max;
        }
    }' "$METRICS_CSV"
    echo ""
    
    # Phi constraint stats (columns 15-18)
    echo "Phi Constraint Statistics:"
    awk -F, 'NR>1 {
        if ($16 ~ /^[0-9]+$/) {
            checked[NR]=$16;
            accepted[NR]=$17;
            rejected[NR]=$18;
            projected[NR]=$19;
        }
    }
    END {
        for (i in checked) {
            chk_sum += checked[i];
            acc_sum += accepted[i];
            rej_sum += rejected[i];
            prj_sum += projected[i];
            count++;
        }
        if (count>0 && chk_sum>0) {
            printf "  Total checked: %d\n", chk_sum;
            printf "  Total accepted: %d (%.1f%%)\n", acc_sum, 100*acc_sum/chk_sum;
            printf "  Total rejected: %d (%.1f%%)\n", rej_sum, 100*rej_sum/chk_sum;
            printf "  Total projected: %d\n", prj_sum;
            printf "  Projection ratio: %.1f%%\n", prj_sum>0 ? 100*prj_sum/acc_sum : 0;
        }
    }' "$METRICS_CSV"
    echo ""
    
    # Phi distribution stats (columns 19-22)
    echo "Phi Distribution (Before/After Projection):"
    awk -F, 'NR>1 {
        if ($20 ~ /^[0-9.-]+$/) {
            phi_before_min[NR]=$20;
            phi_before_max[NR]=$21;
            phi_after_min[NR]=$22;
            phi_after_max[NR]=$23;
        }
    }
    END {
        global_before_min=999;
        global_before_max=0;
        for (i in phi_before_min) {
            if (phi_before_min[i] < global_before_min) global_before_min=phi_before_min[i];
            if (phi_before_max[i] > global_before_max) global_before_max=phi_before_max[i];
            count++;
        }
        if (count>0) {
            printf "  Before-projection phi range: [%.6f, %.6f] meters\n", global_before_min, global_before_max;
            printf "  Max |phi| (before): %.4f m (= %.1f mm)\n", global_before_max, global_before_max*1000;
        }
    }' "$METRICS_CSV"
    
} | tee "$REPORT"

echo ""
echo "[3/4] Report saved to: $REPORT"
echo ""
echo "[4/4] Summary"
echo "============"
echo "✓ Metrics CSV: $METRICS_CSV"
echo "✓ SDF PLY:     $SDF_PLY"
if [ -f "$SDF_PLY" ]; then
    SDF_SIZE=$(du -h "$SDF_PLY" | cut -f1)
    echo "  (size: $SDF_SIZE)"
fi
echo "✓ Report:      $REPORT"
echo ""
echo "Regression evaluation complete!"
echo "Config: $CONFIG_LABEL"
echo ""
