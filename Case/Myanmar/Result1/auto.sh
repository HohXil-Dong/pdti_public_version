#!/bin/zsh

{
    # Start timer
    start_time=$(date +%s)

    # Initial setup
    PDTI_get_knot_info < i_greenf
    PDTI_GreenPointSources < i_greenf
    PDTI_get_ndj_main  < i_Cmatrix
    PDTI_get_init_model_para_TmpSm < i_Cmatrix

    # Backup & next iteration input
    cp fort.40 input_fort.40
    cp fort.40 input_fort_iter1.40
    # d_H.matrix
    PDTI_pre_inv_new  < i_Cmatrix

    # Loop 10 times
    for ((i=1; i<=10; i++)); do
        echo "Running iteration $i..."

        PDTI_get_covariance_grn < i_Cmatrix
        PDTI_ABIC_TA < i_Cmatrix

        cp fort.40 input_fort.40
        cp fort.40 input_fort_iter$((i+1)).40  
        # After the 1st iteration, set the 10th column of the 2nd row to 1
        if [[ $i -eq 1 ]]; then
            sed -i -E '2s/^(([^ ]+[ ]+){9})[^ ]+/\11/' i_Cmatrix
        fi

        # Check if .nstep.inf exists in the current directory
        if [[ -f .nstep.inf ]]; then
            echo ".nstep.inf detected. Ending inversion early."
            break
        fi
    done

    # ==========================================
    # FINAL STEP: Optimization & Final Inversion
    # ==========================================
    echo "------------------------------------------------"
    echo "Searching for optimal hyperparameters (Min ABIC)..."

    # Find global minimum ABIC 
    min_abic=$(grep "^ABIC :" inv.log | awk '{print $11}' | sort -n | head -1)

    if [[ -z "$min_abic" ]]; then
        echo "Error: Min ABIC not found."
        exit 1
    fi
    echo "Minimum ABIC: $min_abic"

    # Extract parameters corresponding to Min ABIC
    # Alpha (1st), Beta (5th), Diff (13th)
    raw_params=($(grep "^ABIC :" inv.log | awk -v target="$min_abic" '$11 == target {print $3, $7, $15; exit}'))

    raw_alpha=${raw_params[1]}
    raw_beta=${raw_params[2]}
    raw_diff=${raw_params[3]}

    if [[ -z "$raw_alpha" || -z "$raw_beta" || -z "$raw_diff" ]]; then
        echo "Error: Failed to extract parameters."
        exit 1
    fi

    # Convert to decimal format
    optimal_alpha=$(python3 -c "from decimal import Decimal; import sys; print(format(Decimal(sys.argv[1]), 'f'))" "$raw_alpha")
    optimal_beta=$(python3 -c "from decimal import Decimal; import sys; print(format(Decimal(sys.argv[1]), 'f'))" "$raw_beta")
    optimal_diff=$(python3 -c "from decimal import Decimal; import sys; print(format(Decimal(sys.argv[1]), 'f'))" "$raw_diff")

    echo "Optimal Parameters: Alpha=$optimal_alpha, Beta=$optimal_beta, Diff=$optimal_diff"

    # Update i_Cmatrix
    echo "Updating i_Cmatrix..."
    cp i_Cmatrix i_Cmatrix.bak
    
    awk -v alpha="$optimal_alpha" -v beta="$optimal_beta" -v diff="$optimal_diff" '
    NR==3 {
        if (NF < 10) {
            print "Error: Row 3 has fewer than 10 columns. Aborting." > "/dev/stderr"
            exit 1
        }
        
        $1 = alpha
        $9 = beta
        
        if (NF == 10) {
            print $0, diff
        } else {
            print "Warning: Row 3 has >10 columns. Replacing col 11." > "/dev/stderr"
            $11 = diff
            print $0
        }
        next
    }
    {print}
    ' i_Cmatrix > i_Cmatrix.tmp

    if [[ $? -ne 0 ]]; then
        echo "Fatal Error: Update failed."
        rm -f i_Cmatrix.tmp
        exit 1
    fi

    mv i_Cmatrix.tmp i_Cmatrix

    # Final Inversion
    echo "Running Final Step..."
    PDTI_inversionA_TAd < i_Cmatrix


    end_time=$(date +%s)
    duration=$(( end_time - start_time ))
    hours=$(( duration / 3600 ))
    minutes=$(( (duration % 3600) / 60 ))
    seconds=$(( duration % 60 ))

    echo "------------------------------------------------"
    printf "Total Runtime: %02d:%02d:%02d (%d seconds)\n" $hours $minutes $seconds $duration
    echo "Done."

} 2>&1 | tee -a inv.log