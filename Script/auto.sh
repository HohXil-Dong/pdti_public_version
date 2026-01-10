#!/bin/zsh

{
    PDTI_get_knot_info < i_greenf
    PDTI_GreenPointSources < i_greenf
    PDTI_get_ndj_main  < i_Cmatrix
    PDTI_get_init_model_para_TmpSm < i_Cmatrix
    # For Next iter input and Backup
    cp fort.40 input_fort.40
    cp fort.40 input_fort_iter1.40
    # ------------------------------
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
        # -----------------------
    done

    echo "Done."

} 2>&1 | tee inv.log
