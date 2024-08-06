#!/bin/bash

# Function to display help
usage() {
    echo "Usage: $0 [-h] [-e exp1|exp2|all]"
    echo "Options:"
    echo "  -h      Display this help message."
    echo "  -e      Choose the experiment to run: exp1 for exp1_1_NS_FL, exp2 for exp1_2_RS_FL, or all for both."
}

# Parse command-line options
while getopts ":he:" option; do
    case $option in
        h) # display help
            usage
            exit 0
            ;;
        e) # choose experiment
            experiment=$OPTARG
            ;;
        \?) # invalid option
            echo "Error: Invalid option"
            usage
            exit 1
            ;;
    esac
done

# Set up experiment commands
run_exp1_1="nvflare simulator jobs/exp1_1_NS_FL  -w workspaces/exp1_1_NS_ws -t 1 -n 2 -gpu 0"
run_exp1_2="nvflare simulator jobs/exp1_2_RS_FL  -w workspaces/exp1_2_RS_ws -t 1 -n 2 -gpu 0"
run_exp1_3="nvflare simulator jobs/exp1_3_CS_FL  -w workspaces/exp1_3_CS_ws -t 1 -n 2 -gpu 0"
run_exp2_1="nvflare job submit -j jobs/exp2_1_NS_HE -debug"
run_exp2_2="nvflare job submit -j jobs/exp2_2_RS_HE -debug"
run_exp2_3="nvflare job submit -j jobs/exp2_3_CS_HE -debug"

prepare_poc='nvflare poc prepare -he'

start_poc_server='nvflare poc start -ex admin@nvidia.com'

stop_poc_server='nvflare poc stop'


# Run the chosen experiment
if [ "$experiment" == "exp1-1" ]; then
    echo "Running experiment exp1_1_NS_FL"
    eval $run_exp1_1

elif [ "$experiment" == "exp1-2" ]; then
    echo "Running experiment exp1_2_RS_FL"
    eval $run_exp1_2

elif [ "$experiment" == "exp1-3" ]; then
    echo "Running experiment exp1_3_CS"
    eval $run_exp1_3

elif [ "$experiment" == "exp2-1" ]; then
    echo "Running experiment exp2_1_NS_HE"

    echo "preparing poc workspace"
    eval $prepare_poc
    echo "starting poc server"
    eval $start_poc_server
    echo "waiting for poc server to start"
    echo submitting job to server
    eval $run_exp2_1

elif [ "$experiment" == "exp2-2" ]; then
    echo "Running experiment exp2_2_RS_HE"

    echo "preparing poc workspace"
    eval $prepare_poc
    echo "starting poc server"
    eval $start_poc_server
    echo "waiting for poc server to start"
    echo submitting job to server
    eval $run_exp2_2

elif [ "$experiment" == "exp2-3" ]; then
    echo "Running experiment exp2_3_CS_HE"

    echo "preparing poc workspace"
    eval $prepare_poc
    echo "starting poc server"
    eval $start_poc_server
    echo "waiting for poc server to start"
    echo submitting job to server
    eval $run_exp2_3
else
    echo "Error: No valid experiment chosen."
    usage
    exit 1
fi

