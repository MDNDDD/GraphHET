#!/usr/bin/env bash

# Make this script executable with:
# chmod +x ./LDBC-GPU-csr.sh

data_dir="${DATA_DIR:-/home/mabojing/data}"
repo_dir="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
executable="${EXECUTABLE:-${repo_dir}/build/bin_gpu/Test_GPU_CSR}"
log_dir="${LOG_DIR:-${repo_dir}/logs}"
input_file="${INPUT_FILE:-input.txt}"
result_file="${RESULT_FILE:-result-gpu-csr.csv}"
gpu_id="${GPU_ID:-0}"

filenames=(
    "wiki-Talk"
    "cit-Patents"
    "kgs"
    # "datagen-7_7-zf"
    # "datagen-7_5-fb"
    # "datagen-7_8-zf"
    # "datagen-7_6-fb"
    # "dota-league"
    # "graph500-22"
    # "datagen-7_9-fb"
    # "datagen-8_2-zf"
    # "datagen-8_0-fb"
    # "graph500-23"
    # "datagen-8_3-zf"
    # "datagen-8_1-fb"
)

mkdir -p "${log_dir}"

if [[ ! -x "${executable}" ]]; then
    echo "Executable not found or not executable: ${executable}" >&2
    exit 1
fi

for filename in "${filenames[@]}"
do
    for i in {1..1}
    do
        echo "Testing dataset ${filename} on GPU ${gpu_id} ..."
        printf '%s\n%s\n' "${data_dir%/}/" "${filename}" > "${input_file}"

        echo "${filename}" >> "${result_file}"

        CUDA_VISIBLE_DEVICES="${gpu_id}" "${executable}" < "${input_file}" > "${log_dir}/${filename}-GPU-csr.output"
        echo "Finished dataset ${filename}."
    done
done
