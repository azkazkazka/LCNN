log_name=log_metric
score_file=path_to_score_file
protocol_file=path_to_protocol_file

echo -e "Generating metric scores"
python calculate_eval.py --scores ${score_file} --protocols ${protocol_file} > ${log_name}.txt 2>${log_name}_err.txt
echo -e "Results of the metric scores can be seen in $PWD/${log_name}.txt"
