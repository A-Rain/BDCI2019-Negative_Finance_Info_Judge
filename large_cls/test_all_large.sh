model_path='/userhome/bert/raw/bert_large'
tokenizer_path='/userhome/bert/raw/bert_large/vocab.txt'
data_path='/userhome/project/data_final/test_all_sub_max512/cv_'
output_path='./proc_data/test_all/bert_large_4v/cv_'
# entity_pretrain is for one step

for i in {0..4}
do
    python ./examples/run_glue.py \
    --model_type bert \
    --model_name_or_path $model_path  \
    --tokenizer_name  $tokenizer_path \
    --do_test   \
    --do_train  \
    --do_eval   \
    --do_entity \
    --task_name qnli     \
    --data_dir $data_path$i  \
    --output_dir $output_path$i   \
    --max_seq_length 512   \
    --per_gpu_eval_batch_size=6  \
    --per_gpu_train_batch_size=6   \
    --max_steps=2600  \
    --learning_rate 2e-5 \
    --overwrite_output_dir   \
    --overwrite_cache \
    --warmup_steps=100  \
    --evaluate_during_training \
    --logging_steps 30 \
    --save_steps 30
done