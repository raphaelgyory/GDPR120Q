python ./run_qa.py \
  --model_name_or_path ./models/bert-base-pretraining/checkpoint-190000 \
  --train_file ./data/GDPR120Q_train_formatted.json \
  --validation_file ./data/GDPR120Q_validation_formatted.json \
  --test_file ./data/GDPR120Q_test_formatted.json \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 4 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative \
  --output_dir ./models/bert-base-uncased-training-pretraining \
  --logging_dir ./logs/bert-base-uncased-training-pretraining \
  --logging_steps 150 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end \
  --save_total_limit 1 \
  --metric_for_best_model f1
  
