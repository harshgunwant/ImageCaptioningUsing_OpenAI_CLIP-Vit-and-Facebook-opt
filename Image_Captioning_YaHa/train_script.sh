ARGS="
--output_dir ./flamingo-coco_2
--run_name flamingo-tiny-vitL
--do_train --do_eval
--optim adamw_torch
--learning_rate 0.0001 
--warmup_steps 5000
--lr_scheduler_type constant_with_warmup
--per_device_train_batch_size 3
--per_device_eval_batch_size 64
--gradient_accumulation_steps 1
--evaluation_strategy steps
--eval_steps 1000
--save_strategy epoch
--save_total_limit 2
--log_level info
--dataloader_num_workers 0
--dataloader_pin_memory True
--fp16
--num_train_epochs 2
--report_to none
--ddp_find_unused_parameters False
"
'C:\Users\yg375\AppData\Local\Programs\Python\Python310\python.exe' ./train_and_eval_combined.py $ARGS
