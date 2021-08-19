# Initial model pre-trained on PD: https://s3.console.aws.amazon.com/s3/object/tri-ml-public?region=us-east-1&prefix=github/permatrack/pd_17fr_21ep_vis.pth
# Resulting model trained on CrowdHuman: https://s3.console.aws.amazon.com/s3/object/tri-ml-public?region=us-east-1&prefix=github/permatrack/crowdhuman.pth

cd src
# train
python main.py tracking --exp_id crowdhuman --occlusion_thresh 0.15 --visibility_thresh 0.05 --dataset joint --dataset1 crowdhuman --dataset2 pd_tracking --dataset_version x --same_aug_pre --hm_disturb 0.0 --lost_disturb 0.0 --fp_disturb 0.0 --gpus 0,1,2,3,4,5,6,7 --batch_size 2 --load_model ../models/pd_17fr_21ep_vis.pth --val_intervals 100 --is_recurrent --gru_filter_size 7 --input_len 17 --pre_thresh 0.4 --hm_weight 0.5 --const_v_over_occl --sup_invis --invis_hm_weight 20 --use_occl_len --occl_len_mult 5 --visibility --num_iter 5000 --num_epochs 9 --lr_step 5 --ltrb_amodal --only_ped --reuse_hm
cd ..