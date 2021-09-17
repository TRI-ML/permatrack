# Initial model pre-trained on PD + CrowdHuman: https://tri-ml-public.s3.amazonaws.com/github/permatrack/crowdhuman.pth
# Resulting model trained on MOT17 half train: https://tri-ml-public.s3.amazonaws.com/github/permatrack/mot_half.pth

cd src
# train
python main.py tracking --exp_id mot17_half --occlusion_thresh 0.15 --visibility_thresh 0.05 --dataset joint --dataset1 mot --dataset2 pd_tracking --dataset_version 17halftrain --same_aug_pre --hm_disturb 0.0 --lost_disturb 0.0 --fp_disturb 0.0 --gpus 0,1,2,3,4,5,6,7 --batch_size 2 --load_model ../models/crowdhuman.pth --val_intervals 1 --is_recurrent --gru_filter_size 7 --input_len 17 --pre_thresh 0.4 --hm_weight 0.5 --const_v_over_occl --sup_invis --invis_hm_weight 20 --use_occl_len --occl_len_mult 5 --visibility --num_iter 1600 --num_epochs 5 --lr_step 4 --visibility_thresh_eval 0.1 --ltrb_amodal --only_ped --reuse_hm
# test
CUDA_VISIBLE_DEVICES=0 python test.py tracking --exp_id mot17_half --dataset mot --dataset_version 17halfval --track_thresh 0.4 --resume --is_recurrent --gru_filter_size 7 --num_gru_layers 1 --visibility_thresh_eval 0.1 --stream_test --only_ped --ltrb_amodal --visibility
# test with T.R.
CUDA_VISIBLE_DEVICES=0 python test.py tracking --exp_id mot17_half --dataset mot --dataset_version 17halfval --track_thresh 0.4 --resume --is_recurrent --gru_filter_size 7 --num_gru_layers 1 --visibility_thresh_eval 0.1 --stream_test --only_ped --ltrb_amodal --visibility --max_age 32
# test with public detection
CUDA_VISIBLE_DEVICES=0 python test.py tracking --exp_id mot17_half --dataset mot --dataset_version 17halfval --track_thresh 0.4 --resume --is_recurrent --gru_filter_size 7 --num_gru_layers 1 --visibility_thresh_eval 0.1 --stream_test --only_ped --ltrb_amodal --visibility --public_det --load_results ../data/mot17/results/val_half_det.json
cd ..
