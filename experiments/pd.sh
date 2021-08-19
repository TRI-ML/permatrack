# Initial model pre-trained on NuScenes3D: https://drive.google.com/open?id=1ZSG9swryMEfBJ104WH8CP7kcypCobFlU
# Resulting model trained on PD: https://s3.console.aws.amazon.com/s3/object/tri-ml-public?region=us-east-1&prefix=github/permatrack/pd_17fr_21ep_vis.pth

cd src
# train
python main.py tracking --exp_id pd_supinvis --occlusion_thresh 0.15 --visibility_thresh 0.05 --dataset pd_tracking --dataset_version val --same_aug_pre --hm_disturb 0.0 --lost_disturb 0.0 --fp_disturb 0.0 --gpus 0,1,2,3,4,5,6,7 --batch_size 2 --load_model ../models/nuScenes_3Ddetection_e140.pth --val_intervals 2 --is_recurrent --gru_filter_size 7 --input_len 17 --pre_thresh 0.4 --hm_weight 0.5 --num_epochs 21 --lr_step 7 --const_v_over_occl --sup_invis --invis_hm_weight 20 --use_occl_len --occl_len_mult 5 --num_iter 5000 --visibility --visibility_thresh_eval 0.2
# test
CUDA_VISIBLE_DEVICES=0 python test.py tracking --exp_id pd_supinvis --dataset pd_tracking --dataset_version val --track_thresh 0.4 --resume --is_recurrent --debug 4 --gru_filter_size 7 --num_gru_layers 1 --stream_test 
cd ..