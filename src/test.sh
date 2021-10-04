#for((i=1;i<=5;i++));
#do
  for((j=2;j<=5;j++));
  do
    python train.py --dataset "chameleon" --sample_size 15 --epoch_num $j --identify "sample_epoch"
  done
#done