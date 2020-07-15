master='localhost'
workers='localhost localhost'

echo 'master(coordinator): '$master
echo 'worker_hosts: '$workers

world_size=0
for i in $workers
do
		world_size=$((world_size+1))
done

home=`pwd`

#model='AlexNet'
#model='CNNKws'
#model='DenseNet'
#model='ResNet18'
#model='VGG'
model='CNN'

if [ $model = 'CNN' ]
then
	dataset='Cifar10'
	initial_lr=0.01
	batch_size=100

	weight_decay=0.01

	sync_frequency=500
	frozen_frequency=500
	ema_alpha=0.99
	stable_threshold=0.05
fi

if [ $model = 'ResNet18' ]
then
	dataset='Cifar10'
	initial_lr=0.1
	batch_size=100

	weight_decay=0.01

	sync_frequency=50
	frozen_frequency=50
	ema_alpha=0.99
	stable_threshold=0.05
fi

if [ $model = 'VGG' ]
then 
	dataset='Cifar10'
	initial_lr=0.01
	batch_size=32

	weight_decay=0.0001

	sync_frequency=1
	frozen_frequency=50
	ema_alpha=0.99
	stable_threshold=0.05
fi

if [ $model = 'AlexNet' ]
then
	dataset='ImageNet'
	initial_lr=0.01
	batch_size=32

	weight_decay=0.0001

	sync_frequency=1
	frozen_frequency=500
	ema_alpha=0.95
	stable_threshold=0.02
fi

if [ $model = 'DenseNet' ]
then
	dataset='ImageNet'
	initial_lr=0.1
	batch_size=32

	weight_decay=0.0001

	sync_frequency=1
	frozen_frequency=500
	ema_alpha=0.95
	stable_threshold=0.02
fi

if [ $model = 'CNNKws' ]
then
	dataset='KWS'
	initial_lr=0.05
	batch_size=50

	weight_decay=0.01

	sync_frequency=650
	frozen_frequency=650
	ema_alpha=0.95
	stable_threshold=0.1
fi

echo $model
	
num=0
for i in $workers
do
	echo "python $home/worker_process_apf.py \ 
			--master_address='tcp://'$master':22222' --rank=$num --world_size=$world_size \ 
			--model=$model --dataset=$dataset --initial_lr=$initial_lr --batch_size=$batch_size \ 
			--weight_decay=$weight_decay \ 
			--sync_frequency=$sync_frequency --frozen_frequency=$frozen_frequency --ema_alpha=$ema_alpha --stable_threshold=$stable_threshold"

	nohup ssh $i "python $home/worker_process_apf.py \
			--master_address='tcp://'$master':22222' --rank=$num --world_size=$world_size \
			--model=$model --dataset=$dataset --initial_lr=$initial_lr --batch_size=$batch_size \
			--weight_decay=$weight_decay \
			--sync_frequency=$sync_frequency --frozen_frequency=$frozen_frequency --ema_alpha=$ema_alpha --stable_threshold=$stable_threshold" \
			> $home/worker_$num.log 2>&1 &

	num=$((num+1))
done

