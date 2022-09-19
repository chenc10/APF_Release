### set home directory
home='/root/adaptive_freezing'

### prepare master and workers
master='localhost'
workers='localhost'
# workers='localhost localhost'
#workers='localhost localhost localhost localhost'
#localhost localhost'
echo 'master(coordinator): '$master
echo 'worker_hosts: '$workers
world_size=0
for i in $workers
do
		world_size=$((world_size+1))
done

### create log dir and code snapshot
read -p "Enter exp-setup notes: " remarks
read -p "Specify allocated GPU-ID (world_size: $world_size): " cuda 
trial_no=$(ls $home/Logs/ | wc -l)
log_dir=$home/Logs/${trial_no}_$remarks

mkdir -p $log_dir/code_snapshot 
cp $home/train.sh $log_dir/code_snapshot
cp $home/*.py $log_dir/code_snapshot
cp -r $home/models $log_dir/code_snapshot
rm -f Latest_Log && ln -s $log_dir Latest_Log
echo 'logs in '$log_dir

### launch process
num=0
job=worker_process.py
for i in $workers
do
	command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $home/$job --master_address=tcp://${master}:$((20000+trial_no)) --rank=$num --world_size=$world_size --remarks=$remarks --trial_no=$trial_no"
	echo $command
	nohup ssh $i $command > $log_dir/worker_$num.log 2>&1 &
	num=$((num+1))
done
