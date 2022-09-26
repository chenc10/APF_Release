### set home directory
home='/root/adaptive_freezing'

### prepare master and workers
master='localhost'
workers='localhost'
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

### launch server process
command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $home/server_process.py --world_size=$world_size --server_port=$((20000+trial_no)) --remarks=$remarks --trial_no=$trial_no"
nohup ssh $master $command > $log_dir/server.log 2>&1 &

### launch worker process
num=0
for i in $workers
do
	command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $home/worker_process.py --server_ip=$master --server_port=$((20000+trial_no)) --rank=$num --world_size=$world_size --remarks=$remarks --trial_no=$trial_no"
	echo $command
	nohup ssh $i $command > $log_dir/worker_$num.log 2>&1 &
	num=$((num+1))
done
