### set home directory, server, edge and clients
home='/root/adaptive_freezing'
server='localhost'
edges='localhost localhost'
echo 'Server_host: '$server
echo 'Edge_hosts: '$edges

### ---- create log dir and code snapshot
read -p "Enter exp-setup notes: " remarks
read -p "Specify allocated GPU-ID: " cuda 
trial_no=$(ls $home/Logs/ | wc -l)
log_dir=$home/Logs/${trial_no}_$remarks

mkdir -p $log_dir/code_snapshot 
cp $home/train.sh $log_dir/code_snapshot
cp $home/*.py $log_dir/code_snapshot
cp -r $home/models $log_dir/code_snapshot
rm -f Latest_Log && ln -s $log_dir Latest_Log
echo 'logs in '$log_dir

### ---- launch server process
edge_number=0
for e_ip in $edges
do
		edge_number=$((edge_number+1))
done
command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $home/server_process.py \
	--port=$((20000+trial_no)) --edge_number=$edge_number \
	--remarks=$remarks --trial_no=$trial_no"
echo "launch server:"
echo $command
echo 
nohup ssh $server $command > $log_dir/server.log 2>&1 &
sleep 1

### ---- launch edge process and the assigned clients one by one
edge_serial_num=0
for e_ip in $edges
do
	clients='localhost localhost'

	client_number=0
	for c_ip in $clients
	do
			client_number=$((client_number+1))
	done

	command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $home/edge_process.py \
		--server_ip=$server --server_port=$((20000+trial_no)) --edge_rank=$edge_serial_num \
		--port=$((30000 + trial_no + 10*edge_serial_num)) --client_number=$client_number \
		--remarks=$remarks --trial_no=$trial_no"
	echo "launch edge "$edge_serial_num": "
	echo $command
	nohup ssh $e_ip $command > $log_dir/edge_$edge_serial_num.log 2>&1 &
	sleep 1


	### respectively launch each client for this edge
	client_serial_num=0
	for c_ip in $clients
	do
		command="export CUDA_VISIBLE_DEVICES=$cuda && python3 $home/worker_process.py \
			--server_ip=$e_ip --server_port=$((30000 + trial_no + 10*edge_serial_num)) --rank=$client_serial_num --world_size=$client_number\
			--remarks=$remarks --trial_no=$trial_no"
		echo "launch client "${edge_serial_num}"-"${client_serial_num}": "
		echo $command
		nohup ssh $c_ip $command > $log_dir/client_${edge_serial_num}_${client_serial_num}.log 2>&1 &
		client_serial_num=$((client_serial_num+1))
	done

	edge_serial_num=$((edge_serial_num+1))
	echo
done