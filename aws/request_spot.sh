SPOT_REQUEST_ID=`aws ec2 request-spot-instances --spot-price "2.69" --instance-count 1 --type "one-time" --launch-specification file://specification.json | grep SpotInstanceRequestId | awk '{print $2}' | sed s/,// | sed s/\"// | sed s/\"//`


####
# to get info about the spot bid:
WAIT_SECONDS=5
while true; do 
    SPOT_INST_ID=`aws ec2 describe-spot-instance-requests --spot-instance-request-ids $SPOT_REQUEST_ID | grep InstanceId | awk '{print $2}' | sed s/,// | sed s/\"// | sed s/\"//`
    if [ ! -z "$SPOT_INST_ID" ]; then
	echo "successfully got spot instance id: $SPOT_INST_ID"
	break
    else
	echo "waiting $WAIT_SECONDS second(s) to check if spot request has been filled"
	sleep $WAIT_SECONDS
    fi
done


###
# to get the ip address:
while true; do
    SPOT_IP=`aws ec2 describe-instances --instance-ids $SPOT_INST_ID | grep PublicIpAddress | awk '{print $2}' | sed s/,// | sed s/\"// | sed s/\"// | sed 's/\./-/g'` 
    if [ ! -z "$SPOT_IP" ]; then
	echo "successfully got ip address: $SPOT_IP"
	break
    else
	echo "waiting $WAIT_SECONDS second(s) to get the IP address"
	sleep $WAIT_SECONDS
    fi
done


while true; do
    STATE=`aws ec2 describe-instances --instance-ids $SPOT_INST_ID | grep \"Name\" | awk '{print $2}' | sed s/\"//g`
    if [ "$STATE" == "running" ]; then 
	break
    else
	echo "waiting $WAIT_SECONDS second(s) for instance to be in 'running' mode"
	sleep $WAIT_SECONDS
    fi
done


###
# for some reason we have to wait for scp to work
for i in `seq 1 100`; do
    echo "waiting for 100 seconds, $i seconds have passed..."
    sleep 1
done

###
# to copy the .pem file over:
echo "about to copy .pem file over..."
scp -i "/home/ec2-user/projects/ARKcat/aws/jesse-key-pair-uswest2.pem" -oStrictHostKeyChecking=no -r /home/ec2-user/projects/ARKcat/aws/jesse-key-pair-uswest2.pem ec2-user@ec2-${SPOT_IP}.us-west-2.compute.amazonaws.com:/home/ec2-user/
echo "copied!"


###
# gets the current instance's ip address
CUR_IP=`curl -s http://169.254.169.254/latest/meta-data/public-ipv4`


###
# train models and move 
ssh -i "/home/ec2-user/projects/ARKcat/aws/jesse-key-pair-uswest2.pem" -oStrictHostKeyChecking=no ec2-user@ec2-${SPOT_IP}.us-west-2.compute.amazonaws.com "source activate arkcat; cd /home/ec2-user/projects/hyperopt; git checkout hyperopt/dpp.py; git pull; cd /home/ec2-user/projects/ARKcat/src; git checkout train_and_eval_pinot.sh; git pull; bash train_and_eval_spot.sh ${1} ${2} $CUR_IP $SPOT_INST_ID ${3}"


###
# to ssh into this machine, I had to go to the console, click on instances, scroll down to security groups, click default, click actions, click edit inbound rules, and change the All TCP source to Anywhere


###
# aws ec2 terminate-instances --instance-ids $SPOT_INST_ID
