if [ "$STY" = "" ]
then
    echo "not in screen! please start a screen session to run this script"
    exit 1
fi


# options: dpp_ham, dpp_cos, dpp_l2, dpp_random, random, bayes_opt
# options: reg, reg_half_lr, reg_bad_lr, arch
SRCH_TPE="mixed_dpp_rbf"
ITERS="20"
SPACE="reg_bad_lr"

NUM_INST=100
SPOT_REQUEST_ID=`aws ec2 request-spot-instances --spot-price "2.69" --instance-count $NUM_INST --type "one-time" --launch-specification file://specification.json | grep SpotInstanceRequestId | awk '{print $2}' | sed s/,// | sed s/\"// | sed s/\"//`


####
# to get info about the spot bid:
WAIT_SECONDS=5
while true; do 
    SPOT_INST_ID=`aws ec2 describe-spot-instance-requests --spot-instance-request-ids $SPOT_REQUEST_ID | grep InstanceId | awk '{print $2}' | sed s/,// | sed s/\"// | sed s/\"//`
    NUM_IDS=`echo ${SPOT_INST_ID} | wc -w`
    if [ "${NUM_IDS}" == "${NUM_INST}" ]; then
    #if [ ! -z "$SPOT_INST_ID" ]; then
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
    NUM_IPS=`echo ${SPOT_IP} | wc -w`
    if [ "${NUM_IPS}" == "${NUM_INST}" ]; then
	echo "successfully got ip address: $SPOT_IP"
	break
    else
	echo "waiting $WAIT_SECONDS second(s) to get the IP address"
	sleep $WAIT_SECONDS
    fi
done


while true; do
    NUM_LINES_PASSED_INIT=`aws ec2 describe-instance-status --instance-ids $SPOT_INST_ID | grep "\"Status\": \"passed\"" | wc -l`
    if [ "$NUM_LINES_PASSED_INIT" == "$(($NUM_INST * 2))" ]; then 
	break
    else
	echo "waiting $WAIT_SECONDS second(s) for instance to pass initialization. requires both system status checks and instance status checks to finish. currenty $(( ${NUM_LINES_PASSED_INIT} / 2 )) have passed."
	sleep $WAIT_SECONDS
    fi
done




###
# gets the current instance's ip address
CUR_IP=`curl -s http://169.254.169.254/latest/meta-data/public-ipv4`


# options: dpp_ham, dpp_cos, dpp_l2, dpp_random, random, bayes_opt
# options: reg, reg_half_lr, reg_bad_lr, arch
SRCH_TPE="mixed_dpp_rbf"
ITERS="2"
SPACE="reg_bad_lr"


###
# train models and move
COUNTER=0
for ONE_SPOT_IP in ${SPOT_IP}; do
    echo "About to try $ONE_SPOT_IP, with COUNTER=${COUNTER}"

    COMMANDS=""
    COMMANDS="${COMMANDS} source activate arkcat;"
    COMMANDS="${COMMANDS} cd /home/ec2-user/projects/hyperopt;"
    COMMANDS="${COMMANDS} git fetch;"
    COMMANDS="${COMMANDS} git reset --hard origin/master;"
    COMMANDS="${COMMANDS} rm /home/ec2-user/projects/hyperopt/hyperopt/*.pyc;"
    COMMANDS="${COMMANDS} cd /home/ec2-user/projects/dpp_mixed_mcmc;"
    COMMANDS="${COMMANDS} git fetch;"
    COMMANDS="${COMMANDS} git reset --hard origin/master;"
    COMMANDS="${COMMANDS} cd /home/ec2-user/projects/ARKcat/src;"
    COMMANDS="${COMMANDS} git fetch;"
    COMMANDS="${COMMANDS} git reset --hard origin/master;"
    COMMANDS="${COMMANDS} bash train_and_eval_spot.sh ${SRCH_TPE} 0${COUNTER} ${CUR_IP} ${ITERS} ${SPACE}"
    COMMANDS="${COMMANDS} bash /home/ec2-user/projects/ARKcat/aws/kill_this_instance.sh;"


    ssh -i "/home/ec2-user/projects/ARKcat/aws/jesse-key-pair-uswest2.pem" -oStrictHostKeyChecking=no ec2-user@ec2-${ONE_SPOT_IP}.us-west-2.compute.amazonaws.com ${COMMANDS} &
    let COUNTER+=1
done



###
# train models and move 
#ssh -i "/home/ec2-user/projects/ARKcat/aws/jesse-key-pair-uswest2.pem" -oStrictHostKeyChecking=no ec2-user@ec2-${SPOT_IP}.us-west-2.compute.amazonaws.com "source activate arkcat; cd /home/ec2-user/projects/hyperopt; git checkout hyperopt/dpp.py; git pull; cd /home/ec2-user/projects/ARKcat/src; git checkout train_and_eval_pinot.sh; git pull; bash train_and_eval_spot.sh ${1} ${2} $CUR_IP $SPOT_INST_ID ${3} ${4}"


###
# to ssh into this machine, I had to go to the console, click on instances, scroll down to security groups, click default, click actions, click edit inbound rules, and change the All TCP source to Anywhere


###
# aws ec2 terminate-instances --instance-ids $SPOT_INST_ID
