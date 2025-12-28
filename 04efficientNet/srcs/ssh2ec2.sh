#!/bin/bash

_loghome="$HOME/sshlogs"
_prefix='aws'
_hostip='54.223.40.69'
_datestr=$(date +'%y%m%dt%H%M%S')
_keyfile='~/.aws/sshPemkey/aws-labelme-ec.pem'
_keyfile='~/.ssh/aws-labelme-ec.pem'

echo "param Number:${#}"
_ip=$(aws ec2 describe-instances --instance-id i-0eb714f9141b66c65 --region cn-northwest-1| jq -r '.. | .PublicIpAddress? //empty')
echo "auto fetch:train machine IP=${_ip}"
if test $# -gt 0; then
        _hostip=${1:-127}
else
	_hostip=${_ip:-127}
fi

echo "${_datestr} ${_hostip}" >> ~/ssh2ec2.ip.history
_logfileName="ssh_${_prefix}${_hostip}-${_datestr}.log"
_logF="${_loghome}/${_logfileName}"
mkdir -p ${_loghome}

echo "start at:[`date +'%y%m%dt%H%M%S'`" |tee -a ${_logF}
ssh -Y -i "${_keyfile}"  -o ServerAliveInterval=3 ubuntu@${_hostip} |tee -a ${_logF}
echo "end at:[`date +'%y%m%dt%H%M%S'`" |tee -a ${_logF}
