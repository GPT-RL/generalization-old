#! /usr/bin/env bash
if [ -z ${1} ]
then
  cat << EOF
usage: ./deploy.sh script_name
  script_name: name of script (contained in notebooks/) to open in Pluto.jl
EOF
  exit
fi

export SCRIPT="$1"
docker-compose build
docker-compose up -d
docker logs -f dashboard_pluto-dashboard_1
