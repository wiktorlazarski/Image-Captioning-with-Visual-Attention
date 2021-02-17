#!/bin/bash

IP_ADDRESS=$(hostname -I | sed s/\ .*//)

echo "---> Server IPv4 address in private network is $IP_ADDRESS <---"

gunicorn -w 1 -b 0.0.0.0:1234 --log-level ERROR API.server:app