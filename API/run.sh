#!/bin/bash

IP_ADDRESS=$(hostname -I | sed s/\ .*//)

echo "---> Server Network address in private network is $IP_ADDRESS:1234 <---"

gunicorn -w 1 -b 0.0.0.0:1234 --log-level ERROR API.server:app
