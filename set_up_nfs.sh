#!/bin/bash

systemctl daemon-reload
sudo mount -t nfs -o vers=4 192.168.1.10:/backups /mnt/data
