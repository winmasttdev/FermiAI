#!/bin/bash
exec ssh -o StrictHostKeyChecking=no -R 80:localhost:8090 serveo.net
