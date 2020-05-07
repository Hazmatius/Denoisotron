#!/bin/bash
docker run -it -v /Users:/Users -v /$PWD:/App -v /Volumes:/Volumes --name=estimator denoiser:latest bash