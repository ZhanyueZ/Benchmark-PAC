### USAGE:
#### 1. run the docker containers under the /container 
#### 2. cd PAC-DP
#### 3. make
#### 4. see the options in the makefile. One example (2 warm up iter/ 10 normal iter/ 2MB weights/activations. #of DPUs/Tasklets are default): 

    ./bin/host_code -w 2 -e 10 -i 262144
    