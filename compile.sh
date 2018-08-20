rm *.o
rm cgsro_multicore
rm cgsro_tesla
#rm cgsro_tesla_p100

# IMPORTANT:
# 1. export LD_LIBRARY_PATH=/opt/pgi/linux86-64/2017/cuda/9.0/lib64
# 2. Make sure that CUDA_PATH is defined properly 

# CPU: target = multicore
pgc++ -o cgsro_multicore     -fast -acc -ta=multicore  main.cpp  cgsro.cpp helpers.cpp cgsro_sequential.cpp cgsro_multicore.cpp cgsro_gpu.cpp

# GPU: target = Tesla K40
pgc++ -o cgsro_tesla     -g -w -fast -acc -ta=tesla:cc35,lineinfo -Minfo=accel,ccff -Mlarge_arrays  main.cpp cgsro.cpp helpers.cpp cgsro_sequential.cpp cgsro_multicore.cpp cgsro_gpu.cpp 

# GPU: target = Tesla P100
#pgc++ -o cgsro_tesla_p100   -I./opt/pgi/linux86-64/2017/cuda/9.0/include -L/opt/pgi/linux86-64/2017/cuda/9.0/lib64 -lcudart  -g  -w -fast -acc -ta=tesla:cc60,lineinfo  -Minfo=accel,ccff -Mlarge_arrays  main.cpp cgsro.cpp helpers.cpp cgsro_sequential.cpp  cgsro_multicore.cpp cgsro_gpu.cpp 


# How to run:
#./cgsro_* rows cols ro_steps target

# CPU (multicore): 
#./cgsro_multicore 100000 100 1 1

# GPU (Tesla K40):
#./cgsro_tesla 100000 100 1 2
