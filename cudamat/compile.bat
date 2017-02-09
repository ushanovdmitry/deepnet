call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat"
call nvcc -O3 -v -o libcudamat.dll      --shared cudamat.cu cudamat_kernels.cu            -lcublas
call nvcc -O3 -v -o libcudamat_conv.dll --shared cudamat_conv.cu cudamat_conv_kernels.cu  -lcublas

