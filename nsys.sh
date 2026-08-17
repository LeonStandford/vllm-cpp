rm -rf report1.nsys-rep
rm -rf report1.sqlite
./test.sh
nsys profile ./build/tiny-vllm --output="report1.nsys-rep"
nsys stats report1.nsys-rep