for i in 14 16 18 20 22;
do j=$((2**$i));
echo $j $(/usr/bin/time -v ./benchmark $j 0 1 2>&1|grep 'Maximum resident set size (kbytes):'|awk '{print $6;}');
done > mem_2

for i in 14 16 18 20 22;
do j=$((2**$i));
echo $j $(/usr/bin/time -v ./benchmark $j 1 0 2>&1|grep 'Maximum resident set size (kbytes):'|awk '{print $6;}');
done > mem_1
