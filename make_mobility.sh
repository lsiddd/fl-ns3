cd BonnMotion/bin/
./bm -f campus ManhattanGrid -n 100 -x 1000 -y 1000 -s 100 -d 1000
./bm NSFile -f campus
./bm CSVFile -f campus

mv campus* ../..
cd ../..

