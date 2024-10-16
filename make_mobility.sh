./bm -f campus ManhattanGrid -n 100 -x 100 -y 100 -d 1000
./bm NSFile -f campus
./bm CSVFile -f campus

mv campus* ../..
cd ../..

