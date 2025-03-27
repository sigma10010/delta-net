# GazeCapture.tar -> gc_tar
tar -xf GazeCapture.tar -C gc_tar

# gc_tar -> gc
python extract_tar.py

# gc -> gc_processed
python prepareDataset.py
