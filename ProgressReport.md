progress-report ->

Downloading Data:
created a data downloading script ./scripts/download_data.py which downloads the data from the provided link and saves it in the ../data/raw/ directory. for the purpose of this project i have targeted first 20 subjects and 3 recordings for each subject(04,08,12).

Data Exploration And Analysis:
Then, I created a data loading script ./src/data.py, which loads the data using mne library and performs some basic preprocessing steps such as filtering the data between 0.5 and 40 Hz. and comparing the raw and filtered data by plotting three graphs one for raw data (time domain) and another for power spectral density (frequency domain) and the last one with color map(c3,c4,cz) to understand how c3,c4 and cz are behaving opposite to each other.
