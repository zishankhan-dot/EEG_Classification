import mne
from mne.datasets import eegbci
import tempfile, os

#function to take uploaded file (browser memory)
#### and store it into disk since mne require file from disk
def load_raw(uploaded_file):
    #saving tmp file in disk
    tmp=tempfile.NamedTemporaryFile(suffix=".edf",delete=False)
    tmp.write(uploaded_file.read())
    tmp.close()

    #bring back that temp file to mne
    raw=mne.io.read_raw_edf(tmp.name,preload=True)
    os.unlink(tmp.name)

    #edf files from EEG dataset have different ch_names 
    #updating it from C2. -> C2 
    Channel_names={ch:ch.replace(".","") for ch in raw.ch_names}
    raw.rename_channels(Channel_names)

    return raw

def load_demo(subject=29, run=4):
    data_path = os.path.join(tempfile.gettempdir(), 'mne_data')
    os.makedirs(data_path, exist_ok=True)
    paths = eegbci.load_data(subject, runs=[run], path=data_path, update_path=False, verbose=False)
    raw = mne.io.read_raw_edf(paths[0], preload=True, verbose=False)
    Channel_names = {ch: ch.replace(".", "") for ch in raw.ch_names}
    raw.rename_channels(Channel_names)
    return raw

def get_filteredBoth(raw):
    # using alpha (8-13) and beta(13-30) bands
    raw_filtered=raw.copy().filter(8.0,30.0)
    return raw_filtered;

def get_fileteredBeta(raw):
    raw_filtered=raw.filter(13.0,30.0)
    return raw_filtered;
