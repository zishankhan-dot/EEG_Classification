import mne
import numpy as np

def get_epochs(raw_filtered,pick_sensors):
    events,eventid=mne.events_from_annotations(raw_filtered)
    #creating epochs using annotations- dividing epochs on basis of 4sec window
    #along with annotations ex t1-(with every 4sec epochs)
    epochs=mne.Epochs(
        raw_filtered,
        events,
        event_id=eventid,
        tmin=0,
        tmax=4,
        picks=pick_sensors,
        baseline=None,
        preload=True
    )
    return epochs,eventid

def get_psd_condition(epochs,eventid):
    result={}
    for cond in eventid:
        if str(cond) == 'T0':
            continue
        psd=epochs[cond].compute_psd(fmin=8,fmax=30)
        power,freq=psd.get_data(return_freqs=True)
        avg_power=np.mean(power,axis=0) # 3d -> 2d (points_annotations, 3 channels, frequency within range(8-30))
        result[cond]=(psd.ch_names,avg_power,freq)
    return result


